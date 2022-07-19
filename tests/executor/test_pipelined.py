import time
import random
from typing import Generator, Optional, TypeVar

import pytest
import numpy as np

from libertem.api import Context
from libertem.common.executor import WorkerQueueEmpty
from libertem.udf.sum import SumUDF
from libertem.executor.pipelined import PipelinedExecutor, _order_results
from libertem.executor import pipelined
from libertem.udf import UDF


@pytest.fixture(scope="module")
def pipelined_ex():
    executor = None
    try:
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
            # to prevent issues in already-pinned situations (i.e. containerized
            # environments), don't pin our worker processes in testing:
            pin_workers=False,
            cleanup_timeout=0.5,
        )
        yield executor
    finally:
        if executor is not None:
            executor.close()


def test_pipelined_executor(pipelined_ex):
    executor = pipelined_ex
    ctx = Context(executor=executor)
    udf = SumUDF()
    data = np.random.randn(4, 4, 128, 128)
    ds = ctx.load("memory", data=data)
    res = ctx.run_udf(dataset=ds, udf=udf)
    assert np.allclose(
        data.sum(axis=(0, 1)),
        res['intensity'].data,
    )


def test_run_function(pipelined_ex):
    assert pipelined_ex.run_function(lambda: 42) == 42


class RaisesUDF(UDF):
    def __init__(self, exc_cls=Exception):
        super().__init__(exc_cls=exc_cls)

    def get_result_buffers(self):
        return {
            "stuff": self.buffer(kind='nav'),
        }

    def process_frame(self, frame):
        raise self.params.exc_cls("what")


def test_udf_exception_queued(pipelined_ex):
    executor = pipelined_ex
    ctx = Context(executor=executor)

    data = np.random.randn(16, 16, 128, 128)
    ds = ctx.load("memory", data=data, num_partitions=16)

    error_udf = RaisesUDF()  # raises an error
    with pytest.raises(RuntimeError):  # raised by executor as expected
        ctx.run_udf(dataset=ds, udf=error_udf)

    # Fails immediately on run_udf because queue is in bad state
    normal_udf = SumUDF()
    ctx.run_udf(dataset=ds, udf=normal_udf)
    # Error is raised during the task dispatch loop when we check if any tasks
    # completed yet


def test_default_spec():
    # make sure `.make_local` works:
    executor = None
    try:
        executor = PipelinedExecutor.make_local()

        # to at least see that something works:
        assert executor.run_function(lambda: 42) == 42
    finally:
        if executor is not None:
            executor.close()


def test_make_with():
    with Context.make_with("pipelined") as ctx:
        assert ctx.executor.run_function(lambda: 42) == 42


_STOP = object()

T = TypeVar('T')


def echo() -> Generator[Optional[T], T, None]:
    last = None
    while last is not _STOP:
        print(f"yielding {last}")
        last = yield last
        print(f"got {last}")
        if last is None:
            import traceback
            traceback.print_stack()


def test_order_results_in_order():
    r1 = object()
    t1 = object()
    t1id = 0

    r2 = object()
    t2 = object()
    t2id = 1

    r3 = object()
    t3 = object()
    t3id = 2

    # results come in as triples (result, task, task_id)
    def _trace_1():
        yield (r1, t1, t1id)
        yield (r2, t2, t2id)
        yield (r3, t3, t3id)

    # _order_results discards the task_id at the end
    ordered = _order_results(_trace_1())
    assert next(ordered) == (r1, t1)
    assert next(ordered) == (r2, t2)
    assert next(ordered) == (r3, t3)


def test_order_results_missing_task():
    r1 = object()
    t1 = object()
    t1id = 0

    r3 = object()
    t3 = object()
    t3id = 2

    # results come in as triples (result, task, task_id)
    def _trace_1():
        yield (r1, t1, t1id)
        yield (r3, t3, t3id)

    # _order_results discards the task_id at the end
    ordered = _order_results(_trace_1())
    assert next(ordered) == (r1, t1)
    with pytest.raises(RuntimeError):
        next(ordered)


def test_order_results_postponed_task():
    r1 = object()
    t1 = object()
    t1id = 0

    r2 = object()
    t2 = object()
    t2id = 1

    r3 = object()
    t3 = object()
    t3id = 2

    # results come in as triples (result, task, task_id)
    def _trace_1():
        yield (r1, t1, t1id)
        yield (r3, t3, t3id)
        yield (r2, t2, t2id)

    # _order_results discards the task_id at the end
    ordered = _order_results(_trace_1())
    assert next(ordered) == (r1, t1)
    assert next(ordered) == (r2, t2)
    assert next(ordered) == (r3, t3)


def test_run_function_failure(pipelined_ex):
    def _f():
        raise Exception("this fails to run")

    with pytest.raises(RuntimeError) as ex_info:
        pipelined_ex.run_function(_f)

    assert ex_info.match("^failed to run function: this fails to run$")


def test_run_function_error():
    executor = None
    try:
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
            pin_workers=False,
        )

        def _break(a, b, c):
            raise RuntimeError("stuff is broken, can't do it.")

        def _do_patch_worker():
            # XXX this completely destroys the workers ability to properly
            # function, but that's okay because it's a throwaway process pool
            # anyways:
            pipelined.worker_run_function = _break

        executor.run_each_worker(_do_patch_worker)
        with pytest.raises(RuntimeError) as e:
            executor.run_function(lambda: 42)
        assert e.match("failed to run function: stuff is broken, can't do it.")
    finally:
        if executor is not None:
            executor.close()


def _broken_pipelined_worker(queues, pin, spec, span_context):
    raise RuntimeError("stuff is broken, can't do it.")


def test_early_startup_error():
    """
    Simulate very early startup error, not even getting to the try/except
    that gives us error feedback.
    """
    executor = None

    # manual patching, we mock.patch doesn't work in multiprocessing
    # environments:
    original_pipelined_worker = pipelined.pipelined_worker
    try:
        pipelined.pipelined_worker = _broken_pipelined_worker
        with pytest.raises(WorkerQueueEmpty):
            executor = PipelinedExecutor(
                spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
                pin_workers=False,
                startup_timeout=0.4,
            )
    finally:
        pipelined.pipelined_worker = original_pipelined_worker
        if executor is not None:
            executor.close()


def _patch_setup_device():
    def _broken_setup_device(spec, pin):
        """
        Broken version of pipelined._setup_device for error injection
        """
        raise RuntimeError("stuff is broken, can't do it.")
    pipelined._setup_device = _broken_setup_device


def test_startup_error():
    """
    Simulate an error when starting up the worker, in this case we raise in _setup_device
    """
    executor = None
    try:
        with pytest.raises(RuntimeError) as e:
            executor = PipelinedExecutor(
                spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
                pin_workers=False,
                early_setup=_patch_setup_device,
            )
        assert e.match("error on startup: stuff is broken, can't do it.")
    finally:
        if executor is not None:
            executor.close()


class FailEventuallyUDF(UDF):
    def get_result_buffers(self):
        return {
            "stuff": self.buffer(kind="nav"),
        }

    def process_partition(self, partition):
        if random.random() > 0.50:
            time.sleep(0.1)
        raise Exception("stuff happens")


def test_failure_with_delay(pipelined_ex):
    ctx = Context(executor=pipelined_ex)
    udf = FailEventuallyUDF()
    data = np.random.randn(1, 32, 16, 16)
    ds = ctx.load("memory", data=data, num_partitions=32)
    with pytest.raises(RuntimeError) as e:
        ctx.run_udf(dataset=ds, udf=udf)
    assert e.match("failed to run tasks: stuff happens")


class SucceedEventuallyUDF(UDF):
    def get_result_buffers(self):
        return {
            "intensity": self.buffer(kind="nav"),
        }

    def process_partition(self, partition):
        if random.random() > 0.50:
            time.sleep(0.1)
        self.results.intensity[:] = np.sum(partition, axis=(-1, -2))


def test_success_with_delay(pipelined_ex):
    ctx = Context(executor=pipelined_ex)
    udf = SucceedEventuallyUDF()
    data = np.random.randn(1, 32, 16, 16)
    ds = ctx.load("memory", data=data, num_partitions=32)
    res = ctx.run_udf(dataset=ds, udf=udf)
    assert np.allclose(res['intensity'].data, np.sum(data, axis=(2, 3)))
