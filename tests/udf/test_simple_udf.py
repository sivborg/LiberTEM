import collections
import functools 

import numpy as np

from libertem.common.buffers import BufferWrapper
from libertem.api import Context
from libertem.udf.stddev import merge, batch_merge, compute_batch, batch_buffer
from utils import MemoryDataSet, _mk_random

def test_sum_frames(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16),
                            partition_shape=(4, 4, 16, 16), sig_dims=2)

    def my_buffers():
        return {
            'pixelsum': BufferWrapper(
                kind="nav", dtype="float32"
            )
        }

    def my_frame_fn(frame, pixelsum):
        pixelsum[:] = np.sum(frame)

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))


def test_3d_ds(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            partition_shape=(4, 16, 16), sig_dims=2)

    def my_buffers():
        return {
            'pixelsum': BufferWrapper(
                kind="nav", dtype="float32"
            )
        }

    def my_frame_fn(frame, pixelsum):
        pixelsum[:] = np.sum(frame)

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(1, 2)))


def test_minibatch(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16),
                            partition_shape=(4, 4, 16, 16), sig_dims=2)

    # res = lt_ctx.run_udf(
    #     dataset=dataset,
    #     fn=my_frame_fn_batch,
    #     make_buffers=my_buffer_batch,
    #     merge=stddev_merge,
    # )

    # assert 'batch' in res

    # N = data.shape[2] * data.shape[3]
    # assert res['batch'].data[:, :, 2][0][0] == N # check the total number of frames 

    # assert np.allclose(res['batch'].data[:, :, 1], np.sum(data, axis=(0, 1))) # check sum of frames

    # sum_var = np.var(data, axis=(0, 1))
    # assert np.allclose(sum_var, res['batch'].data[:, :, 0]/N) # check sum of variances

    bts = lt_ctx.run_udf(
        dataset=dataset,
        fn=compute_batch, 
        make_buffers=functools.partial(
            batch_buffer,
        ),
        merge=batch_merge,
    )

    # assert 'batch_buffer' in bts
    # assert 'sum_frame' in bts
    # assert 'num_frame' in bts

