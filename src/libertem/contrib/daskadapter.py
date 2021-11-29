from typing import Union, Iterable, List
import warnings

import numpy as np
import dask
import dask.array
from libertem.executor.delayed import DelayedJobExecutor

from libertem.io.dataset.base import DataSet
from libertem.udf.base import UDFRunner, UDF
from libertem.corrections import CorrectionSet


def make_dask_array(dataset, dtype='float32', roi=None):
    '''
    Create a Dask array using the DataSet's partitions as blocks.
    '''
    chunks = []
    workers = {}
    for p in dataset.get_partitions():
        d = dask.delayed(p.get_macrotile)(
            dest_dtype=dtype, roi=roi
        )
        workers[d] = p.get_locations()
        chunks.append(
            dask.array.from_delayed(
                d,
                dtype=dtype,
                shape=p.slice.adjust_for_roi(roi).shape,
            )
        )
    arr = dask.array.concatenate(chunks, axis=0)
    if roi is None:
        arr = arr.reshape(dataset.shape)
    return (arr, workers)


def task_results_array(
    dataset: DataSet,
    udf: Union[UDF, Iterable[UDF]],
    roi: np.ndarray = None,
    corrections: CorrectionSet = None,
    backends=None
) -> Union[dict, List[dict]]:
    '''
    Return UDF task results as Dask arrays.

    This function generates dask arrays from task results that are generated by
    running the specified UDFs using the
    :class:`~libertem.executor.delayed.DelayedJobExecutor` without merging or
    results computation. :code:`kind="nav"` buffer results are concatenated and
    other buffer results are stacked.

    Any merging or results computation is the responsibility of the user since
    LiberTEM depends on assignment into buffers for merging. This is not
    supported by Dask arrays at the time this is implemented.

    Please note that this is *not* the result that
    :meth:`libertem.api.Context.run_udf` returns since no merging or results
    computation is performed! Instead, this returns a collection of intermediate
    results.

    If you wish to obtain the same final result as running with
    :meth:`~libertem.api.Context.run_udf`, please check the implementation of
    merging and results computation for the specific UDFs you intend to run to
    implement an equivalent merging and results computation routine.

    For the special case of :code:`kind='nav'` buffers with default merge and
    without a ROI, the final result can be obtained just by reshaping axis 0 of
    the Dask array to match the dataset's nav shape.

    The parameters are equivalent to :meth:`libertem.api.Context.run_udf`.

    Please note that you should run one UDF at a time if you intend to
    call :code:`compute()` on individual UDF results.
    Running more than one UDF only makes sense if your final
    result is obtained with a single :code:`compute()`, i.e. if results from all
    UDFs are combined in one Dask task tree. With separate :code:`compute()` calls,
    all specified UDFs have to run each time.

    Returns
    -------
    Union[dict, List[dict]]
        List of result dictionaries for list of UDFs and
        single result dictionary for single UDFs with buffer names as keys and
        Dask arrays as values. Please note that the keys, shapes and values are
        often different from the normal UDF results, depending on the UDF's
        implementation for merging and results computation.
    '''
    udf_is_list = isinstance(udf, (tuple, list))
    if not udf_is_list:
        udfs = [udf]
    else:
        udfs = list(udf)

    if (roi is not None) and (roi.dtype is not np.dtype(bool)):
        warnings.warn(f"ROI dtype is {roi.dtype}, expected bool. Attempting cast to bool.")
        roi = roi.astype(bool)
    result_iter = UDFRunner(udfs).results_for_dataset_sync(
        dataset=dataset,
        executor=DelayedJobExecutor(),
        roi=roi,
        progress=False,
        corrections=corrections,
        backends=backends,
    )
    results = list(result_iter)
    # initialize with dicts of empty lists for
    # the buffers contained in the UDF results
    result_decls = [udf.get_result_buffers() for udf in udfs]
    resultbuffers = [{} for udf_index in range(len(udfs))]
    for udf_index in range(len(udfs)):
        for key, buf in result_decls[udf_index].items():
            if buf.use != 'result_only':
                resultbuffers[udf_index][key] = []

    # Append Dask array chunks for each result
    for result, task in results:
        for udf_index in range(len(udfs)):
            result_decl = result_decls[udf_index]
            for key, resultbuffer in resultbuffers[udf_index].items():
                buf = result_decl[key]
                buf.set_shape_partition(task.partition, roi=roi)
                resultbuffer.append(
                    dask.array.from_delayed(
                        getattr(result[udf_index], key),
                        shape=buf.shape,
                        dtype=buf.dtype
                    )
                )

    # Combine the individual chunks
    for udf_index in range(len(udfs)):
        result_decl = result_decls[udf_index]
        for key, resultbuffer in list(resultbuffers[udf_index].items()):
            buf = result_decl[key]
            # We have to concatenate and can't stack nav since
            # the nav size depends on the partition size
            if buf.kind == 'nav':
                resultbuffers[udf_index][key] = dask.array.concatenate(
                    resultbuffer,
                    axis=0
                )
            else:
                resultbuffers[udf_index][key] = dask.array.stack(
                    resultbuffer
                )
    if udf_is_list:
        return resultbuffers
    else:
        return resultbuffers[0]
