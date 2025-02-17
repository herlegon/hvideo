# HtoD and DtoH transfers
import math
import time
import cupy as cp
import numpy as np
from .cp_utils import HostDeviceMemory
from .cp_tensor import (
    flip_r_b_channels,
    MemcpyKind,
    to_hwc,
    to_nchw,
)


def htod_transfer(
    htod_mem: HostDeviceMemory,
    img_buffer: np.ndarray,
    img_shape: tuple[int, int, int],
) -> cp.ndarray:
    h_mem: np.ndarray = np.frombuffer(
        htod_mem.host,
        dtype=img_buffer.dtype,
        count=math.prod(img_buffer.shape)
    )
    np.copyto(h_mem, img_buffer)
    d_cp_tensor: cp.ndarray = cp.ndarray(
        h_mem.shape,
        dtype=h_mem.dtype,
        memptr=htod_mem.device.data
    )
    d_cp_tensor.set(h_mem)
    d_cp_tensor.astype(img_buffer.dtype)
    d_cp_tensor = d_cp_tensor.reshape(img_shape)
    return d_cp_tensor



def dtoh_transfer(
    dtoh_mem: HostDeviceMemory,
    d_img: cp.ndarray,
    out_dtype: np.dtype,
    cuda_stream,
) -> np.ndarray:
    out_shape: tuple[int, int, int] = d_img.shape

    d_img = cp.ravel(d_img)

    cp.cuda.runtime.memcpyAsync(
        int(dtoh_mem.host),
        d_img.data.ptr,
        d_img.nbytes,
        MemcpyKind.DeviceToHost,
        cuda_stream.ptr
    )
    time.sleep(0.0001)
    cuda_stream.synchronize()

    out_img: np.ndarray = np.frombuffer(
        dtoh_mem.host,
        dtype=out_dtype,
        count=math.prod(out_shape)
    )

    return out_img



def img_to_tensor(
    d_img: cp.ndarray,
    tensor_dtype: cp.dtype,
    flip_r_b: bool = False,
) -> cp.ndarray:
    """ Create a tensor, reshaped and normalized
    """
    d_cp_tensor: cp.ndarray = d_img
    img_dtype: cp.dtype = d_img.dtype
    if flip_r_b:
        d_cp_tensor = flip_r_b_channels(d_cp_tensor)
    d_cp_tensor = to_nchw(d_cp_tensor)

    divisor: float = (
        float(np.iinfo(img_dtype).max)
        if tensor_dtype != img_dtype
        else 1.
    )

    if divisor != 1.:
        d_cp_tensor = d_cp_tensor.astype(cp.float32)
        d_cp_tensor /= divisor
    d_cp_tensor = d_cp_tensor.astype(tensor_dtype)

    return cp.ascontiguousarray(d_cp_tensor)



def tensor_to_img(
    tensor: cp.ndarray,
    out_dtype: cp.dtype | np.dtype,
    flip_r_b: bool = False,
) -> cp.ndarray:

    cp_ndarray: cp.ndarray = to_hwc(tensor)
    if flip_r_b:
        cp_ndarray = flip_r_b_channels(cp_ndarray)

    cp_ndarray = cp.clip(cp_ndarray, 0, 1.0, cp_ndarray)

    multiplier: float = 1.
    tensor_dtype: cp.dtype = tensor.dtype
    if tensor_dtype != out_dtype:
        num = (
            float(np.iinfo(out_dtype).max)
            if tensor_dtype in (np.float32, np.float16)
            else 1.
        )
        denum = (
            float(np.iinfo(tensor_dtype).max)
            if out_dtype == np.float32
            else 1.
        )
        multiplier: float = num / denum

    if multiplier != 1.:
        cp_ndarray = cp_ndarray.astype(cp.float32, copy=False)
        cp_ndarray = cp_ndarray * multiplier
    cp_ndarray = cp_ndarray.astype(out_dtype, copy=False)

    return cp_ndarray


