# HtoD and DtoH transfers
import math
import time
import cupy as cp
import numpy as np
import torch
from torch import (
    Tensor,
    from_dlpack,
    to_dlpack,
)

from .torch_tensor import (
    flip_r_b_channels,
    to_hwc,
    to_nchw,
    np_dtype_to_torch,
)
from .cp_utils import (
    HostDeviceMemory,
    MemcpyKind,
)



def htod_transfer(
    htod_mem: HostDeviceMemory,
    img_buffer: np.ndarray,
    img_shape: tuple[int, int, int],
) -> Tensor:
    """Asynchronous copy from Host to Device
    """

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
    d_torch_tensor: Tensor = from_dlpack(d_cp_tensor.toDlpack())
    return d_torch_tensor



def dtoh_transfer(
    dtoh_mem: HostDeviceMemory,
    d_img: Tensor,
    out_dtype: np.dtype,
    cuda_stream,
) -> np.ndarray:
    """Asynchronous copy from Device to Host
    """

    out_shape: tuple[int, int, int] = d_img.shape
    d_cp_img = cp.from_dlpack(to_dlpack(d_img))
    d_cp_img = cp.ravel(d_cp_img)

    cp.cuda.runtime.memcpyAsync(
        int(dtoh_mem.host),
        d_cp_img.data.ptr,
        d_cp_img.nbytes,
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
    d_img: Tensor,
    dtype: torch.dtype,
    flip_r_b: bool = False,
) -> Tensor:
    """ Create a 4D tensor from a 3D image (tensor type), reshaped and normalized
    """
    d_tensor: Tensor = d_img
    img_dtype: torch.dtype = d_img.dtype
    if flip_r_b:
        d_tensor = flip_r_b_channels(d_tensor)
    d_tensor = to_nchw(d_tensor)

    divisor: float = (
        float(torch.iinfo(img_dtype).max)
        if dtype != img_dtype
        else 1.
    )

    if divisor != 1.:
        d_tensor = d_tensor.to(dtype=torch.float32) / divisor

    d_tensor = d_tensor.to(dtype=dtype)
    return d_tensor.contiguous()



def tensor_to_img(
    tensor: Tensor,
    dtype: np.dtype,
    flip_r_b: bool = False,
) -> Tensor:

    torch_ndarray: Tensor = to_hwc(tensor)
    if flip_r_b:
        torch_ndarray = flip_r_b_channels(torch_ndarray)

    torch_ndarray = torch.clamp(torch_ndarray, 0, 1.0)

    multiplier: float = 1.
    tensor_dtype: torch.dtype = tensor.dtype
    if tensor_dtype != dtype:
        num = (
            float(np.iinfo(dtype).max)
            if tensor_dtype in (torch.float32, torch.float16, torch.bfloat16)
            else 1.
        )
        denum = (
            float(np.iinfo(tensor_dtype).max)
            if dtype == np.float32
            else 1.
        )
        multiplier: float = num / denum

    if multiplier != 1.:
        torch_ndarray = torch_ndarray.to(dtype=torch.float32, copy=False)
        torch_ndarray = torch_ndarray * multiplier

    torch_ndarray = torch_ndarray.to(dtype=np_dtype_to_torch[dtype])

    return torch_ndarray


