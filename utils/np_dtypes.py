import numpy as np


def np_to_float32(img: np.ndarray) -> np.ndarray:
    """Convert a np array from uint8 to float
        without normalizing it
    """
    if img.dtype == np.float32:
        return img
    div = float(np.iinfo(img.dtype).max)
    img = img.astype(np.float32)
    img /= div
    return img



def np_to_float16(img: np.ndarray) -> np.ndarray:
    """Convert a np array from uint8 to float
        without normalizing it
    """
    if img.dtype == np.float16:
        return img
    div = float(np.iinfo(img.dtype).max)
    img = img.astype(np.float16)
    img /= div
    return img




def np_to_uint16(img: np.ndarray) -> np.ndarray:
    """Convert a np array from float to uint16
        without normalizing it
        warning: slow because no in-place clipping
    """
    if img.dtype == np.float32:
        # return (img.clip(0, 1) * 65535).astype(np.uint16)
        return (img * 65535).astype(np.uint16)

    if img.dtype == np.uint8:
        return (img * 257).astype(np.uint16)

    elif img.dtype == np.uint16:
        return img

    raise NotImplementedError(f"Cannot convert {img.dtype} np.array to uint16")



def np_to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a np array from float to uint8
        without normalizing it
        warning: slow because no in-place clipping
    """
    if img.dtype == np.float32:
        # return (img.clip(0, 1) * 255).astype(np.uint8)
        return (img * 255).astype(np.uint8)

    elif img.dtype == np.uint16:
        return (img // 257).astype(np.uint8)

    if img.dtype == np.uint8:
        return img

    raise NotImplementedError(f"Cannot convert {img.dtype} np.array to uint8")


