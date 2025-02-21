import numpy as np
from torch import Tensor
from media.images_io import write_image
from pynnlib import tensor_to_img


def write_tensor(filepath: str, d_tensor: Tensor) -> None:
    """ Save a 4D tensor as an image". SYnchronous operation. Slow
    """
    d_img: Tensor = tensor_to_img(
        tensor=d_tensor,
        dtype=np.uint8,
        flip_r_b=True,
    )
    h_img: np.ndarray = d_img.to("cpu").numpy()
    write_image(filepath, h_img)
