from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import glob
import multiprocessing
import os
from pathlib import Path
from pprint import pprint
import cv2
import numpy as np
import torch
from utils.np_dtypes import (
    np_to_float32,
    np_to_uint8,
)
from utils.path_utils import absolute_path
CPU_COUNT: int = multiprocessing.cpu_count()



def img_info(img: np.ndarray | torch.Tensor) -> str:
    if isinstance(img, torch.Tensor):
        return f"tensor: {img.shape}, {img.dtype}"

    h, w = img.shape[:2]
    range_str: str = f"[{np.min(img)} .. {np.max(img)}]"
    # range_str: str = (
    #     f"[{np.min(img)} .. {np.max(img)}]"
    #     if img.dtype in (np.uint8, np.uint16, np.uint32)
    #     else f"[{np.min(img):.02} .. {np.max(img):.02}]"
    # )
    return f"{w}x{h}, {img.dtype}, {range_str}"


def load_image(filepath: Path | str) -> np.ndarray:
    """Load an image as 8bit/16bits
    """
    return cv2.imdecode(
        np.fromfile(filepath, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED
    )


def load_image_fp32(filepath: Path | str) -> np.ndarray:
    return np_to_float32(
        cv2.imdecode(
            np.fromfile(filepath, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED
        )
    )


def write_image(filepath: Path | str, img: np.ndarray) -> None:
    # Support uint8 only as these functions aare used for debugging purpose
    # no nedd to improve this
    extension = os.path.splitext(filepath)[1]
    try:
        _, img_buffer = cv2.imencode(
            f".{extension}",
            np_to_uint8(img)
        )
        with open(filepath, "wb") as buffered_writer:
            buffered_writer.write(img_buffer)
    except Exception as e:
        raise RuntimeError(f"Failed to save image as {filepath}, reason: {type(e)}")


def load_images(
    filepaths: list[Path | str],
    cpu_count: int = 4,
) -> list[np.ndarray]:
    imgs: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=min(CPU_COUNT, cpu_count)) as executor:
        for img in executor.map(load_image_fp32, filepaths):
            imgs.append(img)
    return imgs



def write_images(
    filepaths: list[str],
    images: tuple[np.ndarray],
    cpu_count: int = 4,
) -> None:
    with ThreadPoolExecutor(max_workers=min(CPU_COUNT, cpu_count)) as executor:
        executor.map(write_image, filepaths, images)


def get_image_list(directory: str | Path, extension: str = '.png') -> list[str]:
    directory = os.path.normpath(
        os.path.realpath(absolute_path(str(directory)))
    )
    # fastest for simple filtering
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory)) if f.endswith(extension)
    ]

    # +50%
    files: list[str] = glob.glob(
        f"*{extension}",
        root_dir=directory,
        dir_fd=None,
        recursive=False,
        include_hidden=False
    )
    return [os.path.join(directory, f) for f in sorted(files)]
