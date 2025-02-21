from __future__ import annotations
from pprint import pprint
from threading import Event, Lock
from warnings import warn
import torch
from torch import Tensor
from media.images_io import load_image
from pynnlib import (
    img_to_tensor,
)
from .types import BaseThread, NnFrame
from .dh_transfers import htod_transfer
from utils.p_print import *


class ImgReaderThread(BaseThread):
    def __init__(
        self,
        filepaths: list[str],
        device: str = "cuda:0",
        tensor_dtype: torch.dtype = torch.float32,
        name: str | None = None,
    ) -> None:
        """Create a thread which purpose is to load images
        transfer it into the GPU and convert to 4D tensors.

        args:
            device: GPU device
            tensor_dtype: the tensor will be cast to this dtype.
                        To maximize performance, use the dtype of the following filter
        """
        super().__init__(name=name)
        self._decoded: int = 0
        self._stop_event: Event = Event()
        self._lock: Lock = Lock()
        self._lock.acquire(blocking=False)

        self.device: str = device
        self.tensor_dtype: torch.dtype = tensor_dtype
        self.filepaths: list[str] = filepaths


    @property
    def decoded(self) -> int:
        return self._decoded


    @torch.inference_mode()
    def run(self) -> None:
        verbose: bool = self.verbose

        if self.consumer is None:
            raise ValueError(red("[E] No consumer defined for the decoder."))

        # Create a cuda stream
        cuda_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        host_mem: Tensor | None = None

        # Image to tensor
        tensor_dtype = self.tensor_dtype

        # Flow control
        remaining: int = len(self.filepaths)
        f_index: int = 0

        with torch.cuda.stream(cuda_stream):
            while (
                not self._stop_event.is_set()
                and remaining > 0
            ):
                # Read image
                in_fp: str = self.filepaths[f_index]
                try:
                    h_img: Tensor = torch.from_numpy(load_image(filepath=in_fp))
                except:
                    warn(yellow(f"Failed opening {in_fp}"))
                    remaining -= 1
                    f_index += 1

                if h_img.nbytes != host_mem.nbytes or host_mem is None:
                    del host_mem
                    host_mem = torch.empty(
                        h_img.nbytes,
                        dtype=torch.uint8,
                        pin_memory=True
                    )

                # Wait until resource (GPU) is available
                self._lock.acquire(blocking=True)
                if self._stop_event.is_set():
                    return

                if remaining < 0:
                    print(yellow("remaining < 0"))
                    remaining = 0
                    self.release()
                    break

                # HtoD transfer
                d_img: Tensor = htod_transfer(
                    host_mem=host_mem,
                    img_buffer=h_img,
                    img_dtype=h_img.dtype,
                    img_shape=h_img.shape,
                    cuda_stream=cuda_stream
                )

                # Image to 4D tensor
                d_tensor: Tensor = img_to_tensor(
                    d_img=d_img,
                    dtype=tensor_dtype,
                    flip_r_b=True,
                )

                # Create a frame object
                frame: NnFrame = NnFrame(
                    f_no=f_index,
                    tensor=d_tensor,
                    last=bool(remaining == 0)
                )

                print(
                    f"[V][IR] ({lightgreen(f_index)}), {remaining}. Tensor:",
                    f"{d_tensor.shape}, {d_tensor.dtype}"
                )

                # Send the frame to the consumer
                self.consumer.put_frame(frame)

                remaining -= 1
                f_index += 1


        print(lightgreen(f"[V][IR] End of decoding"))


    def stop(self, force: bool=False) -> None:
        self._stop_event.set()

        if force:
            self.release()


    def release(self) -> None:
        try:
            self._lock.release()
        except:
            pass


    def set_produce_flag(self) -> None:
        self.release()

