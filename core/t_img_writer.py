import math
import numpy as np
from pprint import pprint
from queue import Queue
from threading import Event
import torch
from torch import Tensor

from pynnlib import tensor_to_img
from core.dh_transfers import dtoh_transfer
from media.images_io import write_image
from utils.p_print import *
from .types import BaseThread, NnFrame
from .dh_transfers import dtoh_transfer


# Save as 8-bit png


class ImgWriterThread(BaseThread):
    def __init__(
        self,
        name: str | None = None,
        device: str = "cuda:0",
        filepaths: str = "",
    ) -> None:
        super().__init__(name=name)
        self._written: int = 0
        self._stop_event: Event = Event()
        self.in_queue: Queue = Queue(3)
        self.device: str | torch.device = device
        self.image_count: int = 0
        self.filepaths: str = filepaths


    @property
    def written(self) -> int:
        return self._written


    @property
    def encoded(self) -> int:
        # Use this for compatibility with EncoderThread
        # i.e. use the same main loop
        return self._written


    @torch.inference_mode()
    def run(self) -> None:
        verbose: bool = self.verbose

        # Create a cuda stream
        cuda_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        host_mem: Tensor | None = None

        # Flow control
        in_queue: Queue = self.in_queue
        received: int = 0
        remaining: int = self.image_count
        index: int = 0

        with torch.cuda.stream(cuda_stream):
            while (
                not self._stop_event.is_set()
                and remaining > 0
            ):
                # Wait for a frame or a poison pill
                input = in_queue.get(block=True)
                if input is None or self._stop_event.is_set():
                    if verbose:
                        print(purple("[V][IW] Received Null tensor"))
                    break

                frame: NnFrame = input
                d_tensor = frame.tensor

                # Get tensor from frame
                if verbose:
                    print(
                        purple(f"[V][IW] Received no. {received}:"),
                        f"{d_tensor.shape}, {d_tensor.dtype}, {d_tensor.shape}"
                    )
                    received += 1

                d_img: Tensor = tensor_to_img(
                    tensor=d_tensor,
                    dtype=np.uint8,
                    flip_r_b=True,
                )

                # Use the same pinned memory if possible
                if host_mem is None or host_mem.shape != d_img.shape:
                    del host_mem
                    host_mem: Tensor = torch.empty(
                        math.prod(d_img.shape),
                        dtype=torch.uint8,
                        pin_memory=True
                    )

                out_img: np.ndarray = dtoh_transfer(
                    host_mem=host_mem,
                    d_img=d_img,
                    cuda_stream=cuda_stream,
                )

                write_image(self.filepaths[frame.f_no], out_img)

                remaining -= 1
                index += 1
                sent = 1

                del frame.tensor
                if self.progress_thread is not None:
                    self.progress_thread.put(sent)
                self._written += sent
                print(f"written: {self.written}")

                if self.producer is not None:
                    self.producer.set_produce_flag()

        self._processing = False
        print(
            purple(f"[V][IW] all images written"),
            f"{self._written}",
            flush=True
        )


    def stop(self, force: bool=False) -> None:
        self._stop_event.set()
        if force:
            while not self.in_queue.empty():
                self.in_queue.get_nowait()
        self.put_frame(None)
        self._processing = False


    def put_frame(self, frame: NnFrame) -> bool:
        if self._processing:
            self.in_queue.put(frame)
        return True


