from __future__ import annotations
import warnings
import numpy as np
from pprint import pprint
import subprocess
from threading import Event, Lock
import torch
from torch import Tensor

from media import FShape
from media.decoder import decoder_subprocess
from media.utils import VideoPipeInfo
from pynnlib import (
    img_to_tensor,
    np_dtype_to_torch,
    Idtype,
    IdtypeToTorch,
)
from .types import BaseThread, NnFrame
from .dh_transfers import htod_transfer

from utils.p_print import *

warnings.filterwarnings("ignore", category=UserWarning, message=".*non-writable tensors.*")



class DecoderThread(BaseThread):
    def __init__(
        self,
        video_pipe_info: VideoPipeInfo,
        device: str = "cuda:0",
        tensor_dtype: torch.dtype | Idtype = 'fp32',
        name: str | None = None,
        debug: bool = False,
    ) -> None:
        """Create a thread which purpose is to get frames from the encoder
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

        self.vpi: VideoPipeInfo = video_pipe_info
        self.sub_process: subprocess.Popen = decoder_subprocess(
            vpi=self.vpi, debug=debug
        )

        self.device: str = device
        self.tensor_dtype: torch.dtype = (
            IdtypeToTorch[tensor_dtype]
            if not isinstance(tensor_dtype, torch.dtype)
            else tensor_dtype
        )


    @property
    def decoded(self) -> int:
        return self._decoded


    @torch.inference_mode()
    def run(self) -> None:

        if self.consumer is None:
            raise ValueError(red("[E] No consumer defined for the decoder."))

        # Create a cuda stream and allocate Host memory
        cuda_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        host_mem: Tensor = torch.empty(
            self.vpi.nbytes,
            dtype=torch.uint8,
            pin_memory=True
        )

        # Input stream
        img_shape: FShape = self.vpi.shape
        img_dtype: torch.dtype = np_dtype_to_torch.get(self.vpi.dtype, self.vpi.dtype)
        img_nbytes = self.vpi.nbytes

        # Image to tensor
        tensor_dtype = self.tensor_dtype
        flip_r_b: bool = bool(self.vpi.c_order != 'rgb')

        # Flow control
        remaining: int = self.vpi.nframes
        f_no: int = self.vpi.f_no
        f_index: int = 0

        with torch.cuda.stream(cuda_stream):
            while (
                not self._stop_event.is_set()
                and remaining > 0
            ):
                img_buffer: Tensor = torch.frombuffer(
                    self.sub_process.stdout.read(img_nbytes),
                    dtype=torch.uint8,
                )

                # Wait until resource (GPU) is available
                self._lock.acquire(blocking=True)
                if self._stop_event.is_set():
                    return

                if remaining < 0 or img_buffer is None:
                    print(yellow("remaining < 0 or img_buffer is None"))
                    remaining = 0
                    self.release()
                    break

                # HtoD transfer
                d_img: Tensor = htod_transfer(
                    host_mem=host_mem,
                    img_buffer=img_buffer,
                    img_dtype=img_dtype,
                    img_shape=img_shape,
                    cuda_stream=cuda_stream
                )

                # Image to 4D tensor
                d_tensor: Tensor = img_to_tensor(
                    d_img=d_img,
                    tensor_dtype=tensor_dtype,
                    flip_r_b=flip_r_b,
                )

                # Create a frame object
                frame: NnFrame = NnFrame(
                    f_no=f_no,
                    tensor=d_tensor.clone(),
                    last=bool(remaining == 0)
                )
                if self.verbose:
                    print(
                        f"[V][D] ({lightgreen(f_index)}) ({f_no}), {remaining}. Tensor:",
                        f"{d_tensor.shape}, {d_tensor.dtype}"
                    )

                # Send the frame to the consumer
                self.consumer.put_frame(frame)

                remaining -= 1
                f_index += 1
                f_no += 1

            # except:
            #     if verbose:
            #         print(lightgreen(f"[V][D] Encountered end of file or error. Decoded {f_no} frames"))
        if self.verbose:
            print(lightgreen(f"[V][D] End of decoding"))


    def stop(self, force: bool=False) -> None:
        self._stop_event.set()

        if force:
            self.release()
            try:
                self.sub_process.kill()
            except:
                pass


    def release(self) -> None:
        try:
            self._lock.release()
        except:
            pass


    def set_produce_flag(self) -> None:
        self.release()

