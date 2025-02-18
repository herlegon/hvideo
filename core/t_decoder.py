from __future__ import annotations
from copy import deepcopy
import cupy as cp
import numpy as np
from pprint import pprint
import subprocess
from threading import Event, Lock
import time
import torch
from torch import Tensor

from media.utils import VideoPipeInfo
from .dh_transfers import (
    htod_transfer,
    img_to_tensor,
)
from media import FShape

from media.decoder import decoder_subprocess
from .cp_utils import allocate_memory
from .types import BaseThread, NnFrame

from utils.p_print import *





class DecoderThread(BaseThread):
    def __init__(
        self,
        video_pipe_info: VideoPipeInfo,
        tensor_dtype: torch.dtype = torch.float32,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._decoded: int = 0
        self._stop_event: Event = Event()
        self._lock: Lock = Lock()
        self._lock.acquire(blocking=False)

        self.vpi: VideoPipeInfo = video_pipe_info
        self.sub_process: subprocess.Popen = decoder_subprocess(
            vpi=self.vpi, debug=debug
        )
        self.tensor_dtype: torch.dtype = tensor_dtype


    @property
    def decoded(self) -> int:
        return self._decoded


    def run(self) -> None:
        verbose: bool = False

        # Create a cuda stream and allocate Host memory
        cuda_stream = cp.cuda.stream.Stream(non_blocking=True)
        htod_mem = allocate_memory(
            shape=(self.vpi.nbytes, 1, 1),
            dtype=self.vpi.dtype,
            stream=cuda_stream
        )

        remaining: int = self.vpi.nframes

        # Input stream
        img_shape: FShape = self.vpi.shape
        pipe_dtype: np.dtype = self.vpi.dtype
        pipe_img_nbytes = self.vpi.nbytes

        flip_r_b: bool = bool(self.vpi.c_order != 'rgb')

        f_no: int = self.vpi.f_no
        f_index: int = 0
        with cuda_stream:
            while (
                not self._stop_event.is_set()
                and remaining > 0
            ):
                img_buffer: np.ndarray = np.frombuffer(
                    self.sub_process.stdout.read(pipe_img_nbytes),
                    dtype=pipe_dtype,
                )
                remaining -= 1

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
                d_img: cp.ndarray = htod_transfer(
                    htod_mem=htod_mem,
                    img_buffer=img_buffer,
                    img_shape=img_shape
                )

                # cp.ndarray image to tensor
                d_tensor: Tensor = img_to_tensor(
                    d_img=d_img,
                    tensor_dtype=self.tensor_dtype,
                    flip_r_b=flip_r_b,
                )


                # d_tensor = cp.clip(d_tensor + 0.2, 0, 1.)

                # Create a frame object
                frame: NnFrame = NnFrame(
                    f_no=f_no,
                    tensor=d_tensor,
                    last=bool(remaining == 0)
                )

                # Wait -> run other threads
                time.sleep(0.00001)
                cuda_stream.synchronize()

                print(
                    f"[V][D] ({lightgreen(f_index)}) ({f_no}), {remaining}. Tensor:",
                    f"{d_tensor.shape}, {d_tensor.dtype}"
                )

                # Skip following model if inpainting and dsr
                # patch_next_step_no_(frame=frame, model_manager=model_manager)

                # Send the frame to the consumer
                self.consumer.put_frame(frame)

                remaining -= 1
                f_index += 1
                f_no += 1


            # except:
            #     if verbose:
            #         print(lightgreen(f"[V][D] Encountered end of file or error. Decoded {f_no} frames"))

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

