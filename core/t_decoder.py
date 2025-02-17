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
from .dh_transfers import (
    htod_transfer,
    img_to_tensor,
)
from media import FShape

from media.decoder import VideoStreamInfo, decoder_subprocess
from .cp_utils import allocate_memory
from .types import BaseThread

from utils.p_print import *





class DecoderThread(BaseThread):
    def __init__(
        self,
        vsi: VideoStreamInfo,
        device: str = "cuda:0",
        tensor_dtype: torch.dtype = torch.float32,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._decoded: int = 0
        self._stop_event: Event = Event()
        self._lock: Lock = Lock()
        self._lock.acquire(blocking=False)

        self.vsi: VideoStreamInfo = vsi
        self.sub_process: subprocess.Popen = decoder_subprocess(
            vsi=self.vsi, debug=debug
        )
        self.tensor_dtype: torch.dtype = tensor_dtype


    def set_inference_threads(self, i_threads: dict[str, InferenceThread]):
        self.i_threads = i_threads


    @property
    def decoded(self) -> int:
        return self._decoded


    def run(self) -> None:
        verbose: bool = False

        # Create a cuda stream and allocate Host memory
        cuda_stream = cp.cuda.stream.Stream(non_blocking=True)
        htod_mem = allocate_memory(
            shape=(self.vsi.nbytes, 1, 1),
            dtype=self.vsi.dtype,
            stream=cuda_stream
        )

        remaining: int = self.vsi.nframes

        # Input stream
        img_shape: FShape = self.vsi.shape
        pipe_dtype: np.dtype = self.vsi.dtype
        pipe_img_nbytes = self.vsi.nbytes

        f_no: int = 0
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
                    tensor_dtype=(
                        cp.float16 if session.fp16 else cp.float32
                    ),
                    flip_r_b=flip_r_b,
                )


                self.tensor_dtype


                # d_tensor = cp.clip(d_tensor + 0.2, 0, 1.)

                # Create a frame object
                frame: NnFrame = NnFrame(
                    f_no=f_no,
                    tensor=d_tensor,
                    scene=scene,
                    src_scene=src_scene,
                    # TODO remove deepcopy
                    sequence=deepcopy(filter_steps),
                    step_no=start_step_no,
                    encoder_step_no=encoder_step_no,
                )

                # set last flag to inform the temporal inference
                # that it's the latest one and it has to continue producing frames
                # by itself
                if rem_src_scene == 0 and remaining == 1:
                    # print(yellow(f"Last frame: {frame.f_no}"))
                    frame.last = True

                # Wait -> run other threads
                time.sleep(0.00001)
                cuda_stream.synchronize()

                # Execute synchronous filters.
                # TODO: later, use a separate thread for this
                # Apply cuda filters until next inference or decoder
                apply_cuda_filters_(
                    frame,
                    verbose=verbose,
                    cuda_stream=cuda_stream
                )
                if verbose:
                    print(
                        f"[V][D] ({lightgreen(f_index)}) ({in_f_no}), {remaining}. Tensor:",
                        f"{d_tensor.shape}, {d_tensor.dtype}"
                    )
                    print(f"[V][D]     Step no.:{frame.step_no}, {frame.sequence[frame.step_no]}")

                # Skip following model if inpainting and dsr
                # patch_next_step_no_(frame=frame, model_manager=model_manager)

                # Send the frame to the consumer
                if frame.step_no >= frame.encoder_step_no:
                    consumer = e_thread
                else:
                    consumer = self.i_threads[
                        model_manager.xprovider(frame.sequence[step_no])
                    ]
                    consumer.put_frame(frame)
                remaining -= 1

                f_index += 1

                # All frames of the scene have been decoded and sent to consumer
                if to_produce == 0:
                    break

            # except:
            #     if verbose:
            #         print(lightgreen(f"[V][D] Encountered end of file or error. Decoded {f_no} frames"))

            print(lightgreen(f"[V][D] End of src scene {src_scene['no']}"))
            print(lightgreen(f"[V][D] End of scene {scene['no']}"))
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

