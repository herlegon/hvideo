from __future__ import annotations
from enum import IntEnum
from pprint import pprint
from queue import Queue
import time
from typing import Literal, TYPE_CHECKING
import torch
from torch import Tensor

from gpu_filters.gpu_resize import gpu_resize_, gpu_resize_to_
from media.images_io import write_tensor
from utils.p_print import *
from pynnlib import (
    PyTorchModel,
    Idtype,
)

if TYPE_CHECKING:
    from core.types import NnFrame
    from core.t_cuda_temp_inference import CudaTemporalInferenceThread
    from core.types import NnFrame
    from core.t_decoder import DecoderThread
    from core.t_encoder import EncoderThread



class _CACHE_STATE(IntEnum):
    INIT = 0x00
    WAIT_FRAMES = 0x01
    PROCESSING = 0x02
    EMPTYING = 0x03
    EMPTY = 0x04



class TemporalFrameCache():

    def __init__(self, window_size: int = 5) -> None:
        self.window_size: int = window_size
        self.frames: list[NnFrame] = []
        self.tensors: list[Tensor] = []
        self.state: _CACHE_STATE = _CACHE_STATE.INIT


    def reset(self) -> None:
        self.frames.clear()
        self.tensors.clear()
        self.state = _CACHE_STATE.INIT


    def emptying(self) -> bool:
        return bool(self.state == _CACHE_STATE.EMPTYING)


    def is_ready(self) -> bool:
        return bool(
            self.state in (_CACHE_STATE.PROCESSING, _CACHE_STATE.EMPTYING)
        )


    def is_empty(self) -> bool:
        return bool(
            self.state in (_CACHE_STATE.EMPTY, _CACHE_STATE.INIT)
        )


    def append(self, frame: NnFrame | None) -> None:
        if frame is None:
            return

        if self.state == _CACHE_STATE.INIT:
            self.tensors = [frame.tensor] * (self.window_size // 2 + 1)
            self.frames = [frame]
            self.state = _CACHE_STATE.WAIT_FRAMES

        elif self.state == _CACHE_STATE.WAIT_FRAMES:
            self.tensors.append(frame.tensor)
            self.frames.append(frame)
            if len(self.tensors) >= self.window_size:
                self.state = _CACHE_STATE.PROCESSING

        elif self.state == _CACHE_STATE.PROCESSING:
            if len(self.frames) >= self.window_size:
                raise ValueError("Too many frames in cache")
            if len(self.tensors) >= self.window_size:
                raise ValueError("too many tensors")

            self.tensors.append(frame.tensor)
            self.frames.append(frame)
            if frame.last:
                self.state = _CACHE_STATE.EMPTYING

        else:
            raise ValueError(red("cache: append while not ready to receive"))


    def get_window(self) -> list[Tensor] | None:
        if self.state == _CACHE_STATE.PROCESSING:
            return self.tensors

        elif self.state == _CACHE_STATE.EMPTYING:
            to_add: int = (self.window_size - len(self.tensors))
            for _ in range(to_add):
                self.tensors.append(self.tensors[-1])
            return self.tensors

        return None


    def current_frame(self) -> NnFrame | None:
        if _CACHE_STATE.PROCESSING <= self.state < _CACHE_STATE.EMPTY:
            frame: NnFrame = self.frames.pop(0)
            frame.tensor = None
            self.tensors.pop(0)
            if len(self.frames) == 0:
                self.state = _CACHE_STATE.EMPTY
            return frame
        return None



def initialize_temporal_inference(
    self: CudaTemporalInferenceThread,
    model: PyTorchModel | None,
    device: str = "cuda:0",
    dtype: Idtype = 'fp16',
) -> None:
    self.model = model
    self.cache = TemporalFrameCache()
    self.infer_stream = torch.cuda.Stream(device)



@torch.inference_mode()
def perform_temporal_inference(
    self: CudaTemporalInferenceThread,
    verbose: bool = False
) -> None:

    cache: TemporalFrameCache = self.cache

    if verbose:
        print(cyan(f"[V][I][FILTER] Cuda InferenceThread"))
    in_queue: Queue = self.in_queue

    d_thread: DecoderThread = self.producer
    e_thread: EncoderThread = self.consumer

    cuda_stream = self.infer_stream

    with torch.cuda.stream(cuda_stream):
        while not self._stop_event.is_set():
            if verbose:
                print(cyan("[V][I][FILTER] waiting"))

            frame = None
            if not cache.emptying():
                # receive frame from consumer
                input = in_queue.get(block=True)
                if input is None or self._stop_event.is_set():
                    break
                frame: NnFrame = input

                if verbose:
                    verbose_prefix: str = f"[V][I][FILTER][{frame.f_no}]"
                # Resize before inference
                if self.prescale is not None:
                    if isinstance(self.prescale, list | tuple):
                        gpu_resize_to_(
                            frame=frame,
                            out_size=self.prescale,
                            interpolation_method="bilinear"
                        )
                    else:
                        gpu_resize_(
                            frame=frame,
                            scale_factor=self.prescale,
                            interpolation_method="bilinear"
                        )

                print(purple(f"received frame no.{frame.f_no}"), "last" if frame.last else "")

            else:
                print(red("emptying"))

            # Append fram to cache
            cache.append(frame=frame)

            # Do not process if cache is not ready
            if not cache.is_ready():
                print("  ask for frame")
                d_thread.set_produce_flag()
                continue

            # Cache is ready, perform inference
            window = cache.get_window()
            if window is None:
                raise ValueError("Not enough frames in window, why?")

            # inference
            out_tensor = (
                window[0] * 0.5
                + window[1] * 0.75
                + window[2]
                + window[3] * 0.75
                + window[4] * 0.5
            ) / 3.5
            out_tensor = torch.clamp_(out_tensor, 0., 1.)

            time.sleep(0.0001)
            cuda_stream.synchronize()

            # get frame to output
            out_frame = cache.current_frame()
            out_frame.tensor = out_tensor

            # Write (debug)
            write_tensor(f"frame_{out_frame.f_no:02}.png", d_tensor=out_tensor)

            e_thread.put_frame(out_frame)
            print(yellow(f"output:"), out_frame.f_no)

            if cache.is_empty():
                break

            if not cache.emptying():
                d_thread.set_produce_flag()


    # Send a poison pill
    e_thread.put(None)
