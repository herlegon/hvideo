from __future__ import annotations
from enum import IntEnum
from pprint import pprint
from queue import Queue
from typing import Literal, TYPE_CHECKING
import torch
from torch import Tensor
from torch import nn

from gpu_filters.gpu_resize import gpu_resize_, gpu_resize_to_
from utils.p_print import *
from pynnlib import (
    NnModel,
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



class RVRTFrameCache():
    ############################## ADAPT to this use case

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



def initialize_rvrt_inference(
    self: CudaTemporalInferenceThread,
    model: PyTorchModel | None,
    device: str = "cuda:0",
    dtype: Idtype = 'fp16',
) -> None:
    self.model = model
    self.infer_stream = torch.cuda.Stream(device)

    # total number of frames
    num_frame_testing = args.tile[0]
    # tile_overlap[0] = 2
    num_frame_overlapping = 2
    stride = num_frame_testing - num_frame_overlapping
    b, d, c, h, w = lq.size()
    d_idx_list = (
        list(range(0, d-num_frame_testing, stride))
        + [max(0, d-num_frame_testing)]
    )

    # E is
    E = torch.zeros(b, d, c, h, w)
    W = torch.zeros(b, d, 1, 1, 1)

    module: nn.Module = model.module
    lq: Tensor = torch.empty((1,1,1,1,1))
    for d_idx in d_idx_list:
        # create a window
        lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]

        # Inference
        modulo = 8
        _, _, _, h_old, w_old = lq_clip.size()
        h_pad = (modulo - h_old % modulo) % modulo
        w_pad = (modulo - w_old % modulo) % modulo
        lq_clip = torch.cat([lq_clip, torch.flip(lq_clip[:, :, :, -h_pad:, :], [3])], 3) if h_pad else lq_clip
        lq_clip = torch.cat([lq_clip, torch.flip(lq_clip[:, :, :, :, -w_pad:], [4])], 4) if w_pad else lq_clip
        out_clip: Tensor = module(lq_clip).detach().cpu()
        out_clip = out_clip[:, :, :, :h_old, :w_old]

        out_clip_mask = torch.ones(
            (b, min(num_frame_testing, d), 1, 1, 1)
        )
        E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
        W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)


    out_clip = E.div_(W)


    self.cache = RVRTFrameCache(window_size=)




@torch.inference_mode()
def perform_rvrt_inference(
    self: CudaTemporalInferenceThread,
    verbose: bool = False
) -> None:

    cache: RVRTFrameCache = self.cache

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

            # Append fram to cache
            cache.append(frame=frame)

            # Do not process if cache is not ready
            if not cache.is_ready():
                d_thread.release()
                continue

            # Cache is ready, perform inference
            window = cache.get_window()
            if window is None:
                raise ValueError("Not enough frames in window, why?")

            # out_tensor = inference(window)
            out_tensor: Tensor = torch.empty_like(NnFrame.tensor)

            # get frame to output
            out_frame = cache.current_frame()
            out_frame.tensor = out_tensor

            e_thread.put(out_frame)

            if cache.is_empty():
                break

    # Send a poison pill
    e_thread.put(None)
