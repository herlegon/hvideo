from __future__ import annotations
from enum import IntEnum
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from utils.p_print import *

if TYPE_CHECKING:
    from core.types import NnFrame



class _CACHE_STATE(IntEnum):
    INIT = 0x00
    WAIT_FRAMES = 0x01
    PROCESSING = 0x02
    EMPTY = 0x03



class SegmentationFrameCache():

    def __init__(self, window_size: int = 5) -> None:
        self.window_size: int = window_size
        self.frames: list[NnFrame] = []
        self.tensors: list[Tensor] = []
        self.state: _CACHE_STATE = _CACHE_STATE.INIT


    def reset(self) -> None:
        self.frames.clear()
        self.tensors.clear()
        self.state = _CACHE_STATE.INIT


    def processing(self) -> bool:
        return bool(self.state == _CACHE_STATE.PROCESSING)


    def is_ready(self) -> bool:
        return bool(self.state == _CACHE_STATE.PROCESSING)


    def is_empty(self) -> bool:
        return bool(
            self.state in (_CACHE_STATE.EMPTY, _CACHE_STATE.INIT)
        )


    def append(self, frame: NnFrame | None) -> None:
        if frame is None:
            return

        if self.state == _CACHE_STATE.INIT:
            self.tensors = [frame.tensor]
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
            raise ValueError(red("Error cannot append new frame while in processing"))

        else:
            raise ValueError(red("cache: append while not ready to receive"))


    def get_all_tensors(self) -> list[Tensor] | None:
        # print(red(f"get_window: {self.state}, len={len(self.tensors)}"))
        if self.state == _CACHE_STATE.PROCESSING:
            return self.tensors
        return None


    def current_frame(self) -> NnFrame | None:
        if _CACHE_STATE.PROCESSING <= self.state < _CACHE_STATE.EMPTY:
            frame: NnFrame = self.frames.pop(0)
            frame.tensor = None
            if len(self.frames) == 0:
                self.state = _CACHE_STATE.EMPTY
            return frame
        return None

