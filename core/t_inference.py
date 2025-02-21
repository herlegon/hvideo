from __future__ import annotations
import queue
from queue import Queue
from threading import Event
from media.media import VideoInfo
from pynnlib import(
    Idtype,
    NnModel,
    nnlib,
    NnFrameworkType,
    TensorRtSession,
)
from utils.p_print import *
from .types import NnFrame, BaseThread
from .trt_inference import (
    perform_trt_inference,
    initialize_trt_inference,
)


class InferenceThread(BaseThread):
    def __init__(
        self,
        name: str | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._stop_event = Event()
        self._stop_event.clear()
        self.in_queue: Queue = Queue(2)
        self.verbose = debug
        self.prescale: list[int, int, int] | None = None


    def initialize(
        self,
        filepath: str,
        device: str = "cuda:0",
        dtype: Idtype = 'fp16',
        prescale: list[int, int, int] | float | None = None,
    ):
        """Isolate for fps measurement"""
        trt_model: NnModel = nnlib.open(filepath, device)
        if trt_model.framework.type != NnFrameworkType.TENSORRT:
            raise ValueError(red(f"[E] {filepath} is not a TensorRT engine"))

        self.trt_session: TensorRtSession
        initialize_trt_inference(
            self,
            model=trt_model,
            device=device,
            dtype=dtype
        )
        if isinstance(prescale, list | tuple):
            self.prescale = (prescale[1], prescale[0])
        else:
            self.prescale = prescale


    def run(self):
        perform_trt_inference(self, self.verbose)

        while not self.in_queue.empty():
            self.in_queue.get_nowait()


    def stop(self, force: bool=False) -> None:
        self._stop_event.set()
        if force:
            while not self.in_queue.empty():
                self.in_queue.get_nowait()
        self.put_frame(None)


    def put_frame(self, frame: NnFrame) -> bool:
        try:
            self.in_queue.put(frame)
            return True
        except queue.Full:
            print("[V][I] no place")
            pass
        return False


    def ended(self) -> bool:
        return self.in_queue.empty()
