from __future__ import annotations
import queue
from queue import Queue
from threading import Event
import torch
from pynnlib import(
    Idtype,
    TrtModel,
    TensorRtSession,
)
from utils.p_print import *
from .types import NnFrame, BaseThread
from .trt_inference import (
    initialize_trt_inference,
    perform_trt_inference,
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
        self.in_queue: Queue = Queue(1)
        self.verbose = debug
        self.prescale: list[int, int, int] | None = None
        self.infer_stream: torch.cuda.Stream = None


    def initialize(
        self,
        model: TrtModel,
        device: str = "cuda:0",
        dtype: Idtype = 'fp16',
        prescale: list[int, int, int] | float | None = None,
    ):
        """Isolate for fps measurement"""

        self.trt_session: TensorRtSession
        initialize_trt_inference(
            self,
            model=model,
            device=device,
            dtype=dtype
        )
        if isinstance(prescale, list | tuple):
            self.prescale = (prescale[0], prescale[1])
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
