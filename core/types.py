from __future__ import annotations
import abc
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread
from typing import Any, Iterable, Mapping, Type, TYPE_CHECKING
import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from .t_progress import ProgressThread



@dataclass(slots=True)
class NnFrame:
    f_no: int = 0
    tensor: Tensor | np.ndarray = None
    channel_last: bool = False

    # last flag is used by temporal models
    last: bool = False



class BaseThread(abc.ABC, Thread):
    def __init__(self,
        group: None = None,
        target: Callable[..., object] | None = None,
        name: str | None = None,
        args: Iterable[Any] = ...,
        kwargs: Mapping[str, Any] | None = None,
        *,
        daemon: bool | None = None
    ) -> None:
        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self._has_error: bool = False
        self._error_msg: str = ''
        self._verbose: bool = False

        self.producer: Type[BaseThread] | None = None
        self.consumer: Type[BaseThread] | None = None
        self.progress_thread: ProgressThread | None  = None
        self._processing: bool = True
        self._is_cuda_workflow: bool = False


    @property
    def verbose(self) -> bool:
        return self._verbose


    @verbose.setter
    def verbose(self, enabled: bool) -> None:
        self._verbose = enabled


    def error_encountered(self) -> tuple[bool, str]:
        return (self._has_error, self._error_msg)


    @abc.abstractmethod
    def stop(self, force: bool=False) -> None:
        pass


    def set_producer(self, producer: Type[BaseThread]) -> None:
        self.producer = producer


    def set_consumer(self, consumer: Type[BaseThread]) -> None:
        self.consumer = consumer


    # @abc.abstractmethod
    def set_produce_flag(self) -> None:
        pass


    def put_tensor(self, frame: np.ndarray | torch.Tensor | cp.ndarray) -> bool:
        pass


    def put(self, data: Any, force: bool=False) -> None:
        pass


    def put_frame(self, frame: NnFrame) -> None:
        pass


    def set_progress_thread(self, progress_thread: ProgressThread) -> None:
        self.progress_thread = progress_thread


    def processing(self) -> bool:
        return self._processing


    @property
    def is_cuda_workflow(self) -> bool:
        return self._is_cuda_workflow


    @is_cuda_workflow.setter
    def is_cuda_workflow(self, enabled: bool) -> None:
        self._is_cuda_workflow = enabled and torch.cuda.is_available()
