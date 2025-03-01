from __future__ import annotations
import time
from queue import Queue
import torch
from torch import Tensor
from typing import TYPE_CHECKING
from gpu_filters.gpu_resize import gpu_resize_, gpu_resize_to_

from pynnlib import (
    Idtype,
    NnModel,
)
from utils.p_print import *
if TYPE_CHECKING:
    from .t_cuda_inference import CudaInferenceThread
    from .types import NnFrame
    from .t_decoder import DecoderThread
    from .t_encoder import EncoderThread



def initialize_filter_inference(
    self: CudaInferenceThread,
    model: NnModel | None,
    device: str = "cuda:0",
    dtype: Idtype = 'fp16',
) -> None:
    """Initialize internale variables, models, ...
    """

    # self.cuda_session = nnlib.session(model)
    self.infer_stream= torch.cuda.Stream(device)
    # self.cuda_session.initialize(
    #     device=device,
    #     dtype=dtype,
    #     warmup=False,
    # )
    pass


@torch.inference_mode()
def perform_filter_inference(self: CudaInferenceThread, verbose: bool = False):
    if verbose:
        print(cyan(f"[V][I][FILTER] Cuda InferenceThread"))
    in_queue: Queue = self.in_queue

    d_thread: DecoderThread = self.producer
    e_thread: EncoderThread = self.consumer

    cuda_stream = self.infer_stream
    # scale = session.model.scale

    with torch.cuda.stream(cuda_stream):
        while not self._stop_event.is_set():
            if verbose:
                print(cyan("[V][I][FILTER] waiting"))
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
                        interpolation_method="bicubic"
                    )
                else:
                    gpu_resize_(
                        frame=frame,
                        scale_factor=self.prescale,
                        interpolation_method="bicubic"
                    )

            # Input tensor
            in_tensor = frame.tensor
            n, c, in_h, in_w = in_tensor.shape
            # in_tensor = in_tensor.to(dtype=...)
            if verbose:
                print(blue(f"{verbose_prefix} in tensor dtype:{in_tensor.dtype}"))

            # Perform simple inference

            out_tensor: Tensor

            frame.tensor = torch.clamp(out_tensor, 0., 1.)

            time.sleep(0.0001)
            cuda_stream.synchronize()

            e_thread.put_frame(frame)
            d_thread.set_produce_flag()

    if verbose:
        print(cyan(f"[V][I][FILTER] Ended"))
