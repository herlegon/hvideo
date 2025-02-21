from __future__ import annotations
import time
from queue import Queue
import torch
from typing import TYPE_CHECKING

from gpu_filters.gpu_resize import gpu_resize_to_

from .types import NnFrame
from pynnlib import (
    Idtype,
    nnlib,
    TrtModel,
)
from utils.p_print import *
if TYPE_CHECKING:
    from .t_inference import InferenceThread
from .t_decoder import DecoderThread
from .t_encoder import EncoderThread



def initialize_trt_inference(
    self: InferenceThread,
    model: TrtModel,
    device: str = "cuda:0",
    dtype: Idtype = 'fp16',
) -> None:

    # Get model and session
    self.trt_session = nnlib.session(model)
    self.trt_session.infer_stream = torch.cuda.Stream(device)
    self.trt_session.initialize(
        device=device,
        dtype=dtype,
        warmup=False,
    )


@torch.inference_mode()
def perform_trt_inference(self: InferenceThread, verbose: bool = False):
    print(cyan(f"[V][I][TRT] TensorRT InferenceThread"))
    in_queue: Queue = self.in_queue

    d_thread: DecoderThread = self.producer
    e_thread: EncoderThread = self.consumer

    session = self.trt_session
    context, engine = session.context, session.engine
    session_dtype: torch.dtype = session.dtype
    if session_dtype == torch.bfloat16:
        session_dtype = torch.float32
    cuda_stream = session.infer_stream

    scale = session.model.scale

    with torch.cuda.stream(cuda_stream):
        while not self._stop_event.is_set():
            if verbose:
                print(cyan("[V][I][TRT] waiting"))
            input = in_queue.get(block=True)
            if input is None or self._stop_event.is_set():
                break
            frame: NnFrame = input

            if verbose:
                verbose_prefix: str = f"[V][I][TRT][{frame.f_no}]"

            # Resize before inference
            if self.prescale is not None:
                gpu_resize_to_(
                    frame=frame,
                    out_size=self.prescale,
                    interpolation_method="bilinear"
                )

            # Input tensor
            in_tensor = frame.tensor
            n, c, in_h, in_w = in_tensor.shape
            in_tensor = in_tensor.to(dtype=session_dtype)
            in_tensor = torch.ravel(in_tensor)
            if verbose:
                print(blue(f"{verbose_prefix} in tensor dtype:{in_tensor.dtype}"))

            # Prepare output tensor in same device
            out_tensor_shape = (n, c, in_h * scale, in_w * scale)
            out_tensor: torch.Tensor = torch.empty(
                out_tensor_shape,
                dtype=session_dtype,
                device=in_tensor.device
            )

            # Perform simple inference
            bindings = [in_tensor.data_ptr(), out_tensor.data_ptr()]
            for i in range(engine.num_io_tensors):
                context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
            context.execute_async_v3(stream_handle=cuda_stream.cuda_stream)

            frame.tensor = torch.clamp_(out_tensor, 0., 1.)

            time.sleep(0.0001)
            cuda_stream.synchronize()

            e_thread.put_frame(frame)
            d_thread.set_produce_flag()

    print(cyan(f"[V][I][TRT] Ended"))
