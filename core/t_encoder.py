from dataclasses import dataclass
import numpy as np
from pprint import pprint
from queue import Queue
import subprocess
from threading import Event
import torch
from torch import Tensor
from typing import Callable

from media.media import FShape, MediaInfo, VideoInfo
from media.utils import VideoPipeInfo
from media.encoder import EncoderSettings, encoder_subprocess

from utils.p_print import *

from .types import BaseThread, NnFrame
from .dh_transfers import dtoh_transfer_torch
from .torch_tensor import tensor_to_img


@dataclass(slots=True)
class EncoderThreadSettings:
    video_pipe_info: VideoPipeInfo
    video_info: VideoInfo
    e_settings: EncoderSettings
    in_media_info: MediaInfo



class EncoderThread(BaseThread):
    def __init__(
        self,
        settings: EncoderThreadSettings,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._encoded: int = 0
        self._stop_event: Event = Event()
        self.in_queue: Queue = Queue(3)

        self.vpi: VideoPipeInfo = settings.video_pipe_info

        self.sub_process: subprocess.Popen = encoder_subprocess(
            video_pipe_info=self.vpi,
            video_info=settings.video_info,
            e_settings=settings.e_settings,
            in_media_info=settings.in_media_info,
            debug=debug
        )


    @property
    def encoded(self) -> int:
        return self._encoded


    def run(self) -> None:
        verbose: bool = self.verbose

        # Create a cuda stream and allocate Host memory
        cuda_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        host_mem: Tensor = torch.empty(
            self.vpi.nbytes,
            dtype=torch.uint8,
            pin_memory=True
        )

        # Output stream
        img_shape: FShape = self.vpi.shape
        img_dtype: np.dtype = self.vpi.dtype
        img_nbytes = self.vpi.nbytes

        # Tensor to image
        flip_r_b: bool = bool(self.vpi.c_order != 'rgb')

        # Flow control
        in_queue: Queue = self.in_queue
        received: int = 0
        remaining: int = self.vpi.nframes

        with torch.cuda.stream(cuda_stream):
            while (
                not self._stop_event.is_set()
                and remaining > 0
            ):

                # Wait for a frame or a poison pill
                input = in_queue.get(block=True)
                if input is None or self._stop_event.is_set():
                    if verbose:
                        print(purple("[V][E] Received Null tensor"))

                    self.end_encoding(writer_subprocess)
                    writer_subprocess = None
                    break

                frame: NnFrame = input

                # Get tensor from frame
                if verbose:
                    print(
                        purple(f"[V][E] Received no. {received}:"),
                        f"{frame.tensor.shape}, {frame.tensor.dtype}, {frame.tensor.shape}"
                    )

                d_tensor = frame.tensor

                d_img: np.ndarray = tensor_to_img(
                    tensor=d_tensor,
                    dtype=stdin_dtype,
                    flip_r_b=flip_r_b,
                )

                out_img: np.ndarray = dtoh_transfer_torch(
                    dtoh_mem=dtoh_mem,
                    d_img=d_img,
                    out_dtype=stdin_dtype,
                    cuda_stream=cuda_stream,
                ).reshape(d_img.shape)

                writer_subprocess.stdin.write(f.img)
                out_i += 1
                remaining -= 1
                sent = 1

                del frame.tensor
                if self.progress_thread is not None:
                    self.progress_thread.put(sent)
                self._encoded += sent

        #     print(red(f"[V][E] Error while executing: "), " ".join(encoder_command))
        self._processing = False
        print(
            purple(f"[V][E] Encoded all frames"),
            f"{self._encoded}",
            flush=True
        )
        self.end_encoding(writer_subprocess)
        # print(purple(f"[V][E] ended"))


    def end_encoding(self, sub_process: subprocess.Popen| None) -> bool:
        # Close output video
        if sub_process is not None:
            stderr_bytes: bytes | None = None
            stdout_bytes: bytes | None = None
            if sub_process is not None:
                try:
                    # Arbitrary timeout value
                    stdout_bytes, stderr_bytes = sub_process.communicate(input='', timeout=60)
                except:
                    sub_process.kill()
                    pass
            if stdout_bytes is not None:
                pprint(stdout_bytes.decode('utf-8)'))

            if stderr_bytes is not None:
                std_str = stderr_bytes.decode('utf-8)')
                # TODO: parse the output file
                for l in std_str:
                    if not l.startswith("x265 [info]"):
                        print(l.strip())
        return True


    def stop(self, force: bool=False) -> None:
        self._stop_event.set()
        if force:
            while not self.in_queue.empty():
                self.in_queue.get_nowait()
        self.put_frame(None)
        self._processing = False


    def put_frame(self, frame: NnFrame) -> bool:
        if self._processing:
            self.in_queue.put(frame)
        return True


    @property
    def encoded(self) -> int:
        return self._encoded

    def set_progress_callback(self, function: Callable) -> None:
        self.callback = function


