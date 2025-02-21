from dataclasses import dataclass
import math
import numpy as np
from pprint import pprint
from queue import Queue
import subprocess
from threading import Event
import torch
from torch import Tensor
from typing import Callable

from core.dh_transfers import dtoh_transfer
from media.media import FShape, MediaInfo, VideoInfo
from media.utils import VideoPipeInfo
from media.encoder import EncoderSettings, encoder_subprocess
from utils.p_print import *

from .types import BaseThread, NnFrame
from pynnlib import tensor_to_img, np_dtype_to_torch
from .dh_transfers import dtoh_transfer



class EncoderThread(BaseThread):
    def __init__(
        self,
        video_info: VideoInfo,
        video_pipe_info: VideoPipeInfo,
        e_settings: EncoderSettings,
        in_media_info: MediaInfo,
        device: str = "cuda:0",
        name: str | None = None,
        debug: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._encoded: int = 0
        self._stop_event: Event = Event()
        self.in_queue: Queue = Queue(3)

        self.vpi: VideoPipeInfo = video_pipe_info

        self.sub_process: subprocess.Popen = encoder_subprocess(
            video_pipe_info=self.vpi,
            video_info=video_info,
            e_settings=e_settings,
            in_media_info=in_media_info,
            debug=debug
        )
        self.device: str | torch.device = device


    @property
    def encoded(self) -> int:
        return self._encoded


    @torch.inference_mode()
    def run(self) -> None:
        verbose: bool = self.verbose

        # Output stream
        img_shape: FShape = self.vpi.shape
        img_dtype: np.dtype = self.vpi.dtype

        # Create a cuda stream and allocate Host memory
        cuda_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        host_mem: Tensor = torch.empty(
            img_shape,
            dtype=np_dtype_to_torch.get(img_dtype, img_dtype),
            pin_memory=True
        )

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

                    self.end_encoding(self.sub_process)
                    break

                frame: NnFrame = input
                d_tensor = frame.tensor

                # Get tensor from frame
                if verbose:
                    print(
                        purple(f"[V][E] Received no. {received}:"),
                        f"{d_tensor.shape}, {d_tensor.dtype}, {d_tensor.shape}"
                    )
                    received += 1

                d_img: Tensor = tensor_to_img(
                    tensor=d_tensor,
                    dtype=img_dtype,
                    flip_r_b=flip_r_b,
                )

                out_img: np.ndarray = dtoh_transfer(
                    host_mem=host_mem,
                    d_img=d_img,
                    cuda_stream=cuda_stream,
                )
                out_img = np.ascontiguousarray(out_img)

                if verbose:
                    print(
                        purple(f"[V][E] send to pipe:"),
                        f"{out_img.shape}, {out_img.dtype}"
                    )

                try:
                    self.sub_process.stdin.write(out_img)
                except:
                    pprint(self.sub_process.stderr.read())
                    break

                remaining -= 1
                sent = 1

                del frame.tensor
                if self.progress_thread is not None:
                    self.progress_thread.put(sent)
                self._encoded += sent

                if self.producer is not None:
                    self.producer.set_produce_flag()

        #     print(red(f"[V][E] Error while executing: "), " ".join(encoder_command))
        self._processing = False
        if verbose:
            print(
                purple(f"[V][E] All frames encoded or error"),
                f"{self._encoded}",
                flush=True
            )
        self.end_encoding()


    def end_encoding(self) -> bool:
        # Close output video
        stdout_bytes: bytes | None = None
        try:
            # Arbitrary timeout value
            stdout_bytes, _ = self.sub_process.communicate(
                input='', timeout=60
            )
        except:
            self.sub_process.kill()
            pass

        if stdout_bytes is not None:
            std_str = stdout_bytes.decode('utf-8)')
            # pprint(std_str)
            # TODO: parse the output file ?
            for l in std_str.split("\n"):
                if (
                    not l.startswith("x265 [info]")
                    and not l.startswith("frame=")
                ):
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


