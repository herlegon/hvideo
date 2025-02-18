from collections import deque
from dataclasses import dataclass
import cupy as cp
import os
import numpy as np
from pprint import pprint
from queue import Queue
import subprocess
import sys
from threading import Event
import torch
from torch import Tensor
from typing import Callable

from media.utils import VideoPipeInfo
from nn_inference.toolbox.dh_transfers import (
    dtoh_transfer,
    tensor_to_img,
)
from parsers import VideoSettings
from nn_inference.model_mgr import ModelManager
from nn_inference.resource_mgr import NnFrame, RlgThread
from nn_inference.toolbox.filtering import apply_cpu_filters_, get_encoder_ffmpeg_filters
from processing.effects import apply_effect, apply_upscale_effects
from scene.filters import do_watermark, get_ffmpeg_pad_filter, clean_fcomplex
from scene.src_scene import SrcScene
from utils.mco_types import McoFrame, Scene
from utils.mco_utils import ffmpeg_metadata
from utils.media import str_to_video_codec
from utils.p_print import *
from utils.path_utils import path_split
from utils.pxl_fmt import PIXEL_FORMAT
from utils.tools import ffmpeg_exe

from .cp_utils import allocate_memory
from .types import BaseThread, NnFrame

class EncoderThread(BaseThread):
    def __init__(
        self,
        video_pipe_info: VideoPipeInfo,
        debug: bool = False,
    ) -> None:
        super().__init__()
        self._stop_event: Event = Event()
        self.in_queue: Queue = Queue(3)
        self._encoded: int = 0

        self.vpi: VideoPipeInfo = video_pipe_info
        self.sub_process: subprocess.Popen = encoder_subprocess(
            vpi=self.vpi, debug=debug
        )


    @property
    def encoded(self) -> int:
        return self._encoded


    def run(self) -> None:
        verbose: bool = self.verbose

        # Create a cuda stream and allocate Host memory
        cuda_stream = cp.cuda.stream.Stream(non_blocking=True)
        dtoh_mem = allocate_memory(
            shape=(self.vpi.nbytes, 1, 1),
            dtype=self.vpi.dtype,
            stream=cuda_stream
        )

        flip_r_b: bool = bool(self.vpi.c_order != 'rgb')


        cuda_stream = self.model_manager.dtoh_cuda_stream
        dtoh_mem: HostDeviceMemory = self.model_manager.dtoh_mem

        in_queue: Queue = self.in_queue
        writer_subprocess: subprocess.Popen | None = None

        # Default settings
        stdin_pixfmt: str = 'rgb24'
        stdin_dtype: np.dtype = np.uint8
        flip_r_b: bool = False
        if verbose:
            print(purple("[V][E] Encoder, config:"))

        self._encoded: int = 0
        scene: Scene | None = None
        src_scene: SrcScene = None
        frame: NnFrame = None
        backward_imgs: deque = deque()
        received: int = 0
        # try:
        if True:
            with cuda_stream:
                while not self._stop_event.is_set():
                    # Wait for a frame or a poison pill
                    input = in_queue.get(block=True)
                    if input is None or self._stop_event.is_set():
                        if verbose:
                            print(purple("[V][E] Received Null tensor"))

                        if src_scene is not None and src_scene['backward']:
                            count = len(backward_imgs)
                            for _ in range(count):
                                writer_subprocess.stdin.write(backward_imgs.pop())

                        self.end_encoding(writer_subprocess)
                        writer_subprocess = None
                        break

                    frame: NnFrame = input

                    if scene is None:
                        # This is a different scene:
                        # - end the current encoder subprocess
                        # - start a new one
                        if src_scene is not None and src_scene['backward']:
                            count = len(backward_imgs)
                            for _ in range(count):
                                writer_subprocess.stdin.write(backward_imgs.pop())

                        self.end_encoding(writer_subprocess)

                        # New scene
                        scene = frame.scene
                        fp: str = scene['task'].video_file

                        # Use the destination frame count because we integrate all loops here
                        to_produce: int = scene['dst']['count']
                        remaining = to_produce
                        out_i: int = 0
                        print(purple(f"[V][E] output video file: {fp}"))
                        print(purple(f"[V][E] Frames to produce: {to_produce}"))
                        if verbose:
                            print(purple(f"[V][E] output video file: {fp}"))
                            print(purple(f"[V][E]Frames to produce: {to_produce}"))

                        src_scene: SrcScene = None
                        vsettings: VideoSettings = scene['task'].video_settings
                        if vsettings.codec not in str_to_video_codec:
                            sys.exit(red(f"{vsettings.codec} is not supported"))

                        # stdin pixel format
                        if PIXEL_FORMAT[vsettings.pix_fmt]['bpp'] > 8:
                            stdin_pixfmt = 'rgb48'
                            stdin_dtype = np.uint16
                            stdin_torch_dtype: torch.dtype = torch.uint16
                        else:
                            stdin_pixfmt = 'rgb24'
                            stdin_dtype = np.uint8
                            stdin_torch_dtype: torch.dtype = torch.uint8

                        # FFmpeg complex filter
                        filter_complex: list[str] = []
                        # upscale_height = 1080 + 2 * vsettings.pad
                        # if h > upscale_height:
                        #     # used for 'upscale' task only
                        #     fcomplex_str += f"""
                        #         scale=-1:{upscale_height}
                        #             :lanczos+accurate_rnd+bitexact+full_chroma_int
                        #     """
                        filter_complex_list: list[str] = []

                        filter_complex_list.append("""
                            setparams=colorspace=bt709
                                :color_primaries=bt709
                                :color_trc=bt709
                        """)

                        pad_filter: str = get_ffmpeg_pad_filter(scene)
                        if pad_filter:
                            filter_complex_list.append(pad_filter)

                        additional_filters: list[str] = get_encoder_ffmpeg_filters(
                            frame.sequence, start_step_no=frame.step_no
                        )
                        if additional_filters:
                            filter_complex_list.extend(additional_filters)

                        # Generate the FFmpeg filter command
                        fcomplex_str: str = ','.join(filter_complex_list)
                        fcomplex_str = clean_fcomplex(fcomplex_str)
                        if fcomplex_str:
                            filter_complex = [
                                "-filter_complex", f"[0:v]{fcomplex_str}[outv]",
                                "-map", "[outv]"
                            ]

                        if verbose:
                            print(purple(f"[V][E] fcomplex_str:"), fcomplex_str)

                        metadata: list[str] = ffmpeg_metadata(scene)
                        _, _, h, w = frame.tensor.shape
                        encoder_command: list[str] = [
                            ffmpeg_exe,
                            "-hide_banner",
                            "-loglevel", "error",
                            "-nostats",

                            "-f", "rawvideo",
                            '-pixel_format', stdin_pixfmt,
                            '-video_size', f"{w}x{h}",
                            "-r", str(vsettings.frame_rate),

                            "-i", "pipe:0",
                            *filter_complex,

                            "-vcodec", str_to_video_codec[vsettings.codec].value,
                            *vsettings.codec_options,
                            "-pix_fmt", vsettings.pix_fmt,
                            "-color_range", "limited",

                            *metadata,
                            "-y",
                            fp
                        ]

                        print(purple(f"[V][E] command: "), " ".join(encoder_command))

                        os.makedirs(path_split(fp)[0], exist_ok=True)
                        try:
                            writer_subprocess = subprocess.Popen(
                                encoder_command,
                                stdin=subprocess.PIPE,
                                stdout=sys.stdout,
                                stderr=subprocess.STDOUT,
                                # bufsize=10**8
                            )
                        except Exception as e:
                            print(f"[E] Unexpected error: {type(e)}", flush=True)


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

                    out_img: np.ndarray = dtoh_transfer(
                        dtoh_mem=dtoh_mem,
                        d_img=d_img,
                        out_dtype=stdin_dtype,
                        cuda_stream=cuda_stream,
                    ).reshape(d_img.shape)

                    in_f_no: int = frame.f_no

                    # Apply additionnal filters exectuted by the cpu
                    frame.tensor = out_img
                    # if frame.step_no < len(frame.sequence):
                    #     apply_cpu_filters_(frame)

                    in_frame: McoFrame = McoFrame(
                        no=frame.f_no,
                        img=frame.tensor,
                        scene=scene
                    )
                    if verbose:
                        print(purple(f"\nreceived frame no. {in_frame.no}"))

                    if verbose:
                        print(yellow(f"[V][E] {out_i}: send {frame.f_no}"), flush=True)

                    out_frames: list[McoFrame] = apply_upscale_effects(
                        in_f_no, in_frame, out_i
                    )
                    [writer_subprocess.stdin.write(f.img) for f in out_frames]
                    out_i += len(out_frames)
                    remaining -= len(out_frames)
                    sent = len(out_frames)

                    del frame.tensor
                    if self.progress_thread is not None:
                        self.progress_thread.put(sent)
                    self._encoded += sent

        # except:
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


