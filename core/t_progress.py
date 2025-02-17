from queue import Queue
import sys
from threading import Event
import time
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)

from .types import BaseThread
from utils.p_print import *
from utils.path_utils import path_split



class ProgressThread(BaseThread):
    def __init__(
        self,
        total: int = 0,
        refresh_rate: int = 2,
    ) -> None:
        super().__init__()
        self._stop_event = Event()
        self._stop_event.clear()
        self.in_queue: Queue = Queue()
        self.total: int = total
        self._encoded: int = 0
        self.interval: float = 1. / refresh_rate
        self.refresh_rate: int = refresh_rate
        self.is_running: bool = False
        self.previous_timestamp: float = 0.
        self.start_time: float = 0.
        self.previous_encoded = 0
        self._elapsed: float = 0.
        self.encoding_start_time: float = 0.
        self.encoded: int = 0


    def run(self):
        self.is_running = True

        in_queue: Queue = self.in_queue
        in_queue.put((0, 0))
        self.progress = Progress(
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "frame:",
            "{task.fields[encoded]}" + f"/{self.total}",
            "elapsed=",
            TimeElapsedColumn(),
            "fps=",
            "{task.fields[fps]:<3.1f}",
            "ETA",
            TimeRemainingColumn(),
            refresh_per_second=self.refresh_rate,
        )
        self.progress.start()
        self.task_id = self.progress.add_task(
            "[green] Processing...",
            total=self.total,
            encoded=0,
            fps=0,
            start=True
        )

        previous_encoded = 0
        self.start_time = time.time()
        self.previous_timestamp = self.start_time
        self.encoding_start_time = self.start_time
        while not self._stop_event.is_set():
            data = in_queue.get(block=True)
            if data is None:
                break
            encoded, elapsed = data
            self.progress.update(
                self.task_id,
                advance=encoded - previous_encoded,
                encoded=encoded,
                fps=float(encoded) / elapsed if elapsed else 0,
            )
            previous_encoded = encoded

            # if elapsed:
            #     fps = encoded / elapsed
            #     print(
            #         " ".join((
            #         purple(f"{(100 * encoded)/self.total:0.1f}%"),
            #         f"frame: {encoded}/{self.total}",
            #         f"elapsed=", yellow(f"{int(elapsed)}"),
            #         f"fps= {fps:.1f}",
            #         blue(f"ETA {int(fps * encoded)}"),
            #         '\r')),
            #         # end='\r',
            #         file=sys.stdout,
            #         flush = True
            #     )

            if self._stop_event.is_set():
                break

        self.progress.stop()
        time.sleep(0.1)
        self.is_running = False


    def stop(self, force: bool=False) -> None:
        if force:
            self.progress.stop()
            while not self.in_queue.empty():
                self.in_queue.get_nowait()
        self.in_queue.put_nowait(None)
        self._stop_event.set()


    def put(self, value: int | None, force: bool = False):
        if value is None:
            self._elapsed = time.time() - self.encoding_start_time
            self.in_queue.put_nowait(None)
            return

        if self.encoding_start_time == 0:
            self.encoding_start_time = time.time()

        self.encoded += value

        if (time.time() - self.previous_timestamp) >= self.interval or force:
            self.in_queue.put_nowait((self.encoded, time.time() - self.start_time))
            self.previous_timestamp = time.time()


    def elapsed(self) -> float:
        return self._elapsed
