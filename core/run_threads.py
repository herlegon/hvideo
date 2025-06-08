import time

from utils.p_print import red
from .t_decoder import DecoderThread
from .t_encoder import EncoderThread
from .t_progress import ProgressThread
from .types import BaseThread


def run_threads(
    d_thread: DecoderThread,
    i_threads: list[BaseThread] | BaseThread,
    e_thread: EncoderThread,
    progress_thread: ProgressThread | None = None,
    total_frames: int = 0,
    verbose: bool = False
) -> float:
    # Start all threads
    if not isinstance(i_threads, list | tuple):
        i_threads = [i_threads]

    d_thread.verbose = verbose
    e_thread.verbose = verbose

    # Indicates to the decoder/encoder that it's a full GPU workflow
    #   bc modified API for future use (ncnn)
    #   use a null context if cpu
    d_thread.is_cuda_workflow = True
    e_thread.is_cuda_workflow = True

    start_time = time.time()

    d_thread.set_produce_flag()
    for thread in (
        d_thread,
        *i_threads,
        e_thread,
        progress_thread,
    ):
        if thread is not None:
            thread.start()


    decoding: bool = True
    err: bool = False
    ask_to_end: bool = False
    while True:
        if not decoding:
            # All frames encoded, send poison pill to all inference threads
            if e_thread.encoded == total_frames and ask_to_end:
                if verbose:
                    print(f"[V][C] All frames encoded {e_thread.encoded}/{total_frames}, send poison pill to encoder")
                e_thread.put_frame(None)
                ask_to_end = False

            # Encoding has ended
            if (
                not d_thread.is_alive()
                and not e_thread.is_alive()
            ):
                if verbose:
                    print("[V][C] All frames encoded, encoder has ended")
                encoded = e_thread.encoded
                break
            time.sleep(0.00001)
            continue

        time.sleep(0.00001)
        # Detect end of decoding
        if not d_thread.is_alive() and decoding:
            if verbose:
                print("[V][C] decoder has ended")
            err, message = d_thread.error_encountered()
            if err:
                print(red(message))
                break
            else:
                decoding = False
                ask_to_end = True
                if verbose:
                    print(f"[V][C] wait for {total_frames} to be encoded")

        time.sleep(0.00001)
        if not e_thread.is_alive() and d_thread.is_alive():
            print(red("Error: the encoder encountered an unexpected error"))
            err = True
            break

        time.sleep(0.00001)

    elapsed = time.time() - start_time
    if progress_thread is not None:
        progress_thread.put(0, force=True)
    time.sleep(0.00001)

    # Stop remaining threads if not already stopped (error cases)
    for thread in i_threads:
        if thread is not None and thread.is_alive():
            # print(f"[V][C] sent a poison pill at the end because still inference threads")
            thread.stop(force=True)
            while thread.is_alive():
                time.sleep(0.02)

    for thread in (d_thread, *i_threads, e_thread):
        if thread is not None and thread.is_alive():
            thread.stop(force=True)
            time.sleep(0.5)

    if progress_thread is not None:
        while True:
            if progress_thread is not None and progress_thread.is_alive():
                progress_thread.stop(force=True)
                time.sleep(0.1)
            else:
                break

    for thread in (d_thread, *i_threads, e_thread):
        if thread is not None and thread.is_alive():
            print(red(f"Error: {thread.name} is still alive"))

    return elapsed



