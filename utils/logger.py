from __future__ import annotations

from datetime import datetime
import logging
from argparse import Namespace
import os

from .p_print import lightcyan
from .path_utils import get_app_tempdir, path_split

logger: logging.Logger = logging.getLogger("pytc")


def set_logger_settings(
    args: Namespace,
    out_media_fp: str = ""
) -> None:

    if args.log:
        if not out_media_fp:
            log_dir = get_app_tempdir()
            current_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_basename = f"pytc_{current_dt}"
        else:
            log_dir, log_basename = path_split(out_media_fp)[:2]

        log_dir, log_basename = path_split(out_media_fp)[:2]
        log_filepath: str = os.path.join(log_dir, f"{log_basename}.log")
        logger.addHandler(
            logging.FileHandler(log_filepath, mode="w")
        )
        logger.setLevel("DEBUG")
        print(
            lightcyan(f"Log saved in"), log_dir,
            lightcyan("as:"), f"{log_basename}.log"
        )

    else:
        logger.setLevel("WARNING")
