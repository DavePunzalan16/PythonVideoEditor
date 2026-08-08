"""
Centralized logging configuration for the restoration pipeline.
"""

import logging
import os


def setup_logging(log_file: str = "restore_app.log", level: int = logging.INFO):
    """
    Configure logging for the entire application.

    Logs go to both a file and stderr for easy diagnosis.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
