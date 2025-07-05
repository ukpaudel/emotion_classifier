import logging
import os
from datetime import datetime

"""
Utility logger that writes both to console and to file.
Used across training, evaluation, error tracking, and config auditing.
"""

def setup_logger(log_dir, run_name, filename="run.log"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate logs if multiple handlers exist
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    logger.info(f"Logger initialized for run: {run_name} at {datetime.now().isoformat()}")
    return logger
