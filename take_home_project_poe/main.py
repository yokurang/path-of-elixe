import os
import logging
from datetime import datetime

def setup_logging(level: str = "INFO") -> str:
    """
    Initialize logging to console + logs/log_<YYYY-MM-DD-HH-MM>.txt.
    Returns the path to the log file.
    """
    os.makedirs("logs", exist_ok=True)
    log_path = datetime.now().strftime("logs/log_%Y-%m-%d-%H-%M.txt")

    # If logging already configured, don't reconfigure; just add a file handler.
    root = logging.getLogger()
    has_handlers = bool(root.handlers)

    lvl = getattr(logging, level.upper(), logging.INFO)
    if not has_handlers:
        logging.basicConfig(
            level=lvl,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_path, encoding="utf-8")
            ],
        )
    else:
        # Ensure our file handler exists
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(lvl)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%H:%M:%S")
        fh.setFormatter(fmt)
        root.addHandler(fh)
        root.setLevel(lvl)

    logging.info("Logging initialized; file=%s", log_path)
    return log_path