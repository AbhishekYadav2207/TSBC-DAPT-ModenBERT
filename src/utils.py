import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Union

def get_project_root() -> Path:
    """
    Returns the absolute path to the repository project root.
    Assumes this file is located at PROJECT_ROOT/dapt/src/utils.py.
    """
    return Path(__file__).resolve().parent.parent.parent

def get_dapt_root() -> Path:
    """
    Returns the absolute path to the DAPT directory (PROJECT_ROOT/dapt).
    """
    return Path(__file__).resolve().parent.parent

def resolve_path(path_str: str) -> Path:
    """
    Resolves a file or directory path relative to project root if not absolute.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return get_project_root() / path

def setup_logging(log_file: Union[str, Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a logger instance logging to stdout and optionally a log file.
    """
    logger = logging.getLogger("dapt")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def save_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> Path:
    """
    Safely writes a dictionary to a JSON file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    return path

def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely loads a JSON file into a dictionary.
    """
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
