from pathlib import Path
from typing import Any
import logging

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)

    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Cấu hình trong {path} phải là một mapping YAML")

    return config


def resolve_project_path(path: str | Path) -> Path:
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return PROJECT_ROOT / resolved_path


def configure_logging(config: dict[str, Any]) -> None:
    """Configure one console handler and, when configured, one UTF-8 file handler."""
    logging_config = config.get("logging", {})
    level_name = str(logging_config.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    root = logging.getLogger()
    root.setLevel(level)

    if not any(getattr(handler, "_r2ai_console", False) for handler in root.handlers):
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console._r2ai_console = True
        root.addHandler(console)

    log_file = logging_config.get("log_file")
    if log_file:
        path = resolve_project_path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        if not any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == resolved
            for handler in root.handlers
        ):
            file_handler = logging.FileHandler(path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
