"""
AI_Voice 日誌系統
=================
統一的日誌管理，支援 console + file 雙輸出。
"""
import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "ai_voice",
    level: str = "INFO",
    log_file: str | None = None
) -> logging.Logger:
    """建立並配置日誌記錄器

    Args:
        name: 日誌記錄器名稱
        level: 日誌等級（DEBUG/INFO/WARNING/ERROR/CRITICAL）
        log_file: 日誌檔案路徑（若為 None 則只輸出至 console）

    Returns:
        已配置的 Logger 實例
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重複添加 handler
    if logger.handlers:
        return logger

    # 格式化器
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler（如果指定了檔案路徑）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ai_voice") -> logging.Logger:
    """取得已存在的日誌記錄器

    Args:
        name: 日誌記錄器名稱

    Returns:
        Logger 實例
    """
    return logging.getLogger(name)
