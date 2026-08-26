from __future__ import annotations

import logging
import os
from typing import Any

import torch.distributed as dist

_LOG_RECORD_FACTORY_INSTALLED = False


class DistributedRankFilter(logging.Filter):
    """Inject distributed rank metadata into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = get_rank()
        record.world_size = get_world_size()
        return True


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))


def get_local_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", get_rank()))
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))


def is_main_process() -> bool:
    return get_rank() == 0


def _install_rank_log_record_factory() -> None:
    global _LOG_RECORD_FACTORY_INSTALLED
    if _LOG_RECORD_FACTORY_INSTALLED:
        return

    previous_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = previous_factory(*args, **kwargs)
        record.rank = get_rank()
        record.world_size = get_world_size()
        return record

    logging.setLogRecordFactory(record_factory)
    _LOG_RECORD_FACTORY_INSTALLED = True


def setup_logging(
    level: int | str = logging.INFO,
    *,
    non_main_level: int | str = logging.WARNING,
    force: bool = False,
) -> None:
    """Configure default logging once, lowering non-main rank verbosity.

    Libraries should respect existing application logging. If a handler already
    exists, this only attaches rank metadata and adjusts the root level.
    """

    _install_rank_log_record_factory()
    rank_level = level if is_main_process() else non_main_level
    root = logging.getLogger()

    if not root.handlers or force:
        logging.basicConfig(
            level=rank_level,
            format="[rank %(rank)s/%(world_size)s] %(levelname)s:%(name)s: %(message)s",
            force=force,
        )
    else:
        root.setLevel(rank_level)

    rank_filter = DistributedRankFilter()
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, DistributedRankFilter) for f in handler.filters):
            handler.addFilter(rank_filter)


def log_rank_zero(logger: logging.Logger, level: int, message: str, *args: Any, **kwargs: Any) -> None:
    if is_main_process():
        logger.log(level, message, *args, **kwargs)


def log_all_ranks(logger: logging.Logger, level: int, message: str, *args: Any, **kwargs: Any) -> None:
    logger.log(level, message, *args, **kwargs)


def rank_zero_print(*args: Any, **kwargs: Any) -> None:
    if is_main_process():
        print(*args, **kwargs)
