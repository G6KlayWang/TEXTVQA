from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from src.utils.io import read_jsonl, write_jsonl


class ProcessInfo:
    def __init__(self) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.is_main = self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    def wait(self) -> None:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
                return
        except Exception:
            pass
        try:
            from accelerate import PartialState

            PartialState().wait_for_everyone()
        except Exception:
            pass


def current_process() -> ProcessInfo:
    return ProcessInfo()


def shard_indices(total: int, rank: int, world_size: int) -> range:
    return range(rank, total, world_size)


def part_path(output: str | Path, rank: int) -> Path:
    output = Path(output)
    return output.with_suffix(output.suffix + f".part-{rank}")


def merge_jsonl_parts(output: str | Path, world_size: int) -> None:
    output = Path(output)
    rows = []
    for rank in range(world_size):
        path = part_path(output, rank)
        if path.exists():
            rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: int(row.get("_index", 0)))
    for row in rows:
        row.pop("_index", None)
    write_jsonl(rows, output)


def cleanup_parts(output: str | Path, ranks: Iterable[int]) -> None:
    for rank in ranks:
        path = part_path(output, rank)
        if path.exists():
            path.unlink()

