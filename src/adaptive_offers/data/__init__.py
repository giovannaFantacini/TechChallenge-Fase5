"""Ingestão e preparação da base Bank Marketing."""

from .download import download_dataset, load_raw
from .prepare import (
    arm_summary,
    build_events,
    context_from_client,
    load_events,
    prepare,
    save_events,
)

__all__ = [
    "download_dataset",
    "load_raw",
    "build_events",
    "context_from_client",
    "arm_summary",
    "prepare",
    "save_events",
    "load_events",
]
