# Copyright 2026 XYZ AI Lab and contributors.
# SPDX-License-Identifier: Apache-2.0

"""GPU selection utilities for model clients."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def select_least_utilized_gpus(num_gpus: int) -> list[int]:
    """Select the *num_gpus* GPUs with the lowest memory utilization.

    Uses ``pynvml`` (shipped with CUDA drivers) to read per-device memory usage.
    Falls back to sequential assignment ``[0, 1, ..., num_gpus-1]`` if ``pynvml``
    is unavailable.
    """
    try:
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlInit,
            nvmlShutdown,
        )

        nvmlInit()
        device_count = nvmlDeviceGetCount()
        usage: list[tuple[int, int]] = []  # (gpu_id, memory_used_bytes)
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            usage.append((i, mem_info.used))
        nvmlShutdown()

        # Sort ascending by memory used, pick the least-loaded devices.
        usage.sort(key=lambda x: x[1])
        selected = [gpu_id for gpu_id, _ in usage[:num_gpus]]
        return sorted(selected)
    except Exception:
        logger.warning(
            "pynvml unavailable — falling back to sequential GPU assignment [0..%d). "
            "Install nvidia-ml-py (pip install nvidia-ml-py) for auto-selection.",
            num_gpus,
        )
        return list(range(num_gpus))
