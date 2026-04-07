"""GPU detection and adaptive resource allocation.

Provides a single source of truth for GPU availability, memory, and
worker allocation. Used by the simulator, scenario runner, and HPC scripts.

Adapts automatically:
  - 4x V100 on HPC  → 4 GPUs, distributes clients round-robin
  - 2x GPUs available → scales down workers proportionally
  - 1x GPU or CPU     → single-device mode, sequential processing
"""

import os
import torch


class GPUInfo:
    """Information about a single GPU."""
    __slots__ = ('index', 'device', 'name', 'total_memory_gb', 'free_memory_gb')

    def __init__(self, index, device, name, total_gb, free_gb):
        self.index = index
        self.device = device
        self.name = name
        self.total_memory_gb = total_gb
        self.free_memory_gb = free_gb

    def __repr__(self):
        return (f"GPU{self.index}({self.name}, "
                f"{self.free_memory_gb:.1f}/{self.total_memory_gb:.1f}GB free)")


def detect_gpus():
    """Detect all available CUDA GPUs with memory info.

    Respects CUDA_VISIBLE_DEVICES if set (Slurm sets this automatically).

    Returns:
        list of GPUInfo objects, sorted by index. Empty list if no GPUs.
    """
    if not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        device = torch.device(f'cuda:{i}')
        props = torch.cuda.get_device_properties(i)
        name = props.name
        total_gb = props.total_memory / (1024 ** 3)

        # Try to get free memory (requires a CUDA context)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            free_gb = free_bytes / (1024 ** 3)
        except Exception:
            free_gb = total_gb  # assume all free if we can't query

        gpus.append(GPUInfo(i, device, name, total_gb, free_gb))

    return gpus


def get_optimal_config(config=None, verbose=True):
    """Detect GPUs and return optimal resource allocation.

    Adapts clients_per_gpu and max_workers based on:
    - Number of GPUs available
    - GPU memory (V100 16GB → 3 clients/GPU, smaller → fewer)
    - Config overrides (clients_per_gpu=0 means auto)

    Args:
        config: dict with optional 'clients_per_gpu' key (0=auto)
        verbose: print GPU info

    Returns:
        dict with:
            'devices': list of torch.device (at least [cpu])
            'num_gpus': int
            'clients_per_gpu': int
            'max_workers': int (num_gpus * clients_per_gpu)
            'gpu_info': list of GPUInfo
            'device': primary torch.device for model init
    """
    config = config or {}
    gpus = detect_gpus()

    if not gpus:
        if verbose:
            print("  GPU: None detected — using CPU")
        return {
            'devices': [torch.device('cpu')],
            'num_gpus': 0,
            'clients_per_gpu': 1,
            'max_workers': 1,
            'gpu_info': [],
            'device': torch.device('cpu'),
        }

    num_gpus = len(gpus)

    # Determine clients_per_gpu
    cpg = config.get('clients_per_gpu', 0)
    if cpg <= 0:
        # Auto: scale by VRAM
        # V100 16GB → 3 clients, A100 40GB → 6, T4 16GB → 2, etc.
        min_vram = min(g.total_memory_gb for g in gpus)
        if min_vram >= 32:
            cpg = 6
        elif min_vram >= 16:
            cpg = 3
        elif min_vram >= 8:
            cpg = 2
        else:
            cpg = 1

    max_workers = num_gpus * cpg

    if verbose:
        print(f"  GPU: {num_gpus} device(s) detected")
        for g in gpus:
            print(f"    {g}")
        print(f"  Workers: {max_workers} ({cpg}/GPU x {num_gpus} GPUs)")

    return {
        'devices': [g.device for g in gpus],
        'num_gpus': num_gpus,
        'clients_per_gpu': cpg,
        'max_workers': max_workers,
        'gpu_info': gpus,
        'device': gpus[0].device,
    }


def estimate_memory_per_client(model_name='resnet18', batch_size=4, image_size=32):
    """Estimate GPU memory needed per concurrent client (in GB).

    Rough estimates based on model + optimizer + gradient + activations.
    """
    estimates = {
        'smallcnn': 0.3,
        'resnet18': 1.2,
        'resnet34': 1.8,
        'resnet50': 2.5,
    }
    base = estimates.get(model_name, 1.5)
    # Scale with batch size (activations grow linearly)
    return base * (batch_size / 4)


def print_gpu_summary():
    """Print a human-readable GPU summary. Useful for HPC job logs."""
    gpus = detect_gpus()
    if not gpus:
        print("No CUDA GPUs available.")
        print(f"  CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
        return

    print(f"{'=' * 60}")
    print(f"  GPU Summary: {len(gpus)} device(s)")
    print(f"{'=' * 60}")
    for g in gpus:
        used = g.total_memory_gb - g.free_memory_gb
        pct = (used / g.total_memory_gb * 100) if g.total_memory_gb > 0 else 0
        bar_len = 30
        filled = int(bar_len * pct / 100)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"  GPU{g.index}: {g.name}")
        print(f"    Memory: [{bar}] {used:.1f}/{g.total_memory_gb:.1f} GB ({pct:.0f}%)")
    print(f"{'=' * 60}")
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
    print(f"  CUDA_VISIBLE_DEVICES = {cvd}")
    print(f"  PyTorch CUDA version = {torch.version.cuda}")


if __name__ == '__main__':
    print_gpu_summary()
    print()
    cfg = get_optimal_config(verbose=True)
    print(f"\nRecommended config:")
    print(f"  devices: {cfg['devices']}")
    print(f"  clients_per_gpu: {cfg['clients_per_gpu']}")
    print(f"  max_workers: {cfg['max_workers']}")
