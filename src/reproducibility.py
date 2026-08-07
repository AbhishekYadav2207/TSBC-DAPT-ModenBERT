import random
import sys
import platform
import numpy as np
import torch
import transformers
import datasets
from typing import Dict, Any

def set_seed(seed: int = 42) -> None:
    """
    Sets deterministic random seeds for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_system_environment() -> Dict[str, Any]:
    """
    Collects runtime environment details including OS, Python, PyTorch, Transformers,
    CUDA availability, GPU hardware specifications, and system memory.
    """
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": []
    }

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            env["gpus"].append({
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                "multi_processor_count": props.multi_processor_count
            })

    return env
