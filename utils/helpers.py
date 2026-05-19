"""通用工具函数，从EchoVLM utils.py精简适配。"""

import torch
import torch.distributed as dist
from typing import Dict, List


def rank0_print(*args):
    """仅 rank 0 打印，多卡训练时避免重复输出。"""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(*args)
    else:
        print(*args)


def find_all_linear_names(named_modules: Dict, target_modules: List[str]):
    """从命名模块中筛选出 Linear 层的名字（排除 moe/coefficient/lm_head）。"""
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in named_modules.items():
        if not any(module_name in name for module_name in target_modules):
            continue
        if isinstance(module, cls) and 'moe' not in name and 'coefficient' not in name:
            lora_module_names.add(name)
    for name in list(lora_module_names):
        if 'lm_head' in name:
            lora_module_names.remove(name)
    return sorted(list(lora_module_names))
