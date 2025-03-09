import torch

from generators.base_generator import BaseGenerator
from lora.lora_config import LoraConfig


class FluxGenerator(BaseGenerator):
    def __init__(
        self,
        model_path: str,
        loras: list[LoraConfig] = [],
        torch_dtype=torch.bfloat16,
    ):
        super().__init__(model_path, loras, torch_dtype, use_safetensors=True)
