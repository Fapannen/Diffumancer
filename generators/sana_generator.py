import torch
from diffusers import SanaPAGPipeline

from generators.base_generator import BaseGenerator
from lora.lora_config import LoraConfig


class SanaGenerator(BaseGenerator):
    def __init__(
        self,
        model_path: str,
        loras: list[LoraConfig] = [],
        torch_dtype=torch.bfloat16,
        use_pag: bool = False,
        pag_scale: float = 3.0,
    ):
        self.use_pag = use_pag
        self.pag_scale = pag_scale

        super().__init__(model_path, loras, torch_dtype)

    def init_model_pipeline(self, model_path):
        if self.use_pag:
            self.pipeline = SanaPAGPipeline.from_pretrained(
                model_path,
                pag_applied_layers=["transformer_blocks.8"],
                torch_dtype=self.dtype,
            )

            self.pipeline.enable_model_cpu_offload()

        else:
            super().init_model_pipeline(model_path)
