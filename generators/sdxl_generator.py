from typing import List

import torch
from diffusers import AutoPipelineForText2Image, DiffusionPipeline
from diffusers.utils.outputs import BaseOutput as PipelineOutput

from generators.base_generator import BaseGenerator
from lora.lora_config import LoraConfig


class SDXLGenerator(BaseGenerator):
    def __init__(
        self,
        model_path: str,
        loras: List[LoraConfig] = [],
        torch_dtype: torch.dtype = torch.bfloat16,
        use_refiner: bool = False,
        refiner_frac: float = 0.8,
        use_pag: bool = False,
        pag_scale: float = 3.0,
    ):
        """Initialize the SDXL generator

        Parameters
        ----------
        model_path : str
            Path to the model, either local or huggingface
        loras : List[LoraConfig]
            List of used LORAs
        torch_dtype : torch.dtype, optional
            dtype to use, by default torch.bfloat16
        use_refiner : bool, optional
            Whether to use SDXL refiner, by default False
        refiner_frac : float, optional
            When using refiner, fraction specifying when the refiner
            takes over. By default 0.8
        use_pag : bool, optional
            Whether to use Perturbed-Attention Guidance, by default False
        pag_scale : float, optional
            PAG scale, by default 3.0. When pag_scale increases, images
            gain more semantically coherent structures and exhibit fewer
            artifacts. However overly large guidance scale can lead to
            smoother textures and slight saturation in the images,
            similarly to CFG. pag_scale=3.0 is used in the official
            demo and works well in most of the use cases, but feel
            free to experiment and select the appropriate value
            according to your needs! PAG is disabled when pag_scale=0.

        Raises
        ------
        ValueError
            When invalid arguments are encountered
        """
        if use_pag:
            self.use_pag = use_pag
            self.pag_scale = pag_scale

            if self.pag_scale <= 0:
                print(
                    "[SDXL]: Warning: 'pag_scale' set to zero or lower. PAG will not be used."
                )

        if use_refiner:
            if not 0.0 < refiner_frac < 1.0:
                raise ValueError(
                    f"Invalid value {refiner_frac} provided for the 'refiner_frac' parameter. Expected (0.0, 1.0)"
                )

            self.use_refiner = use_refiner
            self.refiner_frac = refiner_frac

        super().__init__(
            model_path,
            loras,
            torch_dtype,
            use_safetensors=True,
            variant=self.infer_variant(torch_dtype),
        )

    def init_model_pipeline(self, model_path: str, **kwargs) -> None:
        """In addition to the standard pipeline, SDXL can
        make use of other tweaks, eg. PAG, refiner, etc. These
        tweaks are additional pipelines which are to be initialized

        Parameters
        ----------
        model_path : str
            Path to the model, either a HF-like path, or standard
            path on local machine.
        """
        super().init_model_pipeline(model_path, **kwargs)

        if self.use_pag:
            self.pipeline = AutoPipelineForText2Image.from_pipe(
                self.pipeline, enable_pag=True
            )
            self.pipeline.enable_model_cpu_offload()

        if self.use_refiner:
            self.init_refiner()

    def init_refiner(self) -> None:
        """Init the refiner. The refiner's task is to improve
        the generated outputs from the standard underlying pipeline.
        As such, use specifies the 'refiner_frac' argument, which
        specified after how many steps the refiner is to take over
        the generation.
        """
        self.refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.pipeline.text_encoder_2,
            vae=self.pipeline.vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            variant=self.infer_variant(self.dtype),
        )

        self.refiner.enable_model_cpu_offload()

    def run_inference(self, *args, **kwargs) -> PipelineOutput:
        """See docstring of the parent class for details."""

        for lora in self.loras:
            if lora.triggerword is not None:
                if "prompt" not in kwargs:
                    raise ValueError("No prompt found in kwargs!")
                kwargs["prompt"] = f"{lora.triggerword}, {kwargs['prompt']}"

        if not self.use_refiner:
            self.refiner_frac = 1.0

        with torch.inference_mode():
            if self.use_pag:
                self.output = self.pipeline(
                    *args,
                    **kwargs,
                    denoising_end=self.refiner_frac,
                    pag_scale=self.pag_scale,
                )
            else:
                self.output = self.pipeline(
                    *args,
                    **kwargs,
                    max_sequence_length=512,
                    denoising_end=self.refiner_frac,
                )

            if self.use_refiner:
                num_images = (
                    kwargs["num_images_per_inference"]
                    if "num_images_per_inference" in kwargs
                    else 1
                )

                self.output = self.refiner(
                    prompt=kwargs["prompt"],
                    num_inference_steps=kwargs["num_inference_steps"],
                    denoising_start=self.refiner_frac,
                    image=self.output.images,
                    num_images_per_prompt=num_images,
                )
        return self.output
