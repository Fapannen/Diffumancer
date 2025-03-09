import json
from pathlib import Path
from typing import List

import cv2
import numpy as np
import PIL.Image
import torch
from diffusers import DiffusionPipeline
from diffusers.utils.outputs import BaseOutput as PipelineOutput
from PIL import Image

from grid_settings.grid_inference_settings import GridInferenceSettings
from lora.lora_config import LoraConfig
from utils.image_utils import convert_images_to_numpy, plot_images, save_images


class BaseGenerator:
    """
    Base class for all generators. Most of the common functionality for all
    derived image generators can be found here. Generator-specific adjustments
    are expected to be implemented in the children classes.
    """

    def __init__(
        self,
        model_path: str,
        loras: list[LoraConfig] = [],
        torch_dtype=torch.bfloat16,
        **kwargs,
    ):
        self.model_path = model_path
        self.pipeline = None
        self.loras = None
        self.output = None
        self.dtype = torch_dtype

        self.init_model_pipeline(self.model_path, **kwargs)
        self.init_loras(loras)

    def init_model_pipeline(self, model_path: str, **kwargs) -> None:
        """
        Initialize the underlyring pipeline. Can be overridden
        in children classes to avoid double instantiation of a
        pipeline, or to modify the underlying pipeline post-hoc,
        eg. with refiners, etc.

        Parameters
        ----------
        model_path
            expected to be a path to the model. It can
            either be a huggingface link, or a path to a local
            huggingface-like directory
        kwargs
            Other custom keyword arguments to be passed to the
            DiffusionPipeline constructor.
        """
        self.pipeline = DiffusionPipeline.from_pretrained(
            model_path, torch_dtype=self.dtype, **kwargs
        )

        self.pipeline.enable_model_cpu_offload()

    def init_loras(self, loras: List[LoraConfig]) -> None:
        """Initialize the defined loras, if applicable

        Parameters
        ----------
        loras : List[LoraConfig]
            List of LoraConfig instances to be set up.
        """
        self.loras = loras
        self.use_loras = len(self.loras) > 0
        if self.use_loras:
            for lora in self.loras:
                self.pipeline.load_lora_weights(
                    lora.lora_path,
                    weight_name=lora.weights_file,
                    adapter_name=lora.name,
                )
            lora_names = [lora.name for lora in self.loras]
            lora_weights = [lora.weight for lora in self.loras]

            self.pipeline.set_adapters(lora_names, lora_weights)

    def infer_variant(self, dtype: type = None) -> str:
        """
        Infer which dtype of the model should be loaded. This can then
        be passed as a parameter to the constructor of the generator.
        """
        dtype_matcher = dtype if dtype is not None else self.dtype

        match dtype_matcher:
            case torch.float16:
                variant = "fp16"
            case torch.bfloat16:
                variant = "fp16"
            case torch.float8_e4m3fn:
                variant = "fp8"
            case torch.float8_e5m2:
                variant = "fp8"
            case torch.float32:
                variant = "fp32"
        return variant

    def run_inference(self, *args, **kwargs) -> PipelineOutput:
        """Run inference of the underlying pipeline.

        Parameters
        ----------
        prompt : str
            Prompt to evaluate
        guidance_scale : float
            Guidance scale. This parameter describes the output's adherence
            to the prompt. The higher the value, the higher the adherence.
            Values are expected to be nonnegative, with the
            upper bound not defined.

            The optimal value is to be determined by trial and error
            experiments, however, most generators recommend something
            in the range of [5.0, 12.0]. My experiments have shown similar
            pattern, where this range produces the most visually pleasing
            outputs.

        num_inference_steps : int
            Number of denoising steps of the underlying diffusion model.
            Again, a hyperparameter that is to be set by trial and error.
            Most diffusers recommend settings of eg. 20, 40, where my
            experiments have shown similar pattern.
        seed : int
            Seed to be used for initializing the noise vector, ensures
            reproducible results, if needed.
        height : int
            Height of the output images
        width : int
            Width of the output images
        num_images_per_inference : int
            How many images should be generated per each call. Generally
            should be left at 1, unless you have a very powerful GPU
            ( >= 24GB VRAM ).

        Returns
        -------
        ImagePipelineOutput
            Output of the pipeline
        """

        # Apply LORA triggerwords, if required
        for lora in self.loras:
            if lora.triggerword is not None:
                prompt = f"{lora.triggerword}, {prompt}"

        with torch.inference_mode():
            self.output = self.pipeline(*args, **kwargs)

        return self.output

    def grid_inference(
        self,
        grid_inference_setting: GridInferenceSettings,
        save_images_progressively: bool = True,
        save_images_dir: str | Path = f"outputs/",
    ) -> List[PipelineOutput]:
        """Run multiple inferences using different settings
        (therefore 'grid').

        Parameters
        ----------
        grid_inference_setting : GridInferenceSettings
            The settings that are to be explored. See the
            GridInferenceSettings class for more details how
            to set this up.
        save_images_progressively : bool, optional
            If True, images are saved to an output directory after
            each inference call. This enables you to track the results
            earlier than only after all setting have been inferred, and
            potentially kill the inference early, if you are not satisfied
            with the quality of the images. By default set to True
        save_images_dir : str | Path, optional
            Path where the intermediate images are to be stored.
            By default "outputs".

        Returns
        -------
        List[PipelineOutput]
            List of outputs, each corresponding to one individual setting.
        """

        images = []
        metadata = []
        outputs = []

        settings = grid_inference_setting.get_settings()
        for setting in settings:
            print(f"Generating setting {settings.index(setting) + 1} / {len(settings)}")
            inference_output = self.run_inference(**setting)
            outputs.append(inference_output)
            images_output = convert_images_to_numpy(
                self.get_images_from_pipe_output(inference_output)
            )
            images.extend(images_output)

            images_metadata = [
                {key: str(val) for key, val in setting.items()}
                | {
                    "image": len(images) + i,
                    "loras": list(
                        zip(
                            [lora.name for lora in self.loras],
                            [lora.weight for lora in self.loras],
                        )
                    ),
                    "refiner_used": (
                        self.use_refiner if hasattr(self, "use_refiner") else "nan"
                    ),
                    "refiner_frac": (
                        self.refiner_frac if hasattr(self, "refiner_frac") else "nan"
                    ),
                    "pag_used": self.use_pag if hasattr(self, "use_pag") else "nan",
                    "pag_scale": (
                        self.pag_scale if hasattr(self, "pag_scale") else "nan"
                    ),
                }
                for i in range(len(images_output))
            ]
            metadata.extend(images_metadata)

            if save_images_progressively:
                Path(save_images_dir).mkdir(exist_ok=True, parents=True)
                for i in range(len(images_output)):
                    cv2.imwrite(
                        str(Path(save_images_dir) / f"image_{len(images) + i}.jpg"),
                        cv2.cvtColor(images_output[i], cv2.COLOR_RGB2BGR),
                    )

        self.output = outputs
        self.metadata = metadata
        self.save_metadata(save_images_dir)

        return self.output

    def save_metadata(self, output_dir: str | Path) -> None:
        """Flush the generation metadata into a JSON file.

        Parameters
        ----------
        output_dir : str | Path
            Where to place the 'metadata.json' file
        """
        with open(str(Path(output_dir) / "metadata.json"), "w") as metadata_file:
            json.dump(self.metadata, metadata_file)

    def get_images_from_pipe_output(
        self, pipe_outputs: PipelineOutput | List[PipelineOutput]
    ) -> List[Image.Image]:
        """Extract raw images from the outputs of the pipeline.
        This method can handle both a single-output pipe output,
        eg. from the 'run_inference()' method, or a multi-output
        pipe output, eg. from the 'grid_inference()' method.

        Parameters
        ----------
        pipe_outputs : PipelineOutput | List[PipelineOutput]
            The outputs from which to extract the images

        Returns
        -------
        List[Image.Image]
            List of PIL images
        """

        if issubclass(type(pipe_outputs), PipelineOutput):
            images = [img for img in pipe_outputs["images"]]

        elif isinstance(pipe_outputs, list):
            images = []
            for ipo in pipe_outputs:
                images.extend([img for img in ipo["images"]])

        return images

    def show_images(self, images: List[np.ndarray], images_per_row: int = 5) -> None:
        """Plot images using matplotlib. Useful in interactive Jupyter
        notebooks.

        Parameters
        ----------
        images : List[np.ndarray]
            Images to be plotted. Expected to be in RGB format.
        images_per_row : int, optional
            Number of images per row in the visualization. By default 5
        """
        plot_images(images, images_per_row=min(images_per_row, len(images)))

    def save(
        self,
        images: PipelineOutput | List[np.ndarray | Image.Image | PipelineOutput],
        output_dir: Path | str,
    ) -> None:
        """Save (generated) images to a directory on the local machine.

        Parameters
        ----------
        images : PipelineOutput | List[np.ndarray  |  Image.Image  |  PipelineOutput]
            Images to be saved to a local machine. This method can handle various
            formats of the provided images. You can provide the images to be saved
            either as
                - PipelineOutput
                - List of numpy arrays
                - List of PIL images
                - List of PipelineOutput-s

        output_dir : Path | str
            Path to the directory where to save the images. The path is
            created if it does not exist.

        Raises
        ------
        ValueError
            If types are not resolved correctly.
        """
        # Single pipeline output
        if issubclass(type(images), PipelineOutput):
            save_images(self.get_images_from_pipe_output(images), output_dir)

        # List of either pipeline outputs, numpy arrays, or PIL images
        elif isinstance(images, list):
            if issubclass(type(images[0]), PipelineOutput):
                images_to_be_saved = []
                for ipo in images:
                    images_to_be_saved.extend(self.get_images_from_pipe_output(ipo))
                images = images_to_be_saved

            save_images(images, output_dir)
            self.save_metadata(output_dir)

        else:
            raise ValueError("Unknown type supplied for saving.")

    def select_images(
        self, indices: List[int] = [0]
    ) -> List[PIL.Image.Image | np.ndarray]:
        """Run after calling the 'run_inference' method to
        obtain images. These images may be subject for
        further processing, eg. upscale, refining, etc.

        Parameters
        ----------
        indices : List[int], optional
            list of image indices to be gathered

        Returns
        -------
        List[PIL.Image.Image | np.ndarray]
            List of selected images either in the PIL or
            numpy format.
        """
        return [self.output.images[i] for i in indices]
