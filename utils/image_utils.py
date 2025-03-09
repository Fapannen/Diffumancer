from pathlib import Path
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_images(numpy_images: List[np.ndarray], images_per_row: int) -> None:
    """Plot images using matplotlib.

    Parameters
    ----------
    numpy_images : List[np.ndarray]
        Images to be plotted, expected to be in RGB format.
    images_per_row : int
        How many images should there be within one row.
    """
    div = int(len(numpy_images) // images_per_row)
    nrows = div + 1 if len(numpy_images) % images_per_row != 0 else div
    fig, axs = plt.subplots(nrows, images_per_row, figsize=(30, 30))

    for i in range(len(numpy_images)):
        if nrows == 1:
            axs[i % images_per_row].title.set_text(f"{i}")
            axs[i % images_per_row].imshow(numpy_images[i])

        else:
            axs[i // images_per_row, i % images_per_row].title.set_text(f"{i}")
            axs[i // images_per_row, i % images_per_row].imshow(numpy_images[i])


def convert_images_to_numpy(pil_images: List[Any]) -> List[np.ndarray]:
    return [np.array(img) for img in pil_images]


def convert_images_to_pil(numpy_images: List[np.ndarray]) -> List[Image.Image]:
    return [Image.fromarray(img) for img in numpy_images]


def save_images(images: List[np.ndarray | Image.Image], output_dir: str | Path) -> None:
    """Save images to directory.

    Parameters
    ----------
    images : List[np.ndarray  |  Image.Image]
        Images to be saved, either in numpy format or PIL Image. Provided
        images are expected to be in the RGB format for both cases.
    output_dir : str | Path
        Destination directory
    """
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    _type = type(images[0])
    if _type == np.ndarray:
        for i in range(len(images)):
            cv2.imwrite(
                str(Path(output_dir) / f"image_{i}.jpg"),
                cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR),
            )
    else:
        for i in range(len(images)):
            images[i].save(f"{str(Path(output_dir))}/image_{i}.jpg")
