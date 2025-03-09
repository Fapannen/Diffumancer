# Diffumancer - Summoning images from the depths of noise

![alt text](img/thumbnail.png)

(All showcase images were generated using this repository)

## Description

This repository serves as means to run inference of Stable Diffusion
models via the `diffusers` library. No GUIs like ComfyUI, Automatic1111
are required.

## Setup

- Build your docker image
    - You might wish to edit `FBUILD_*` variables.
    - It is de-facto necessary to have a NVIDIA GPU. (The code can
        run on CPU only, however it will take forever.)
        - NVIDIA Contaner Toolkit must be 
          [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
    - `docker build -t image-generation .`

- Decide how you want to handle Huggingface Cache directory and set it up.
    It is recommended to create a shared cache for both the
    host and the docker. Eg. designate a folder on your
    machine and set it to be the huggingface cache. Then,
    mount the directory into docker during initialization
    and set it to be also the cache there.
    Eg.  
    ```
    (host-designated cache dir) : /mnt/whatever
    (docker run command)        : --mount "type=bind,src=/mnt/whatever,dst=/path/to/docker/cache"
    (docker)                    : export HF_HOME=/path/to/docker/cache"
    ```

    If you haven't worked with HuggingFace before and do not have any cache
    set up, the most seamless setup is to create a directory directly in this
    cloned repository named `huggingface`. In such a case, you do not need to
    worry about any setup.
    
- Run the docker
    ```bash
    docker run \
        --mount "type=bind,src=.,dst=/home/<FBUILD_USERNAME>" \
        --runtime=nvidia \
        --gpus all \
        -it \
        image-generation
    ```  
    If you created the `huggingface` directory as a cache in this repository,
    it gets mounted as a part of the `mount` command and is accurately exported
    during the build of the image. See the end of the [Dockerfile](Dockerfile)
    for clarifiation.  
    
    If you wish to set the cache somewhere else, use something
    along the lines of  

    ```bash
    docker run \
        --mount "type=bind,src=.,dst=/home/<FBUILD_USERNAME>" \
        --mount "type=bind,src=path/to/host/cache,dst=path/to/docker/cache" \
        -e HF_HOME="path/to/docker/cache"
        --runtime=nvidia \
        --gpus all \
        -it \
        image-generation
    ```  

- Generate images in docker, either via devcontainer environment, or directly
    in command line

## Example

See the example [Jupyter Notebook](image_generation_pub.ipynb) for an example
how to run the inference.