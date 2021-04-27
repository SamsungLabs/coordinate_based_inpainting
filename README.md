# Coordinate-based texture inpainting for pose-guided human image generation

### <img align=center src=./data/icons/project.png width='32'/> [Project](https://saic-violet.github.io/coordinpaint) &ensp; <img align=center src=./data/icons/paper.png width='24'/> [Paper](https://arxiv.org/abs/1906.08240v3) &ensp; <img align=center src=./data/icons/supmat.png width='24'/> [Sup.Mat.](https://saic-violet.github.io/coordinpaint/files/SupMat.pdf) 

This is the repository with the inference code for the paper **"Coordinate-based texture inpainting for pose-guided human image generation"** (CVPR 2019).

## About

We present a new deep learning approach to pose-guided resynthesis of human photographs. At the heart of the new approach is the estimation of the complete body surface texture based on a single photograph. Since the input photograph always observes only a part of the surface, we suggest a new inpainting method that completes the texture of the human body. Rather than working directly with colors of texture elements, **the inpainting network estimates an appropriate source location in the input image for each element of the body surface**. This correspondence field between the input image and the texture is then further warped into the target image coordinate frame based on the desired pose, effectively establishing the correspondence between the source and the target view **even when the pose change is drastic**. The final convolutional network then uses the established correspondence and all other available information to synthesize the output image using a fully-convolutional architecture with deformable convolutions. We show the state-of-the-art result for pose-guided image synthesis. Additionally, we demonstrate the performance of our system for garment transfer and pose-guided face resynthesis.

<p align="center">
  <img src="./assets/idea.gif" alt="drawing", width="1280"/>
</p>

## Install
To use this repository, please first run the following:

```bash
$ bash install_deps.sh    # creates a new conda environment
$ conda activate coordinpaint    # switches to it
```

Then, download the following files:
1. A model checkpoint from here: [link to Google Drive](https://drive.google.com/file/d/10k4_JTVTVADyR2YGcnc8Z-dF58jpFffn/view?usp=sharing). It consists of two files `inpainter.pth` and `refiner.pth`. **They need to be placed under `data/checkpoint/` directory.**
2. Download `smpltexmap.npy` file from here: [link to Google Drive](https://drive.google.com/file/d/1F-aQx-5VQly1OvB5VvvHGkqpJUgzUYlU/view?usp=sharing) and **put it under `data/` directory.** It is required to convert uv renders produced by [DensePose](http://densepose.org/) algorithm (`*_IUV.png` files) to SMPL format used by our model.

## Usage   
Two simple ways to run the code are:

- Use [demo.ipynb](demo.ipynb) notebook
- Run `convert_uv_render.py` and `infer_sample.py` scripts.

This repository contains some examples of input data (rgb images and UV renders) 
from [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html) dataset. 
You can run the scripts above with these samples. 

The scripts are for demonstrational purposes and are not suited to process multiple files, so you'll probably need to write your own processing loop using functions from this repo.

#### Usage examples
`convert_uv_render.py` converts densepose IUV renders into smpl format. 
It takes output of densepose method (`*_IUV.png` file) from `data/samples/source_iuv` directory and
saves resulting uv render to `data/samples/source_uv` as a `.npy` file.


Usage example:
```
$ conda activate coordinpaint
$ python convert_uv_render.py --sample_id=WOMEN#Blouses_Shirts#id_00000442#01_2_side
```  

`infer_sample.py` takes a source rgb image and a target uv render to produce an image of source person in target pose. 
Rusulting images are saved to `data/results`.

```
$ conda activate coordinpaint
$ python infer_sample.py --source_sample=WOMEN#Blouses_Shirts#id_00000442#01_2_side --target_sample=WOMEN#Dresses#id_00000106#03_1_front
```

## Citation
This repository contains code corresponding to:

A. Grigorev, A. Sevastopolsky, A. Vakhitov, and V. Lempitsky.
**Coordinate-based texture inpainting for pose-guided human image generation**. In
*IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

Please cite as:

```
@inproceedings{grigorev2019coordinate,
  title={Coordinate-based texture inpainting for pose-guided human image generation},
  author={Grigorev, Artur and Sevastopolsky, Artem and Vakhitov, Alexander and Lempitsky, Victor},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12135--12144},
  year={2019}
}
```
