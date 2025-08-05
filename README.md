# LISA: Reasoning Segmentation via Large Language Model



**LISA: Reasoning Segmentation via Large Language Model [[Paper](https://arxiv.org/abs/2308.00692)]** <br />


## Abstract
LISA, short for Language Instructed Segmentation Assistant, is a new kind of artificial intelligence system that can understand both pictures and language together. Unlike many computer programs that can only recognize objects they were trained to see, LISA can understand new and unfamiliar objects by using language to guide its thinking. It works by taking a natural language instruction, such as "find the object used for cooking," and then looks at an image to find the most relevant part. LISA combines the ability to see with the ability to read and reason. This makes it useful in many real-life situations, such as helping robots understand their surroundings or helping users search for specific items in complex images.

## Highlights
**LISA** unlocks the new segmentation capabilities of multi-modal LLMs, and can handle cases involving: 
1. complex reasoning; 
2. world knowledge; 
3. explanatory answers; 
4. multi-turn conversation. 


## Installation
```
git clone https://github.com/sontung2310/LISA.git
cd LISA
conda create -n lisa python=3.9
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```


## Inference 

```
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1'
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1-explanatory'
```
To use `bf16` or `fp16` data type for inference:
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1' --precision='bf16'
```
To use `8bit` or `4bit` data type for inference (this enables running 13B model on a single 24G or 12G GPU at some cost of generation quality):
```
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1' --precision='fp16' --load_in_8bit
CUDA_VISIBLE_DEVICES=0 python chat.py --version='xinlai/LISA-13B-llama2-v1' --precision='fp16' --load_in_4bit
```
Hint: for 13B model, 16-bit inference consumes 30G VRAM with a single GPU, 8-bit inference consumes 16G, and 4-bit inference consumes 9G.

After that, input the text prompt and then the image path. For example，
```
- Please input your prompt: Where can the driver see the car speed in this image? Please output segmentation mask.
- Please input the image path: imgs/example1.jpg

- Please input your prompt: Can you segment the food that tastes spicy and hot?
- Please input the image path: imgs/example2.jpg
```
The results should be like:
<p align="center"> <img src="imgs/example1.jpg" width="22%"> <img src="vis_output/example1_masked_img_0.jpg" width="22%"> <img src="imgs/example2.jpg" width="25%"> <img src="vis_output/example2_masked_img_0.jpg" width="25%"> </p>

To run the gradio interface (load in 8bit and fp16):
```
CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python gradio_lisa.py 
```



## Citation 
If you find this project useful in your research, please consider citing:

```
@article{lai2023lisa,
  title={LISA: Reasoning Segmentation via Large Language Model},
  author={Lai, Xin and Tian, Zhuotao and Chen, Yukang and Li, Yanwei and Yuan, Yuhui and Liu, Shu and Jia, Jiaya},
  journal={arXiv preprint arXiv:2308.00692},
  year={2023}
}
@article{yang2023improved,
  title={An Improved Baseline for Reasoning Segmentation with Large Language Model},
  author={Yang, Senqiao and Qu, Tianyuan and Lai, Xin and Tian, Zhuotao and Peng, Bohao and Liu, Shu and Jia, Jiaya},
  journal={arXiv preprint arXiv:2312.17240},
  year={2023}
}
```

## Acknowledgement
-  This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA) and [SAM](https://github.com/facebookresearch/segment-anything). 
