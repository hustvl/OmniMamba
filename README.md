<div align ="center">
<h1>☯OmniMamba </h1>
<h3>Efficient and Unified Multimodal Understanding and Generation 
  
via State Space Models</h3>

[Jialv Zou](https://github.com/Doctor-James)<sup>1</sup>, [Bencheng Liao](https://github.com/LegendBC)<sup>2,1</sup>, [Qian Zhang](https://scholar.google.com/citations?user=pCY-bikAAAAJ&hl=zh-CN)<sup>3</sup>, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/)<sup>1</sup>, [Xinggang Wang](https://xwcv.github.io/)<sup>1,📧</sup>

<sup>1</sup>  School of EIC, HUST, <sup>2</sup>  Institute of Artificial Intelligence, HUST,   <sup>3</sup> Horizon Robotics

(<sup>📧</sup>) corresponding author.


[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2503.08686)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

<div align="center">
<img src="./assets/teaser.png">
</div>

## 📋 News
`[2025-3-19]:` We release the initial version of code and weight, along with documentation and training/inference scripts.

`[2025-3-12]:` OmniMamba [arXiv](https://arxiv.org/abs/2503.08686) paper released. Code and Weight are coming soon. Please stay tuned! ☕️

## ✨ Highlights
* To the best of our knowledge, OmniMamba is the first linear model based unified multimodal understanding and visual generation model.
* OmniMamba achieves competitive performance with only 2M data for training.
* OmniMamba is highly efficient, achieving up to a 119.2 times speedup and 63\% GPU memory reduction for long-sequence generation compared to Transformer-based counterparts.

## 🛠️ Architecture

</div>

<div align="center">
<img src="./assets/arch.png">
</div>

## 📊 Qualitative Results


<div align="center">
<img src="./assets/vis.png">
</div>

<!-- ## [Main Results](docs/Results.md)

## Getting Started -->
## 🏁 Getting Started

- [Install](#install)
- [Inference](#inference)
- [Prepare data](#prepare-data)
- [Train](#train)

<!-- - [Evaluation](docs/Evaluation.md) -->


### Install

1. Clone this repository and navigate to OmniMamba folder
```bash
git clone https://github.com/hustvl/OmniMamba
cd OmniMamba
```

2. Install Package
```Shell
# Install PyTorch (with CUDA 11.8) before everything else. those assume you are using cu118
conda create -n omnimamba python=3.10 -y
conda activate omnimamba
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
wandb login
```

### Inference
Please download our pretrained model at [OmniMamba-1.3b](https://huggingface.co/Doctor-James/OmniMamba)

Multimodal Understanding
```
python scripts/inference_mmu.py --image_path mmu_validation/cat_dog.png --question 'Please describe it in detail.'
```
Visual Generation
```
python scripts/inference_t2i.py --prompt 'A bed in a bedroom between two lamps.'
```

### Prepare data
ShareGPT4V: Please refer to the [document](https://tinyllava-factory.readthedocs.io/en/latest/Prepare%20Datasets.html#id2) of the [TinyLLaVA](https://github.com/TinyLLaVA/TinyLLaVA_Factory) to download the dataset without SAM for pretrain.

LVIS-Instruct-4V: Please refer to the [document](https://huggingface.co/datasets/X2FD/LVIS-Instruct4V)

LRV-Instruct: Please refer to the [document](https://github.com/FuxiaoLiu/LRV-Instruction)

You can download our preprocessed JSON file on [Hugging Face](https://huggingface.co/Doctor-James/OmniMamba).

Folder structure
```
OmniMamba
├── dataset/
│   ├── pretokenized_coco_train2014.jsonl
│   ├── llava/
│   │   ├── gqa/
│   │   ├── LAION-CC-SBU/
│   │   ├── ocr_vqa/
│   │   ├── POPE/
│   │   ├── share_textvqa/
│   │   ├── textvqa/
│   │   ├── vg/
│   │   ├── web-celebrity/
│   │   ├── web-landmark/
│   │   ├── wikiart/
│   │   ├── coco/
│   │   ├── lrv_Instruct/
│   │   ├── share-captioner_coco_lcs_676k_1121.json
│   │   ├── sharegpt4v_llava_v1_5_lvis4v_lrv_mix1231k.json
```

### Train
Stage 1: MMU Pre-Training
```
accelerate launch --mixed_precision=bf16 --machine_rank=0 --num_processes=8 --num_machines=1 --main_process_port=8888 train_stage2.py --config config/config_stage1_mmu.yaml
```

Stage 1: T2I Pre-Training
```
accelerate launch --mixed_precision=bf16 --machine_rank=0 --num_processes=8 --num_machines=1 --main_process_port=8888 train_stage2.py --config config/config_stage1_t2i.yaml
```

Stage 2: Unifid Fine-Tuning
```
accelerate launch --mixed_precision=bf16 --machine_rank=0 --num_processes=8 --num_machines=1 --main_process_port=8888 train_stage2.py --config config/config_stage2.yaml
```

## ❤️ Acknowledgements
We build our project based on
- [Mamba](https://github.com/state-spaces/mamba)
- [Cobra](https://github.com/h-zhao1997/cobra)
- [AiM](https://github.com/hp-l33/AiM)
- [Show-o](https://github.com/showlab/Show-o)
- [LLaVA](https://github.com/haotian-liu/LLaVA)

Thanks for their great works.

## 📚 Citation
If you find OmniMamba useful in your research or applications, please consider giving us a star &#127775; and citing it by the following BibTeX entry.


```bibtex
@misc{zou2025omnimambaefficientunifiedmultimodal,
      title={OmniMamba: Efficient and Unified Multimodal Understanding and Generation via State Space Models}, 
      author={Jialv Zou and Bencheng Liao and Qian Zhang and Wenyu Liu and Xinggang Wang},
      year={2025},
      eprint={2503.08686},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.08686}, 
}
```
