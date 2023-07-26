# DDPS: Denoising Diffusion Prior for Segmentation

Zeqiang Lai*, Yuchen duan*, Jifeng Dai, Ziheng Li, Ying Fu, Hongsheng Li, Yu Qiao, and Wenhai Wang

[[`Paper`](https://arxiv.org/abs/2306.01721)] [[`Demo`](https://github.com/OpenGVLab/InternGPT)] [[`Bibtex`](#citation)]

PyTorch implementation and pretrained models for **DDPS** (Denoising Diffusion Prior for Segmentation). For details, see paper: [**Denoising Diffusion Semantic Segmentation with Mask Prior Modeling**](https://arxiv.org/abs/2306.01721).

> DDPS explore the mask prior modeled by a recently developed denoising diffusion generative model for ameliorating the semantic segmentation quality of existing discriminative approaches. DDPS focuses on a discrete instantiation of a unified architecture that adapts diffusion models for mask prior modeling, with two novel training and inference strategies, i.e., `Diffusion-on-First-Prediction` diffusion strategy and `Free-Re-noising` denoising strategy.

<img width="700" alt="arch" src="https://github.com/OpenGVLab/DDPS/assets/26198430/769ad1f2-d5b9-442e-bd0a-9211be705dc1">

> Working in progress.

## News

- **[TBD]** Integrate an online demo into [InternGPT](https://github.com/OpenGVLab/InternGPT).
- **[2023/7/26]** Release preview code and pretrained models.
- **[2023/6/2]** Release paper to arXiv. 

## DDPS Model

Semantic segmentation performance on ADE20K.

| Method          | Backbone     | Params (M) | mIoU (ss) | mIoU (ms) | mBIoU | Download       |
| --------------- | ------------ | ---------- | --------- | --------- | ----- | -------------- |
| DDPS-DeeplabV3+ | MobibleNetV2 | 27.9       | 38.07     | 39.07     | 23.47 | [TBD]() |
| DDPS-Segformer  | MiT-B0       | 15.1       | 41.67     | 41.95     | 26.77 | [TBD]() |
| DDPS-DeeplabV3+ | ResNet50     | 54.2       | 45.26     | 46.01     | 30.39 | [checkpoint](https://huggingface.co/aaronb/DDPS-all/tree/main/deeplabv3plus_r50_multistep) |
| DDPS-Segformer  | MiT-B2       | 65.5       | 49.73     | 49.96     | 33.95 | [checkpoint](https://huggingface.co/aaronb/DDPS-all/tree/main/segformer_b2_multistep) |
| DDPS-DeeplabV3+ | ResNet101    | 104.3      | 46.32     | 46.99     | 31.82 | [checkpoint](https://huggingface.co/aaronb/DDPS-all/tree/main/deeplabv3plus_r101_multistep) |
| DDPS-Segformer  | MiT-B5       | 122.8      | 51.11     | 51.71     | 35.66 | [TBD]() |

Semantic segmentation performance on Cityscapes.

| Method          | Backbone     | Params (M) | mIoU (ss) | mIoU (ms) | mBIoU | Download       |
| --------------- | ------------ | ---------- | --------- | --------- | ----- | -------------- |
| DDPS-DeeplabV3+ | MobibleNetV2 | 27.9       | 78.11     | 79.73     | 61.14 | [TBD]() |
| DDPS-Segformer  | MiT-B0       | 15.1       | 78.19     | 79.62     | 64.07 | [checkpoint](https://huggingface.co/aaronb/DDPS-all/tree/main/cityscapes/segformer_b0_multistep) |
| DDPS-DeeplabV3+ | ResNet50     | 54.2       | 81.39     | 82.27     | 65.76 | [checkpoint](https://huggingface.co/aaronb/DDPS-all/tree/main/cityscapes/deeplabv3plus_r50_multistep) |
| DDPS-Segformer  | MiT-B2       | 65.5       | 81.77     | 82.22     | 67.75 | [checkpoint](https://huggingface.co/aaronb/DDPS-all/tree/main/cityscapes/segformer_b2_multistep) |
| DDPS-DeeplabV3+ | ResNet101    | 104.3      | 81.68     | 82.51     | 65.96 | [checkpoint](https://huggingface.co/aaronb/DDPS-all/tree/main/cityscapes/deeplabv3plus_r101_multistep) |
| DDPS-Segformer  | MiT-B5       | 122.8      | 82.42     | 82.94     | 66.45 | [TBD]() |

## Installation

See [environment.yaml](environment.yaml) for details.

## Usage

**Evaluation**

```bash
sh tools/dist_test_diffusion_origin.sh configs/ade20k/segformer_b2_ade20k_multistep.py checkpoints/ade20k/segformer_b2_multistep/best_mIoU_iter_144000.pth  8 --eval "mIoU"
```

We are still cleaning up the code and checkpoints, so the results might slightly differ from the reported ones, but you should expect the following results with the above command.

```
+-------------------+
|    mIoU 0,10,19   |
+-------------------+
| 49.35,49.55,49.66 |
+-------------------+
```


## Visualization

Mask Prior Modeling

<img width="450" alt="prior" src="https://github.com/OpenGVLab/DDPS/assets/26198430/3bec572b-c2b5-4094-9fdb-b9f3fcf41333">

## Citation

If you find this work helpful, please consider cite our paper.

```bibtex
@misc{lai2023ddps,
  title = {Denoising Diffusion Semantic Segmentation with Mask Prior Modeling},
  author = {Zeqiang Lai and Yuchen duan and Jifeng Dai and Ziheng Li and Ying Fu and Hongsheng Li and Yu Qiao and Wenhai Wang},
  publisher = {arXiv},
  year = {2023},
}
```

## Acknowledgement

[MMSegmentation](https://github.com/open-mmlab/mmsegmentation) &ensp;  [VQDiffusion](https://github.com/cientgu/VQ-Diffusion)  &ensp;  [Segformer](https://github.com/NVlabs/SegFormer) &ensp; [Diffusers](https://github.com/huggingface/diffusers)

