# Fine-Grained Cell Segmentation using a Pathology Foundation Model

**Author:** Ying Yuan  
**Role Applied:** Sr. Scientist (Biostatistics) @ Merck  
**Date:** October 2025


## Problem

Accurate segmentation of individual blood cells in microscopy images is critical for clinical analysis (e.g., cell counting, morphology assessment). The core challenges are:

- **Touching/overlapping cells**: densely packed cells share boundaries that are hard to separate
- **Staining variation**: image appearance varies across slides and preparation conditions
- **Semantic vs. instance gap**: pixel-level overlap metrics (IoU/Dice) alone are insufficient — each cell must be individually identified (mAP)


## Dataset

[BCCD Dataset](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask?resource=download) — blood cell microscopy images with binary instance mask annotations.

- All original train/test data merged and re-split: **70% train / 15% val / 15% test**
- Preprocessing: resize to 512×512, channel-wise z-score stain normalization (approximating Macenko), random 512×512 patch extraction
- Augmentation: HorizontalFlip, VerticalFlip, RandomRotate90, RandomBrightnessContrast, HueSaturationValue, GaussianBlur — via Albumentations


## Model Design: UNI-Adapted UNet

Recent pathology foundation models (e.g., **UNI**) show strong generalization across tissue types. Here I adapt the foundation model paradigm for **fine-grained cell instance segmentation**, using a ResNet50 backbone as a UNI-like encoder coupled with a UNet-style decoder.

| Component | Description |
|-----------|-------------|
| **Encoder** | ResNet50 pretrained on ImageNet — high-capacity backbone analogous to the UNI feature extractor; outputs 2048-ch feature map at 1/32 input resolution |
| **Decoder** | 5-stage ConvTranspose2d upsampling (16×16 → 512×512) with BatchNorm + ReLU, recovering spatial detail while preserving global context |
| **Segmentation Head** | 1×1 conv → single-channel mask, trained with 0.5×BCE + 0.5×Dice loss |
| **Instance Embedding Head** | Parallel 1×1 conv → 16-ch embedding map; pixels of the same cell cluster in feature space to support instance separation |

This design balances **generalizable feature extraction** (ResNet/UNI encoder) and **fine-grained morphological resolution** (UNet decoder), following the principle of repurposing a general-purpose foundation model for high-resolution pathology prediction.


## Training

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 110 |
| Batch Size | 64 |
| Input Size | 512×512 |
| Learning Rate | 1e-4 |
| Optimizer | Adam |
| Scheduler | Cosine Annealing LR |
| Loss | 0.5 × BCE + 0.5 × Dice |
| Precision | AMP (torch.amp.autocast + GradScaler) |
| Hardware | 2× NVIDIA RTX 4090 (DataParallel) |

- AMP reduced memory ~35% and gave ~1.4× speedup with no precision loss
- Checkpoints saved every 2 epochs; full evaluation every 5 epochs


## Post-processing: Watershed Instance Separation

Raw model output merges touching cells into blobs. To recover individual instances:
1. **Morphological closing** to fill small holes in predicted masks
2. **Distance transform** on the binary mask to estimate cell centers
3. **Watershed segmentation** using local maxima as seeds to split overlapping cells

| Metric | Before | After | Δ |
|--------|--------|-------|---|
| IoU | 0.933 | 0.928 | −0.005 |
| Dice | 0.964 | 0.961 | −0.003 |
| **mAP@0.5** | 0.761 | **0.763** | **+0.001** |

Post-processing improves instance-level detection (mAP) at a minor cost to pixel-level overlap — confirming the semantic–instance gap.


## Results

| Split | IoU | Dice | mAP@0.5 |
|-------|-----|------|---------|
| Train | 0.9401 | 0.9685 | 0.7597 |
| Validation | 0.9390 | 0.9673 | 0.7686 |
| **Test** | **0.9283** | **0.9705** | **0.7727** |

- IoU/Dice variation < 1.5% across splits → strong generalization, no overfitting
- Lower mAP reflects residual under-segmentation of dense, touching cell clusters

**Training curves:**

![Training Curves](Merck_YingYuan_PreInterview_Assignment/figures/new_training_curves.png)

**Performance across splits:**

![Dataset Comparison](Merck_YingYuan_PreInterview_Assignment/figures/dataset_comparison.png)

**Segmentation examples** (original → ground truth → raw prediction → post-processed):

![Watershed Post-processing](Merck_YingYuan_PreInterview_Assignment/figures/postprocess_watershed.png)

![Visualization Example](Merck_YingYuan_PreInterview_Assignment/figures/vis_example_8.png)

**Metric reference:**

![Metric Visualization](Merck_YingYuan_PreInterview_Assignment/figures/metric_visualization.png)


## Failure Cases

- **Clustered cells**: adjacent lymphocytes sometimes merge into a single blob — preserves Dice but lowers mAP
- **Weak staining / color drift**: hue variability causes under-segmentation of pale nuclei
- **Boundary artifacts**: decoder upsampling produces blurred edges, slightly underestimating IoU


## Reflections & Future Work

- Foundation models like UNI excel at morphology encoding (high Dice) but need instance-aware refinement to maximize mAP — boundary loss (Boundary IoU, contour Dice) or embedding clustering could close this gap
- **CellPose** as a post-hoc instance refiner: takes UNet binary mask as input, produces separated cell instances without retraining
- **SAM** for boundary-aware refinement: UNet output used as a soft prompt to correct incomplete contours
- **LLM-guided QA**: GPT-4V / LLaVA-Med to identify merged or missing nuclei and suggest corrections
- **LLM segmentation agent**: LangChain pipeline combining UNet + CellPose + LLM for end-to-end automated segmentation, metric selection, and reporting


## Repository Structure

```
├── Merck_YingYuan_PreInterview_Assignment/
│   ├── YingYuan_CellSegmentation.ipynb   # main notebook
│   ├── environment.yml                    # conda environment
│   ├── requirements.txt                   # pip dependencies
│   └── figures/                           # all result figures
├── results/                               # full evaluation outputs
│   ├── full_evaluation_report.csv
│   ├── final_report_summary.md
│   └── figures/
└── datasets/                              # not tracked (too large)
```


## Setup

```bash
# conda
conda env create -f Merck_YingYuan_PreInterview_Assignment/environment.yml
conda activate <env_name>

# or pip
pip install -r Merck_YingYuan_PreInterview_Assignment/requirements.txt
```

Then open `Merck_YingYuan_PreInterview_Assignment/YingYuan_CellSegmentation.ipynb`.


## References

1. Chen et al., *Nature Medicine* (2024) — Towards a general-purpose foundation model for computational pathology
2. Stringer et al., *Nature Methods* (2020) — Cellpose: a generalist algorithm for cellular segmentation
3. Kirillov et al., arXiv (2023) — Segment Anything
4. Ma et al., *Nature Communications* (2024) — Segment Anything in Medical Images
5. BCCD Dataset — [Kaggle](https://www.kaggle.com/datasets/jeetblahiri/bccd-dataset-with-mask?resource=download)
