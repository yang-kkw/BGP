
# Multimodal Learning for Cover-Based Book Genre Prediction

This repository contains the implementation of a CLIP-Large-based multimodal framework for cover-based book genre prediction. The model uses book cover images, book titles, and OCR text as inputs. Visual and textual features are extracted by CLIP-Large, adapted with LoRA-enhanced projection layers, recalibrated by an SE block, and finally classified by an MLP classifier.

## Overview

Book covers contain rich visual and textual information. Image-only methods may suffer from visual ambiguity and unclear category boundaries, while simple multimodal fusion methods may not fully exploit the relationship between cover images and textual information.

To address these issues, this project uses a CLIP-Large-based multimodal framework. The cover image is encoded by the CLIP visual encoder, while the book title and OCR text are used to construct a text prompt and encoded by the CLIP text encoder. LoRA is introduced into the visual and textual projection layers to reduce fine-tuning cost, and an SE block is used to recalibrate the concatenated multimodal features.

## Repository Structure

```text
BGP/
├── dataset/                                  # Sample annotation files and category mapping
├── train_main.py                             # Main training script
├── extract_ocr.py                            # OCR extraction script
├── evaluate_per_class.py                     # Per-class Recall and F1 evaluation
├── ablation_experiment.py                     # Ablation experiment script
├── no_lora_no_se_ablation_experiment.py       # Ablation without LoRA and SE block
├── category.py                                # Category/statistical processing script
├── .gitignore
└── README.md
```

## Method

The proposed model takes a book cover image and a text prompt as inputs. The text prompt is constructed from the book title and OCR text:

```text
a photo of a book cover titled {title}. text: {ocr_text}
```

The image and text are encoded by CLIP-Large. LoRA is introduced into the visual and textual projection layers. During fine-tuning, only the last two layers of the visual encoder and the text encoder are unfrozen.

The normalized image and text features are concatenated and then passed through an SE block for channel-wise feature recalibration. Finally, an MLP classifier is used to predict the book genre.

## Datasets

Experiments are conducted on two public datasets:

- Kaggle Amazon Books Dataset
- BookCover30

The full datasets, book cover images, processed JSON files, pretrained weights, and trained checkpoints are not included in this repository due to file size and license restrictions.

This repository only provides sample annotation files and category mapping files in the `dataset/` directory to show the required data format.

## Kaggle Category Reorganization

The original Kaggle Amazon Books Dataset was cleaned and reorganized before training. Some original categories were merged or renamed into 14 final genres, while noisy or irrelevant categories were removed.

The category mapping file is provided in:

```text
dataset/kaggle_category_mapping.json
```

The final Kaggle genres used in this work are:

```text
Art
Business
Children
Comics & YA
Education
Genre Fiction
Health
History
Lifestyle
Literary Fiction
Reference
Religion
Romance
Science
```

Each processed sample follows the format:

```json
{
  "image_name": "example.jpg",
  "title": "book title",
  "ocr_text": "recognized cover text",
  "category_final": "Science",
  "label": 13
}
```

For BookCover30, the original category labels are used. A sample annotation file is also provided in the `dataset/` directory.

## Environment

The experiments were conducted under the following environment:

```text
Ubuntu 20.04.5
Python 3.x
PyTorch 2.1.2
CUDA 11.8
NVIDIA RTX 4090
```

Main dependencies include:

```text
torch
torchvision
transformers
scikit-learn
numpy
Pillow
easyocr
opencv-python-headless
pandas
tqdm
```

PyTorch should be installed according to the local CUDA environment. In our experiments, PyTorch 2.1.2 with CUDA 11.8 was used.

## Data Preparation

Before training, please download the original datasets and organize the image files and annotation files according to your local environment.

The dataset paths and CLIP-Large path should be modified in `train_main.py`.

Example path settings:

```python
DATASETS = [
    {
        "name": "Kaggle",
        "json_path": "path/to/kaggle_processed.json",
        "img_root": "path/to/kaggle_images"
    },
    {
        "name": "BookCover30",
        "json_path": "path/to/bookcover30_processed.json",
        "img_root": "path/to/bookcover30_images"
    }
]

clip_path = "path/to/clip-large"
```

## OCR Extraction

OCR text is extracted from book cover images and used together with the book title to construct the text prompt.

To extract OCR text, run:

```bash
python extract_ocr.py
```

Please modify the image path and output JSON path in the script before running it.

## Training

To train the proposed CLIP-Large + LoRA + SE model, run:

```bash
python train_main.py
```

Main training settings:

```text
Image size: 224 × 224
Batch size: 8
Gradient accumulation steps: 4
Effective batch size: 32
Maximum epochs: 25
Optimizer: AdamW
Weight decay: 3e-2
Label smoothing: 0.15
LoRA rank: 8
LoRA alpha: 16
SE reduction ratio: 16
```

## Ablation Experiments

To run ablation experiments, use:

```bash
python abltion_experiment.py
```

For the setting without both LoRA and SE block, run:

```bash
python no_lora_no_se_abltion_experiment.py
```

The ablation experiments are used to analyze the effects of OCR text, LoRA, and the SE block.

## Per-Class Evaluation

To evaluate per-class Recall and F1 scores, run:

```bash
python evaluate_per_class.py
```

This script is used to analyze the classification performance of each book genre.

## Experimental Results

The proposed method achieves the following overall performance:

| Dataset                     | Accuracy (%) | Macro-F1 (%) |
| --------------------------- | -----------: | -----------: |
| Kaggle Amazon Books Dataset |        70.44 |        62.04 |
| BookCover30                 |        70.80 |        70.72 |

The results show that the proposed method achieves better overall performance than several image-only, multimodal, and CLIP-based baseline methods.

## Notes

- Full datasets and book cover images are not included in this repository.
- Processed full JSON files are not included due to file size limitations.
- Pretrained CLIP-Large weights should be downloaded separately.
- Trained checkpoints are not included.
- The sample JSON files are only used to illustrate the required annotation format.
- The Kaggle category mapping file is provided to explain how the original categories were reorganized into the final 14 genres.

## Citation

If you find this project helpful, please cite this work:

```bibtex
@article{liu2026bgp,
  title={Multimodal Learning for Cover-Based Book Genre Prediction},
  author={Liu, Xuelin and Yao, Yangchun and Yan, Jiebin and Fang, Chengyang and Fang, Yuming},
  journal={},
  year={2026}
}
```
