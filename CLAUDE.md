# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python tool that detects the orientation (0°, 90°, 180°, 270°) of pages in PDF documents or image files using a fine-tuned ResNet18 CNN. Outputs JSON mapping page numbers to their rotation deviation from upright.

## Setup

Python 3.12+ is required. The Python executable is at `C:\Users\tyki\AppData\Local\Programs\Python\Python312\python.exe` (not on PATH).

```bash
pip install -r requirements.txt
```

Dependencies: PyTorch (CPU), torchvision, PyMuPDF, Pillow, NumPy. Use `--index-url https://download.pytorch.org/whl/cpu` for torch to avoid pulling CUDA wheels.

## Commands

```bash
# Train model from a correctly-oriented reference PDF
python orient_detect.py --train reference.pdf

# Detect orientation of a PDF or image
python orient_detect.py input.pdf
python orient_detect.py input.pdf -o result.json

# Train then detect in one run
python orient_detect.py --train reference.pdf input.pdf

# Force heuristic-only mode (no model)
python orient_detect.py --heuristic input.pdf
```

## Architecture

Single-file application (`orient_detect.py`) with three components:

1. **`OrientationResNet18`** (nn.Module): ResNet18 with ImageNet-pretrained convolutional layers frozen, replaced by a trainable 4-class FC layer (512→4). Input is 224×224 normalized RGB. Saved model weights go to `orientation_resnet18.pth` (~44MB).

2. **`train_model()`**: Self-supervised training pipeline. Renders a reference PDF's pages as upright examples, generates 4 rotated versions per page (PIL `rotate()` which is CCW), augments with horizontal flips, fine-tunes only the FC layer for 30 epochs.

3. **`OrientationDetector`**: Loads the trained model and predicts per-page orientation. Falls back to `_text_projection_score()` heuristic (horizontal/vertical projection profile variance) when no model file exists.

### Rotation Convention

This is critical and non-obvious:

- **PIL `Image.rotate(angle)`** rotates **counterclockwise**.
- Training labels: `enumerate([0, 90, 180, 270])` → labels 0–3 correspond to CCW rotations applied to upright images.
- **Output convention**: Reports how many degrees CW the page is rotated from upright.
- `LABEL_TO_ANGLE = {0: 0, 1: 270, 2: 180, 3: 90}` — labels 1 and 3 are swapped because CCW 90° = CW 270°.

### PDF Rendering

PyMuPDF (`fitz`) renders pages via `page.get_pixmap(matrix=mat)`, which **applies the PDF's built-in rotation attribute**. The rendered image dimensions reflect the MediaBox adjusted by rotation. For scanned PDFs, there is no extractable text — orientation must be determined from pixel content alone.

## Key Design Decisions

- **Why fine-tune from a reference PDF?**: Pure heuristic methods (projection profiles) can distinguish horizontal vs vertical text but cannot differentiate 0° from 180° or 90° from 270° due to projection symmetry. Training on the actual document type solves this.
- **Why freeze convolutional layers?**: With only ~180 training samples (23 pages × 4 rotations × 2 augmented), fine-tuning only the FC layer prevents overfitting while still achieving >99% accuracy.
