# Image Orientation Correction

A tool for detecting the orientation of pages in PDF documents or image files.
Uses a fine-tuned ResNet18 convolutional neural network to classify each page's
rotation as 0°, 90°, 180°, or 270° from its correct upright orientation.

## Setup

Install Python 3.12+, then install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model (using a correctly-oriented reference PDF)

```bash
python orient_detect.py --train reference.pdf
```

This renders all pages from the reference PDF, generates 4 rotated versions
of each page (0°, 90°, 180°, 270°), and fine-tunes the ResNet18 FC layer.
The trained model is saved as `orientation_resnet18.pth`.

### 2. Detect orientation

```bash
# For a PDF file:
python orient_detect.py input.pdf

# For an image file:
python orient_detect.py image.png

# Save results to JSON file:
python orient_detect.py input.pdf -o result.json

# Force heuristic mode (no model):
python orient_detect.py --heuristic input.pdf
```

Output format (JSON):
```json
{
  "1": 90,
  "2": 180,
  "3": 0,
  "4": 0
}
```

Where each value indicates how many degrees the page is rotated from
its correct upright orientation.

### Train and detect in one command

```bash
python orient_detect.py --train reference.pdf input.pdf
```

## Architecture

- **Model**: ResNet18 (ImageNet pre-trained) with custom 4-class FC layer
- **Training**: Freezes all convolutional layers, fine-tunes only the FC layer
- **Data augmentation**: Horizontal flip, small translation, color jitter
- **Fallback**: Text projection profile heuristic when no model is available

## Dependencies

- PyTorch + torchvision (ResNet18 model)
- PyMuPDF (PDF rendering)
- Pillow (image processing)
- NumPy (numerical operations)
