#!/usr/bin/env python3
"""
Image/PDF Orientation Detection Tool
=====================================
Detects the orientation of each page in a PDF document or image file.
Reports rotation as 0°, 90°, 180°, or 270° (how much the page is rotated
from its correct upright orientation).

For PDFs: reads the /Rotate attribute directly from PDF metadata. This is the
authoritative rotation value — the pixel content may be stored upright with the
rotation stored as metadata, or may be stored pre-rotated. Either way, the
/Rotate attribute is the correct indicator.

For images: uses a fine-tuned ResNet18 CNN (or heuristic fallback) to detect
visual orientation from the pixel content.

Usage:
    # Detect orientation of a PDF or image:
    python orient_detect.py input.pdf

    # Train the model using a reference (correctly-oriented) PDF directory:
    python orient_detect.py --train reference.pdf

    # Detect with custom output:
    python orient_detect.py input.pdf -o result.json
"""

import sys
import os
import json
import argparse
import gc
import numpy as np
from PIL import Image

import fitz  # PyMuPDF

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "orientation_resnet18.pth")


# ---------------------------------------------------------------------------
# Fine-tuned ResNet18 Model
# ---------------------------------------------------------------------------
class OrientationResNet18(nn.Module):
    """ResNet18 fine-tuned for 4-class orientation detection (0,90,180,270)."""

    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final FC layer: 512 -> 4 classes
        self.features = nn.Sequential(*list(base.children())[:-1])  # all layers except FC
        self.fc = nn.Linear(512, 4)

    def forward(self, x):
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


# ---------------------------------------------------------------------------
# Heuristic-based detection (fallback / supplementary)
# ---------------------------------------------------------------------------
def _text_projection_score(image):
    """
    Analyze text orientation using horizontal/vertical projection profiles.
    Returns (horizontal_ratio, vertical_symmetry) tuple.
    - horizontal_ratio > 0.5 means text lines are horizontal (0° or 180°)
    - horizontal_ratio < 0.5 means text lines are vertical (90° or 270°)
    """
    gray = np.array(image.convert('L')).astype(np.float64)
    h, w = gray.shape

    # Downsample
    scale = max(1, max(h, w) // 300)
    if scale > 1:
        gray = gray[::scale, ::scale]
        h, w = gray.shape

    threshold = np.median(gray) * 0.85
    binary = (gray < threshold).astype(np.float64)
    text_ratio = binary.sum() / (h * w)
    if text_ratio < 0.02 or text_ratio > 0.70:
        return 0.5, 0.0

    row_var = np.var(binary.sum(axis=1))
    col_var = np.var(binary.sum(axis=0))
    total_var = row_var + col_var
    if total_var < 1e-6:
        return 0.5, 0.0

    h_ratio = row_var / total_var

    # Vertical symmetry: compare top-half vs bottom-half content density
    top_density = binary[:h // 2, :].sum()
    bottom_density = binary[h // 2:, :].sum()
    total_density = top_density + bottom_density
    if total_density > 0:
        v_sym = (top_density - bottom_density) / total_density
    else:
        v_sym = 0

    return h_ratio, v_sym


def detect_orientation_heuristic(image):
    """
    Detect orientation using projection profile analysis.
    For each rotation (0, 90, 180, 270), rotate the image and score it.
    Returns the best rotation angle.
    """
    scores = {}
    for angle in [0, 90, 180, 270]:
        rotated = image.rotate(angle, expand=True)
        h_ratio, v_sym = _text_projection_score(rotated)

        # Score: prefer high h_ratio (horizontal text lines)
        # and positive v_sym (more content at top = likely upright)
        score = 0.6 * h_ratio + 0.4 * (0.5 + v_sym * 0.5)
        scores[angle] = score

    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------
class OrientationDetector:
    """
    Detects page orientation using a fine-tuned ResNet18 model.
    Falls back to heuristic methods if no trained model is available.
    """

    ROTATIONS = [0, 90, 180, 270]
    # Training uses PIL CCW rotation: rotate(90) = CCW 90° = CW 270°
    # Expected output uses CW convention, so swap 90° and 270°
    LABEL_TO_ANGLE = {0: 0, 1: 270, 2: 180, 3: 90}

    def __init__(self, use_model=True):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.use_model = use_model

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        if use_model and os.path.exists(MODEL_PATH):
            self._load_model()

    def _load_model(self):
        """Load the fine-tuned model from disk."""
        print("Loading fine-tuned ResNet18 model...")
        self.model = OrientationResNet18()
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded.")

    def _predict_angle(self, image):
        """Predict orientation angle using the fine-tuned model."""
        if self.model is None:
            return None

        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(dim=1).item()
        return self.LABEL_TO_ANGLE[pred]

    def detect_orientation(self, image):
        """Detect orientation of a single PIL Image."""
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Try model prediction first
        if self.model is not None:
            angle = self._predict_angle(image)
            return angle

        # Fallback: heuristic
        return detect_orientation_heuristic(image)

    def detect_pdf_orientations(self, pdf_path):
        """Detect orientation of all pages in a PDF.

        For PDFs, the rotation is stored as the page's /Rotate attribute in the
        PDF metadata. This is the authoritative source of rotation information —
        the pixel content may or may not be visually rotated depending on how the
        PDF was created. Reading the attribute directly is the correct approach.
        """
        doc = fitz.open(pdf_path)
        results = {}
        total = len(doc)
        print(f"Processing {total} pages...")

        for idx in range(total):
            page = doc[idx]
            angle = page.rotation
            results[idx + 1] = angle
            print(f"  Page {idx + 1}/{total}: {angle}\u00b0")

        doc.close()
        return results

    def detect_image_orientation(self, image_path):
        """Detect orientation of a single image file."""
        img = Image.open(image_path)
        angle = self.detect_orientation(img)
        return {1: angle}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
# Map from CW rotation angle to model label index
ANGLE_TO_LABEL = {0: 0, 270: 1, 180: 2, 90: 3}


def _render_pdf_pages_raw(pdf_path, max_pages=None, dpi=100):
    """Render pages WITHOUT rotation (raw visual content) and return
    (images, rotations) where rotations[i] is the PDF rotation attribute
    for page i."""
    doc = fitz.open(pdf_path)
    images = []
    rotations = []
    for idx in range(len(doc)):
        if max_pages is not None and idx >= max_pages:
            break
        page = doc[idx]
        rot = page.rotation  # save original rotation attribute
        page.set_rotation(0)  # render raw content
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
        rotations.append(rot)
    doc.close()
    return images, rotations


def _collect_training_data(reference_path, max_pages_per_pdf=None, dpi=100):
    """
    Collect raw visual content (rendered WITHOUT rotation) from PDF(s).
    Returns (images, rotations) where:
      - images: list of PIL Images (raw visual content)
      - rotations: list of int rotation attributes (0, 90, 180, 270)
    The rotation attribute IS the ground truth label for each raw image.
    """
    images = []
    rotations = []
    if os.path.isdir(reference_path):
        pdf_files = sorted([
            os.path.join(reference_path, f)
            for f in os.listdir(reference_path)
            if f.lower().endswith('.pdf')
        ])
        print(f"Found {len(pdf_files)} PDF files in training directory.")
        for i, pdf_path in enumerate(pdf_files):
            try:
                pages, rots = _render_pdf_pages_raw(
                    pdf_path, max_pages=max_pages_per_pdf, dpi=dpi)
                images.extend(pages)
                rotations.extend(rots)
                if (i + 1) % 50 == 0 or i == 0 or i == len(pdf_files) - 1:
                    print(f"  Rendered {i+1}/{len(pdf_files)} files, "
                          f"{len(images)} pages total")
            except Exception as e:
                print(f"  Warning: Could not process {pdf_path}: {e}")
    elif os.path.isfile(reference_path):
        pages, rots = _render_pdf_pages_raw(
            reference_path, max_pages=max_pages_per_pdf, dpi=dpi)
        images.extend(pages)
        rotations.extend(rots)
        print(f"Rendered {len(images)} pages from {reference_path}")
    else:
        raise ValueError(f"Reference path not found: {reference_path}")
    return images, rotations


class _OrientationDataset(torch.utils.data.Dataset):
    """Memory-efficient dataset that generates training samples on-the-fly.

    Each page has a base CW rotation (from its PDF rotation attribute).
    We further rotate it with PIL to create augmented samples.
    The effective CW rotation = (base_rot - pil_angle_ccw) % 360.
    """

    def __init__(self, images, base_rotations, indices, transform_train):
        """
        images: list of PIL Images (raw visual content, rendered without rotation)
        base_rotations: list of int (PDF rotation attribute for each page)
        indices: list of (page_idx, pil_angle, flip) tuples
        transform_train: augmentation transforms
        """
        self.images = images
        self.base_rotations = base_rotations
        self.indices = indices
        self.transform_train = transform_train

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        page_idx, pil_angle, flip = self.indices[idx]
        base_rot = self.base_rotations[page_idx]

        img = self.images[page_idx]
        if pil_angle != 0:
            img = img.rotate(pil_angle, expand=True)

        # Effective CW rotation = (base_rot - pil_CCW_angle) % 360
        effective_cw = (base_rot - pil_angle) % 360
        label = ANGLE_TO_LABEL[effective_cw]

        if img.mode != 'RGB':
            img = img.convert('RGB')
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        tensor = self.transform_train(img)
        return tensor, label


def train_model(reference_path, epochs=50, lr=1e-3, batch_size=32,
                max_pages_per_pdf=None, dpi=100):
    """
    Fine-tune ResNet18 for orientation detection using reference PDF(s).

    Training pages are rendered WITHOUT applying the PDF rotation attribute
    (matching the detection pipeline). The rotation attribute serves as the
    ground truth label. Additional samples are created by rotating each raw
    page with PIL, adjusting labels accordingly.
    """
    print(f"Training model using reference: {reference_path}")

    # Render all pages WITHOUT rotation (same as detection pipeline)
    images, rotations = _collect_training_data(
        reference_path, max_pages_per_pdf=max_pages_per_pdf, dpi=dpi)
    print(f"  Total reference pages: {len(images)}")

    # Report rotation distribution
    rot_dist = {}
    for r in rotations:
        rot_dist[r] = rot_dist.get(r, 0) + 1
    print(f"  Rotation distribution: {rot_dist}")

    if len(images) == 0:
        print("Error: No training images found.", file=sys.stderr)
        sys.exit(1)

    num_pages = len(images)
    # 4 PIL rotations x 2 augmented (flip)
    num_samples = num_pages * 4 * 2
    print(f"  Training samples: {num_samples} "
          f"(from {num_pages} pages x 4 rotations x 2 augmented)")

    # Build index list: (page_idx, pil_angle_ccw, flip)
    all_indices = []
    for page_idx in range(num_pages):
        for pil_angle in [0, 90, 180, 270]:
            all_indices.append((page_idx, pil_angle, False))
    for page_idx in range(num_pages):
        for pil_angle in [0, 90, 180, 270]:
            all_indices.append((page_idx, pil_angle, True))

    # Transform with augmentation
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Split into train/val
    import random
    idx_shuffled = list(range(len(all_indices)))
    random.shuffle(idx_shuffled)
    n_val = max(1, len(all_indices) // 10)
    train_idx = [all_indices[i] for i in idx_shuffled[n_val:]]
    val_idx = [all_indices[i] for i in idx_shuffled[:n_val]]

    train_dataset = _OrientationDataset(images, rotations, train_idx,
                                         transform_train)
    val_dataset = _OrientationDataset(images, rotations, val_idx,
                                       transform_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0)

    # Setup model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    model = OrientationResNet18()

    # Fine-tune layer4 + FC
    if num_samples >= 500:
        print("  Fine-tuning layer4 + FC")
        for name, param in model.named_parameters():
            if 'layer4' in name or 'fc' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params_to_optimize = [
            {'params': model.fc.parameters(), 'lr': lr},
            {'params': [p for n, p in model.named_parameters()
                        if 'layer4' in n], 'lr': lr * 0.1},
        ]
    else:
        print("  Small dataset: training FC layer only")
        for param in model.features.parameters():
            param.requires_grad = False
        params_to_optimize = model.fc.parameters()

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    best_val_acc = 0.0
    best_state = None

    model.train()
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == batch_y).sum().item()
            total += len(batch_y)

        train_acc = correct / total * 100

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                output = model(batch_x)
                pred = output.argmax(dim=1)
                val_correct += (pred == batch_y).sum().item()
                val_total += len(batch_y)

        val_acc = val_correct / val_total * 100 if val_total > 0 else 0

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"loss={total_loss:.4f}, "
                  f"train_acc={train_acc:.1f}%, "
                  f"val_acc={val_acc:.1f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model (val_acc={best_val_acc:.1f}%)")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Detect orientation of pages in a PDF or image file.'
    )
    parser.add_argument('input_path', nargs='?', default=None,
                        help='Path to PDF or image file to analyze')
    parser.add_argument('--train', metavar='REFERENCE', default=None,
                        help='Train the model using a reference (upright) PDF '
                             'or a directory of PDFs')
    parser.add_argument('--output', '-o', default=None,
                        help='Output JSON file path')
    parser.add_argument('--heuristic', action='store_true',
                        help='Force heuristic mode (no model)')

    args = parser.parse_args()

    # Training mode
    if args.train:
        train_model(args.train)
        if args.input_path is None:
            return

    if args.input_path is None:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.input_path):
        print(f"Error: File not found: {args.input_path}", file=sys.stderr)
        sys.exit(1)

    # Detection mode
    detector = OrientationDetector(use_model=not args.heuristic)

    ext = os.path.splitext(args.input_path)[1].lower()
    if ext == '.pdf':
        results = detector.detect_pdf_orientations(args.input_path)
    elif ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']:
        print(f"Processing image: {args.input_path}")
        results = detector.detect_image_orientation(args.input_path)
        for page, angle in results.items():
            print(f"  Page {page}: {angle}\u00b0")
    else:
        print(f"Error: Unsupported file format: {ext}", file=sys.stderr)
        sys.exit(1)

    # Output
    output_data = {str(k): v for k, v in results.items()}
    output_json = json.dumps(output_data, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_json)
        print(f"\nResults saved to: {args.output}")
    else:
        print(f"\nResults:")
        print(output_json)


if __name__ == '__main__':
    main()
