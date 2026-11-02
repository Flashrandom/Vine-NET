# 🍇 Grape Vine Segmentation using U-Net

<div align="center">

**Semantic segmentation of grape bunches in vineyard imagery using a U-Net convolutional neural network**

*TensorFlow · Keras · Flask*

</div>
<div align="center">

### 🚀 [Try the Live Demo](https://abhayrathorethakur--vinenet.us-east.modal.run)

</div>---

## 📌 Overview

This project applies a **U-Net** architecture to aerial/close-range vineyard images to segment and detect **grape bunches**, generating pixel-level binary masks. Accurate grape-bunch detection enables precision viticulture use-cases such as **yield estimation** and **targeted harvesting**.

The pipeline covers the full lifecycle:

1. **Preprocessing** — loading and resizing high-resolution images (1920 × 960) with their instance masks
2. **Training** — a 5-level U-Net with dropout regularization and data caching via NumPy
3. **Evaluation** — accuracy, precision, and recall metrics logged via TensorBoard
4. **Deployment** — a Flask web app that overlays predicted masks on uploaded images in real time

## 🏗️ Model Architecture

A classic encoder–decoder U-Net implemented in `tf.keras`:

| Component | Details |
|---|---|
| **Input** | 1920 × 960 × 3 RGB image (normalized to `[0, 1]`) |
| **Encoder (contraction)** | 5 blocks — 16 → 32 → 64 → 128 → 256 filters, double `Conv2D` + `ReLU` (`he_normal` init) with dropout (0.1–0.3), `MaxPooling2D` between levels |
| **Decoder (expansion)** | `Conv2DTranspose` upsampling with skip connections to corresponding encoder features |
| **Output** | 1 × 1 `Conv2D` with `sigmoid` activation → binary mask |
| **Loss / Optimizer** | `binary_crossentropy` / Adam |
| **Metrics** | Accuracy, Precision, Recall |
| **Training** | Batch size 16, up to 20 epochs, 10% validation split, `ModelCheckpoint` (best model) + `EarlyStopping` |

## 📂 Project Structure

```
.
├── hack.py                 # Data preprocessing + U-Net definition + training loop
├── eval.ipynb              # Evaluation notebook (metrics & predicted masks)
├── eval-checkpoint.ipynb   # Evaluation checkpoint
├── app.py                  # Flask inference web app
├── model_for_nuclei.keras  # Trained model weights
└── data/
    ├── train/
    │   ├── images/         # Training images
    │   └── masks/          # Ground-truth masks (*_instanceIds.png)
    └── test/
        ├── images/         # Test images
        └── Predicted/      # Predicted masks output
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Dependencies: `numpy`, `scikit-image`, `matplotlib`, `tqdm`, `opencv-python`, `flask`

```bash
pip install tensorflow numpy scikit-image matplotlib tqdm opencv-python flask
```

### Training

1. Place your dataset under `data/train/images` (with matching masks in `data/train/masks`) and test images under `data/test/images`.
2. Set the dataset paths in `hack.py` (`TRAIN_PATH` / `TEST_PATH`).
3. Run the script:

```bash
python hack.py
```

> Preprocessed arrays are cached as `X_train.npy`, `Y_train.npy`, and `X_test.npy` — subsequent runs skip preprocessing automatically. Training logs are written to `logs/` (view with `tensorboard --logdir logs`), and the best model is saved as `model_for_nuclei.keras`.

### Evaluation

Open `eval.ipynb` to run inference on the test set. Accuracy and precision are reported in the training logs, and predicted masks for all test images are saved to `data/test/Predicted/`.

### Web App (Inference)

```bash
python app.py
```

Upload a vineyard image through the web UI — the app returns the image with predicted grape-bunch regions overlaid in red.

## 🤝 Contributors

- [Flashrandom](https://github.com/Flashrandom)

