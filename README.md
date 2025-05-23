# DANCAST

**DANCAST (Deep Autoencoder Network for Casting Surface Anomaly Detection using Unsupervised Learning)** is a deep learning pipeline built to detect surface defects in casting components. It leverages a convolutional autoencoder trained exclusively on non-defective images to identify anomalies by computing reconstruction errors (MAE) and uses KMeans clustering for classification.

---

## 📌 Features

- Grayscale image preprocessing for lightweight computation
- 12-layer convolutional autoencoder architecture
- Trained solely on non-defective samples (unsupervised anomaly detection)
- Reconstruction-based defect identification
- KMeans clustering for binary classification (defective vs non-defective)
- Interactive single-image prediction
- Visual comparison of original and reconstructed images
- Confusion matrix and classification report for evaluation

---

## 🧠 Methodology

1. **Training**: Autoencoder is trained on non-defective casting images to learn normal features.
2. **Inference**: Defective and non-defective images are reconstructed.
3. **Anomaly Scoring**: Mean Absolute Error (MAE) is computed between input and reconstruction.
4. **Clustering**: KMeans clustering is applied to MAE scores to separate defective and non-defective images.
5. **Prediction**: New image MAE is classified based on trained KMeans.

---

## 🗃️ Dataset Structure

The model expects the dataset in the following structure:
```text
casting_data/
├── train/
│ └── ok_front/ # Non-defective training images
├── test/
│ ├── ok_front/ # Non-defective test images
│ └── def_front/ # Defective test images
```

---

## ⚙️ Requirements

```yaml
- Python 3.7+
- TensorFlow
- NumPy
- scikit-learn
- matplotlib
- scikit-image
- Pillow
```

---

### Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Output

- Training and validation loss plots
- Classification report
- Confusion matrix
- Visual reconstruction of defective vs non-defective samples

---
