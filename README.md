<div align="center">

<!-- Animated Header -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1b2a,50:1a237e,100:4a148c&height=200&section=header&text=Brain%20Tumor%20MRI%20Classifier&fontSize=38&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=End-to-End%20Deep%20Learning%20%2B%20MLOps%20Pipeline&descAlignY=55&descSize=18" width="100%"/>

<!-- Typing Animation -->
<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=7C4DFF&center=true&vCenter=true&width=750&lines=VGG16+Transfer+Learning+%F0%9F%A7%A0;4-Class+Brain+MRI+Classification+%F0%9F%A7%AC;DVC+Pipeline+%2B+Experiment+Tracking+%F0%9F%94%AC;Docker+%2B+GitHub+Actions+CI%2FCD+%F0%9F%9A%80;AWS+ECR+%2B+EC2+Cloud+Deployment+%E2%98%81%EF%B8%8F;Flask+REST+API+Serving+%F0%9F%8C%90;89.2%25+Test+Accuracy+%E2%9C%85" alt="Typing SVG" />
</a>

<br/><br/>

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST_API-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-ECR_+_EC2-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-Pipeline_Versioning-945DD6?style=for-the-badge&logo=dvc&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)

<br/>

![Best Val Accuracy](https://img.shields.io/badge/Best_Val_Accuracy-89.29%25-brightgreen?style=flat-square&logo=checkmarx)
![Test Accuracy](https://img.shields.io/badge/Test_Accuracy-89.17%25-blue?style=flat-square&logo=checkmarx)
![Classes](https://img.shields.io/badge/Classes-4_Tumor_Types-purple?style=flat-square)
![Early Stopping](https://img.shields.io/badge/Early_Stopping-Enabled-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

</div>

---

## 🌟 About The Project

> **An end-to-end MLOps pipeline for automated detection and classification of brain tumors from MRI scans using Deep Learning — from data ingestion to cloud deployment.**

Brain tumors affect millions of people worldwide and early, accurate diagnosis is critical for treatment outcomes. Manual MRI analysis is time-consuming and requires expert radiologists. This project tackles that challenge with a production-grade AI system that classifies brain MRI scans into 4 categories — **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor** — enabling faster, automated screening support.

What makes this project stand out is not just the model, but the **complete MLOps infrastructure** built around it:
- ⚙️ Reproducible DVC-managed pipelines
- 🔁 Automated CI/CD from code push to cloud deployment
- 🐳 Docker containerization for environment consistency
- ☁️ AWS-native deployment with ECR + EC2
- 🌐 REST API for real-time inference
- 🎯 Early stopping + best-checkpoint restoration to prevent overfitting from degrading the deployed model

---

## 📸 Project Highlights

<div align="center">

| 🎯 Problem | 🔬 Approach | 🏆 Result |
|:---:|:---:|:---:|
| Brain tumor classification from MRI | VGG16 Transfer Learning | **89.2% Test Accuracy** |
| Manual MRI reading is slow & costly | 4-class automated classification | Real-time REST API |
| No reproducibility in experiments | DVC pipeline versioning | Fully reproducible runs |
| Manual deployments are error-prone | GitHub Actions CI/CD | Push-to-deploy automation |
| Overfitting past the best epoch | Early stopping + checkpoint restore | Val/test gap shrunk from ~5 pts to ~0.1 pt |

</div>

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  🧠 BRAIN TUMOR CLASSIFIER SYSTEM                │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  📦 DATA LAYER              🧠 MODEL LAYER                       │
│  ┌──────────────┐           ┌─────────────────────────────┐      │
│  │ Data Source  │──────────▶│  VGG16 (ImageNet weights)   │      │
│  │ (HuggingFace)│           │  + Custom Dense Head (x4)   │      │
│  └──────────────┘           │  + Adam Optimizer (lr=0.001)│      │
│         │                   └─────────────────────────────┘      │
│         ▼                              │                          │
│  ┌──────────────┐                     ▼                          │
│  │ DVC Pipeline │           ┌─────────────────────────────┐      │
│  │  Stage 1: Data Ingest    │  Training Pipeline           │      │
│  │  Stage 2: Base Model     │  + TensorBoard Logging       │      │
│  │  Stage 3: Training  │    │  + Best Model Checkpointing  │      │
│  │  Stage 4: Evaluation│    │  + Early Stopping (patience=5)│     │
│  └──────────────┘           │  + Data Augmentation         │      │
│                              └─────────────────────────────┘      │
│                                                                   │
│  🚀 DEPLOYMENT LAYER                                             │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  GitHub Push → CI (Lint/Test) → Docker Build → ECR Push  │    │
│  │  → EC2 Pull → Container Run → Flask API (port 8080) ✅   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 ML Pipeline (DVC)

```mermaid
flowchart LR
    A[📥 Data Ingestion\nDownload & Unzip\nBrain MRI Images] --> B[🏗️ Prepare Base Model\nLoad VGG16\nFreeze Layers\nAdd 4-class Dense Head]
    B --> C[🏋️ Model Training\nAugmentation\nEarly Stopping\nBest Checkpoint\nTensorBoard]
    C --> D[📊 Evaluation\nTest Set 1680 images\nSave scores.json\nDVC Metrics]
    D --> E[🌐 Flask API\nREST Endpoint\nBase64 Input\nJSON Output]

    style A fill:#0d1b2a,color:#fff,stroke:#7C4DFF
    style B fill:#1a237e,color:#fff,stroke:#7C4DFF
    style C fill:#4a148c,color:#fff,stroke:#7C4DFF
    style D fill:#7C4DFF,color:#fff,stroke:#4a148c
    style E fill:#0f9b8e,color:#fff,stroke:#7C4DFF
```

---

## 🚀 CI/CD Pipeline

```mermaid
flowchart TD
    A[👨‍💻 Git Push to Main] --> B[⚙️ GitHub Actions Triggered]
    B --> C[🔍 Continuous Integration\nCode Lint + Unit Tests]
    C --> D[🐳 Continuous Delivery\nDocker Build + Push to ECR]
    D --> E[☁️ Continuous Deployment\nEC2 Self-Hosted Runner]
    E --> F[⬇️ Pull Latest Image from ECR]
    F --> G[🛑 Stop Old Container]
    G --> H[▶️ Run New Container\non Port 8080]
    H --> I[🧹 Docker System Prune]
    I --> J[✅ Live & Serving Users!]

    style A fill:#2b2d42,color:#edf2f4
    style J fill:#2d6a4f,color:#fff
```

---

## 📂 Project Structure

```
Brain-Tumor-MRI-Classifier/
│
├── 📁 .github/
│   └── workflows/
│       └── main.yaml              # CI/CD GitHub Actions pipeline
│
├── 📁 .dvc/                       # DVC configuration
│
├── 📁 artifacts/                  # Auto-generated model artifacts
│   ├── data_ingestion/            # Downloaded MRI dataset (Training/ Testing/)
│   ├── prepare_base_model/        # VGG16 base + updated models (.h5)
│   ├── prepare_callbacks/         # TensorBoard logs + checkpoints
│   └── training/                  # Final trained model (best-epoch weights)
│
├── 📁 config/
│   └── config.yaml                # All path & URL configurations
│
├── 📁 research/                   # Jupyter notebooks (exploration)
│   ├── data_ingestion.ipynb
│   ├── prepare_base_model.ipynb
│   ├── training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── 📁 src/cnnClassifier/
│   ├── components/                # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── prepare_base_model.py  # VGG16 transfer learning
│   │   ├── prepare_callbacks.py   # TensorBoard + checkpointing + early stopping
│   │   ├── training.py            # Train + best-checkpoint model saving
│   │   └── evaluation.py
│   ├── pipeline/                  # Stage orchestrators
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_prepare_base_model.py
│   │   ├── stage_03_training.py
│   │   ├── stage_04_evaluation.py
│   │   └── predict.py             # 4-class inference pipeline
│   └── utils/common.py
│
├── 📁 templates/
│   └── index.html                 # BrainScan AI web UI
├── 📄 app.py                      # Flask REST API server
├── 📄 main.py                     # Full pipeline runner
├── 📄 dvc.yaml                    # DVC stage definitions
├── 📄 params.yaml                 # Hyperparameters
├── 📄 scores.json                 # Model evaluation metrics
├── 📄 Dockerfile                  # Container definition
├── 📄 requirements.txt
└── 📄 pyproject.toml
```

---

## 🧠 Model Architecture

<div align="center">

| Layer | Details |
|:------|:--------|
| **Base** | VGG16 (pre-trained on ImageNet) |
| **Input Shape** | 224 × 224 × 3 |
| **VGG16 Layers** | All frozen (transfer learning) |
| **Custom Head** | Flatten → Dense(4, softmax) |
| **Optimizer** | Adam (lr = 0.001) |
| **Loss** | Categorical Crossentropy |
| **Classes** | Glioma / Meningioma / No Tumor / Pituitary |
| **Augmentation** | Enabled (rotation, flip, zoom, shift) |
| **Batch Size** | 16 |
| **Max Epochs** | 30 (early stopping patience = 5 on val_accuracy) |
| **Checkpointing** | Saves best `val_accuracy` epoch only; best weights restored automatically |
| **Trainable Params** | 100,356 (392 KB) |
| **Total Params** | 14,815,044 (56.51 MB) |

</div>

### 📊 Model Performance

```
┌──────────────────────────────────────────┐
│           TRAINING RESULTS               │
├──────────────────┬───────────────────────┤
│  Best Val Acc    │       89.29%          │  ← Epoch 6 (restored)
│  Final Test Acc  │       89.17%          │  ← Unseen Testing/ (1,680 imgs)
│  Test Loss        │       0.3438          │
│  Training stopped │  Epoch 11 (early stop)│
│  Training Images  │       4,480           │
│  Validation Images│       1,120 (20% split)│
│  Test Images      │       1,680           │
│  Classes          │       4               │
└──────────────────┴───────────────────────┘
```

> **Fix note:** an earlier version of this pipeline saved the *last* epoch's weights instead of the best checkpoint, which silently deployed an overfit model — visible as an ~5-point gap between validation accuracy (89.7%) and test accuracy (84.6%). Adding `EarlyStopping(restore_best_weights=True)` and explicitly reloading the checkpoint file closed that gap to ~0.1 points and lifted test accuracy by +4.5 points.

### 📈 Training Curve (current run)

| Epoch | Train Acc | Val Acc | Val Loss | Note |
|-------|-----------|---------|----------|------|
| 1 | 72.3% | 83.8% | 0.5252 | first checkpoint |
| 2 | 77.7% | 81.7% | 0.4874 | |
| 3 | 82.9% | 83.7% | 0.4733 | |
| 4 | 81.9% | 86.7% | 0.3672 | checkpoint improved |
| 5 | 82.8% | 89.1% | 0.3343 | checkpoint improved |
| **6** | **84.0%** | **89.3% ← best** | **0.3447** | **restored & deployed** |
| 7 | 82.9% | 87.6% | 0.3854 | overfitting begins |
| 8 | 85.7% | 88.9% | 0.3550 | |
| 9 | 85.2% | 80.3% | 0.8410 | val loss spikes |
| 10 | 82.5% | 78.0% | 1.1025 | |
| 11 | 84.3% | 85.5% | 0.4705 | early stopping triggered |

---

## 🏷️ Detectable Conditions

<div align="center">

| Class | Condition | Description |
|:------|:----------|:------------|
| 🔴 **Glioma** | Malignant | Tumor in glial cells — can be aggressive, urgent attention needed |
| 🟠 **Meningioma** | Usually Benign | Arises from meninges — slow-growing but location-dependent severity |
| 🟣 **Pituitary Tumor** | Benign | Forms in pituitary gland — affects hormone production |
| 🟢 **No Tumor** | Healthy | No tumor markers detected in MRI scan |

</div>

---

## ⚡ Quick Start

### Prerequisites

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Git](https://img.shields.io/badge/Git-required-orange?logo=git)
![Docker](https://img.shields.io/badge/Docker-optional-blue?logo=docker)

> **Note on GPU training:** TensorFlow ≥ 2.11 does not support native GPU acceleration on Windows. If training feels slow (CPU-only), use WSL2 with a GPU-enabled TensorFlow build, or the TensorFlow-DirectML plugin, to use an NVIDIA GPU on Windows.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/techakash32/Brain-Tumor-MRI-Classifier.git
cd Brain-Tumor-MRI-Classifier
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4️⃣ Run the Full DVC Pipeline

```bash
dvc repro
```

> Runs all 4 stages: Data Ingestion → Prepare Base Model → Training (with early stopping) → Evaluation

### 5️⃣ Launch the Flask API

```bash
python app.py
```

The BrainScan AI web app will be live at `http://localhost:8080`

---

## 🌐 API Reference

### `GET /`
Returns the BrainScan AI web UI.

### `POST /predict`
Classifies a brain MRI image into one of 4 tumor categories.

**Request Body:**
```json
{
  "image": "<base64-encoded-image-string>"
}
```

**Response:**
```json
[
  {
    "image": "Glioma",
    "class_key": "glioma",
    "confidence": "94.21",
    "info": "Glioma is a tumor that occurs in the brain and spinal cord..."
  }
]
```

### `GET /train` or `POST /train`
Triggers a full model retraining pipeline.

```bash
# Example prediction using curl
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64_string>"}'
```

---

## 🐳 Docker

### Build & Run Locally

```bash
# Build the Docker image
docker build -t brain-tumor-classifier .

# Run the container
docker run -p 8080:8080 brain-tumor-classifier
```

### Pull from AWS ECR

```bash
# Authenticate
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin <ecr-uri>

# Pull and run
docker pull <ecr-uri>/<repo-name>:latest
docker run -d -p 8080:8080 <ecr-uri>/<repo-name>:latest
```

---

## ☁️ AWS Deployment Guide

### Required GitHub Secrets

Go to **Repository → Settings → Secrets & Variables → Actions** and add:

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g., `ap-south-1` |
| `ECR_REPOSITORY_NAME` | Your ECR repo name |

### Self-Hosted Runner (EC2)

1. Launch an EC2 instance (Ubuntu 22.04 recommended)
2. Install Docker on the instance
3. Go to **Repo → Settings → Actions → Runners → New Self-Hosted Runner**
4. Follow the setup script on your EC2 instance
5. Push to `main` branch — deployment happens automatically 🎉

---

## 📈 Experiment Tracking

### DVC Metrics

```bash
# View tracked metrics
dvc metrics show

# Compare across git commits
dvc metrics diff
```

### TensorBoard

```bash
tensorboard --logdir artifacts/prepare_callbacks/tensorboard_log_dir
```

Open `http://localhost:6006` to visualize training curves.

---

## 🔧 Configuration

**`config/config.yaml`** — Paths & URLs:
```yaml
data_ingestion:
  source_URL: https://huggingface.co/datasets/beastboy21718/Brain_MRI_Dataset/resolve/main/brain_tumor.zip
  training_data: Training
```

**`params.yaml`** — Hyperparameters:
```yaml
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
EPOCHS: 30
LEARNING_RATE: 0.001
AUGMENTATION: True
CLASSES: 4
WEIGHTS: imagenet
```

> `EPOCHS` is a ceiling, not a target — `EarlyStopping(patience=5)` halts training automatically once validation accuracy stops improving, and the best-epoch weights are restored and saved.

---

## 🛠️ Tech Stack

<div align="center">

| Category | Technology |
|:---------|:-----------|
| **Deep Learning** | TensorFlow 2.x, Keras, VGG16 |
| **Pipeline & Versioning** | DVC (Data Version Control) |
| **Experiment Tracking** | TensorBoard |
| **API Framework** | Flask + Flask-CORS |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud Registry** | AWS ECR (Elastic Container Registry) |
| **Cloud Compute** | AWS EC2 (Self-Hosted Runner) |
| **Dataset** | Brain MRI Dataset (HuggingFace) |
| **Config Management** | YAML (config.yaml + params.yaml) |
| **Language** | Python 3.12 |

</div>

---

## 🚧 Future Improvements

- [x] ✅ Fix checkpoint bug — save best-`val_accuracy` epoch instead of last epoch
- [x] ✅ Add early stopping to halt training once validation accuracy plateaus
- [ ] 🔢 Unfreeze VGG16 top layers for fine-tuning to push accuracy beyond current 89.2%
- [ ] 📊 Integrate MLflow for richer experiment tracking
- [ ] 🧪 Add real unit tests to CI pipeline
- [ ] 🖼️ Build a Streamlit frontend for easy demo
- [ ] 📱 Export model to TensorFlow Lite for mobile inference
- [ ] 🔔 Add Grad-CAM visualization to highlight tumor regions in MRI
- [ ] 📦 Track per-class precision/recall/F1 (not just aggregate accuracy) — critical for a class like Glioma where false negatives are costly
- [ ] 🔐 Add API authentication, rate limiting, and HTTPS
- [ ] 📦 Add more tumor subtypes for granular classification

---

## 🤝 Contributing

Contributions are welcome! Here's how:

```bash
# Fork the repo, then:
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request 🚀
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">

<!-- Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4a148c,50:1a237e,100:0d1b2a&height=120&section=footer" width="100%"/>

**Built with ❤️ | Deep Learning × MLOps × Cloud**

*Star ⭐ this repo if you found it useful!*

[![GitHub stars](https://img.shields.io/github/stars/techakash32/Brain-Tumor-MRI-Classifier?style=social)](https://github.com/techakash32/Brain-Tumor-MRI-Classifier)

</div>
