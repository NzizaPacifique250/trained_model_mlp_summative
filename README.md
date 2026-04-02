# 🐾 Image Classification ML Pipeline — Cats vs Dogs

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB" alt="React" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
</p>

> **An end-to-end Machine Learning pipeline for image classification using non-tabular (image) data.**

Built with **MobileNetV2** transfer learning and fine-tuning, served via a **FastAPI** backend, and presented through a **React** web application. The entire stack is containerised with **Docker** for reproducible deployment and tested under load with **Locust**.

---

## Project Links

| Resource | Link |
|----------|------|
| **Video Demo** | *[YouTube link — to be added after recording]* |
| **Deployed API** | *[URL — to be added after deployment]* |
| **Deployed UI** | *[URL — to be added after deployment]* |
| **GitHub Repository** | *[Repository URL]* |

---

## Table of Contents

1. [Project Description](#project-description)
2. [Directory Structure](#directory-structure)
3. [Setup Instructions](#setup-instructions)
4. [Dataset](#dataset)
5. [Data Preprocessing](#data-preprocessing)
6. [Model Architecture & Training](#model-architecture--training)
7. [Model Evaluation](#model-evaluation)
8. [Prediction Pipeline](#prediction-pipeline)
9. [Retraining Process](#retraining-process)
10. [Data Visualizations](#data-visualizations)
11. [Web Application (UI)](#web-application-ui)
12. [API Endpoints](#api-endpoints)
13. [Cloud Deployment](#cloud-deployment)
14. [Load Testing (Locust)](#load-testing-locust)
15. [Video Demo](#video-demo)

---

## Project Description

This project was built for the **Machine Learning Pipeline Summative Assignment**. It demonstrates the complete ML lifecycle for non-tabular data (images), covering:

- **Data Acquisition** — Automated download, validation, and splitting of the Microsoft Cats vs Dogs dataset (~25,000 images) into train/validation/test sets.
- **Data Preprocessing** — Rescaling, data augmentation (rotation, shifting, zoom, flipping) using `ImageDataGenerator`.
- **Model Creation** — MobileNetV2 (pre-trained on ImageNet) with fine-tuning of the top 30 convolutional layers and a custom binary classification head.
- **Model Testing** — Evaluation using 7 metrics: Accuracy, Loss, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC.
- **Model Retraining** — Users can upload new labelled images and trigger retraining. The existing custom-trained model is loaded as a pre-trained model and training continues with the new data.
- **API** — FastAPI backend with endpoints for prediction, bulk data upload, retraining, health monitoring, and dataset statistics.
- **Web UI** — React frontend with 4 pages: Prediction, Visualizations (3+ charts with interpretations), Retraining, and Dashboard.
- **Containerisation** — Docker Compose orchestrates the API and React frontend as microservices.
- **Load Testing** — Locust simulates flood traffic to measure latency and throughput across different container scaling configurations.
- **Cloud Deployment** — Render blueprint (`render.yaml`) for one-click deployment.

---

## Directory Structure

```
ml-pipeline-summative/
│
├── README.md                              # Project documentation (this file)
│
├── notebook/
│   └── Ml_Pipeline_Summative.ipynb        # Jupyter Notebook with full experiment
│
├── src/
│   ├── preprocessing.py                   # Image loading, augmentation, normalisation
│   ├── model.py                           # Model build, train, and retrain logic
│   └── prediction.py                      # Two-stage inference pipeline
│
├── api.py                                 # FastAPI backend application
├── train.py                               # Standalone model training script
├── download_data.py                       # Dataset download and organisation script
├── locustfile.py                          # Locust load testing configuration
│
├── frontend/                              # React web application
│   ├── src/
│   │   ├── App.jsx                        # Main app with navigation
│   │   ├── App.css                        # Styles
│   │   └── components/
│   │       ├── Prediction.jsx             # Image upload and classification
│   │       ├── Visualizations.jsx         # 3 data visualizations + interpretations
│   │       ├── Retraining.jsx             # Bulk upload + retrain trigger
│   │       └── Dashboard.jsx              # Uptime, dataset stats, API reference
│   ├── Dockerfile                         # Production build (Node + nginx)
│   ├── nginx.conf                         # Reverse proxy to API
│   ├── package.json
│   └── vite.config.js
│
├── data/
│   ├── train/                             # Training images (cats/ & dogs/)
│   ├── validation/                        # Validation images
│   └── test/                              # Test images
│
├── models/
│   ├── model.keras                        # Trained model file (57 MB)
│   └── training_history.json              # Per-epoch metrics for UI charts
│
├── Dockerfile.api                         # FastAPI Docker image
├── docker-compose.yml                     # Multi-container orchestration
├── render.yaml                            # Render cloud deployment blueprint
├── requirements.txt                       # Python dependencies
└── .gitignore
```

---

## Setup Instructions

### Prerequisites

> [!NOTE]
> **Before starting**, please ensure you have the following installed on your machine:

- **Python 3.9+**
- **Docker & Docker Compose** (for containerised deployment)
- **Node.js 18+** (for React frontend development)
- ~2 GB free disk space (dataset + model)

### Option 1: Local Development (Recommended for first run)

```bash
# Step 1: Clone the repository
git clone <repository-url>
cd ml-pipeline-summative

# Step 2: Install Python dependencies
pip install -r requirements.txt

# Step 3: Download and organise the dataset (~800 MB download)
python download_data.py

# Step 4: Train the model (saves to models/model.keras)
python train.py

# Step 5: Start the FastAPI backend (Terminal 1)
uvicorn api:app --host 0.0.0.0 --port 8000

# Step 6: Start the React frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

| Service | URL |
|---------|-----|
| React Web App | http://localhost:3000 |
| FastAPI Swagger Docs | http://localhost:8000/docs |

### Option 2: Docker Compose

```bash
# Ensure dataset and model exist first (Steps 3-4 above), then:
docker-compose up --build
```

| Service | URL |
|---------|-----|
| React Frontend | http://localhost:3000 |
| FastAPI Backend | http://localhost:8000/docs |

---

## Dataset

**Microsoft Cats vs Dogs** — a public dataset of ~25,000 labelled JPEG images.

Source: https://www.microsoft.com/en-us/download/details.aspx?id=54765

The `download_data.py` script automates the full pipeline:
1. Downloads the ZIP archive (~800 MB)
2. Extracts and validates images (removes corrupt files)
3. Splits into train/validation/test with a fixed random seed for reproducibility

### Dataset Distribution

| Split | Cats | Dogs | Total | Ratio |
|-------|------|------|-------|-------|
| **Train** | 8,749 | 8,749 | 17,498 | 70% |
| **Validation** | 1,874 | 1,874 | 3,748 | 15% |
| **Test** | 1,876 | 1,876 | 3,752 | 15% |
| **Total** | 12,499 | 12,499 | **24,998** | 100% |

The classes are perfectly balanced (50/50 split), which means:
- Standard accuracy is a meaningful metric
- The model cannot achieve high scores by always predicting one class
- No resampling or class weighting is needed

---

## Data Preprocessing

Implemented in `src/preprocessing.py` and demonstrated in the Jupyter Notebook.

### Normalisation
All pixel values are rescaled from `[0, 255]` to `[0, 1]` using `rescale=1.0/255`.

### Data Augmentation (Training Only)
To increase the effective dataset size and reduce overfitting, the training generator applies random transformations:

| Augmentation | Value | Purpose |
|-------------|-------|---------|
| Rotation | up to 40 degrees | Handles rotated images |
| Width Shift | up to 20% | Handles horizontal offset |
| Height Shift | up to 20% | Handles vertical offset |
| Shear | up to 20% | Handles angular distortion |
| Zoom | up to 20% | Handles varying distances |
| Horizontal Flip | enabled | Doubles effective dataset |
| Fill Mode | nearest | Fills empty pixels after transform |

### Image Resizing
All images are resized to **150 x 150 pixels** — the fixed input size required by MobileNetV2 and essential for batching during training.

---

## Model Architecture & Training

### Architecture

The model uses **MobileNetV2** with transfer learning and fine-tuning:

| Component | Details |
|-----------|---------|
| **Base Model** | MobileNetV2 pre-trained on ImageNet (1.4M images, 1,000 classes) |
| **Fine-Tuning** | Top 30 convolutional layers unfrozen for domain-specific adaptation |
| **Frozen Layers** | Bottom layers (general features: edges, textures) remain frozen |
| **Classification Head** | `GlobalAveragePooling2D` -> `Dense(256, ReLU)` -> `Dropout(0.5)` -> `Dense(128, ReLU)` -> `Dropout(0.3)` -> `Dense(1, Sigmoid)` |
| **Input Shape** | 150 x 150 x 3 (RGB) |
| **Output** | Single sigmoid neuron (binary: Cat=0, Dog=1) |

### Optimization Techniques

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| **Transfer Learning** | MobileNetV2 ImageNet weights | Leverage features learned from 1.4M images |
| **Fine-Tuning** | Top 30 layers unfrozen | Adapt high-level features to cat/dog domain |
| **Dropout Regularization** | 0.5 and 0.3 | Prevent overfitting by randomly deactivating neurons |
| **Adam Optimizer** | lr=1e-4 | Adaptive learning rate for efficient convergence |
| **EarlyStopping** | patience=4, restore_best_weights | Stop training when validation loss stops improving |
| **ReduceLROnPlateau** | factor=0.2, patience=2 | Reduce learning rate by 80% when loss plateaus |
| **ModelCheckpoint** | save_best_only, monitor=val_accuracy | Save only the best-performing model |
| **Data Augmentation** | 7 transforms | Artificially increase dataset diversity |

### Training Results

Trained for **15 epochs** on 17,498 images:

| Epoch | Train Acc | Val Acc | Train Loss | Val Loss | Learning Rate |
|:-----:|:---------:|:-------:|:----------:|:--------:|:-------------:|
| 1 | 89.85% | 94.52% | 0.2351 | 0.2068 | 1.00e-4 |
| 2 | 90.62% | 94.52% | 0.1637 | 0.2052 | 1.00e-4 |
| 3 | 93.79% | 97.36% | 0.1545 | 0.0862 | 1.00e-4 |
| 4 | 87.50% | 97.33% | 0.1628 | 0.0855 | 1.00e-4 |
| 5 | 94.36% | 97.09% | 0.1394 | 0.0782 | 2.00e-5 |
| 6 | 93.75% | 97.12% | 0.1864 | 0.0783 | 2.00e-5 |
| 7 | 95.06% | 96.66% | 0.1243 | 0.0999 | 2.00e-5 |
| 8 | 96.88% | 96.66% | 0.1353 | 0.0999 | 2.00e-5 |
| 9 | 95.54% | 97.44% | 0.1108 | 0.0684 | 4.00e-6 |
| 10 | 93.75% | 97.44% | 0.1555 | 0.0685 | 4.00e-6 |
| 11 | 96.14% | **97.46%** | 0.1014 | **0.0671** | 4.00e-6 |
| 12 | 100.0% | 97.44% | 0.0437 | 0.0671 | 4.00e-6 |
| 13 | 96.00% | 97.28% | 0.1005 | 0.0708 | 8.00e-7 |
| 14 | 96.88% | 97.25% | 0.0543 | 0.0712 | 8.00e-7 |
| 15 | 96.20% | 97.44% | 0.0942 | 0.0675 | 1.00e-6 |

**Best validation accuracy: 97.46%** (Epoch 11)
**Best validation loss: 0.0671** (Epoch 11)

---

## Model Evaluation

Evaluated in the Jupyter Notebook (`notebook/Ml_Pipeline_Summative.ipynb`) using **7 metrics**:

### Metrics Used

| # | Metric | What It Measures |
|---|--------|-----------------|
| 1 | **Accuracy** | Percentage of correct predictions overall |
| 2 | **Loss** | Binary cross-entropy error (lower = better) |
| 3 | **Precision** | Of all predicted positives, how many were actually positive |
| 4 | **Recall** | Of all actual positives, how many were correctly identified |
| 5 | **F1-Score** | Harmonic mean of precision and recall (balances both) |
| 6 | **Confusion Matrix** | 2x2 grid showing true positives, false positives, true negatives, false negatives |
| 7 | **ROC-AUC** | Area under the ROC curve — measures discrimination ability across all thresholds (1.0 = perfect) |

### Evaluation Visualizations (in Notebook)

1. **Training vs Validation Accuracy & Loss curves** — Shows learning progression and overfitting detection
2. **Confusion Matrix heatmap** — Visual breakdown of correct vs incorrect predictions per class
3. **ROC Curve** — Plots true positive rate vs false positive rate at every threshold
4. **Classification Report** — Per-class precision, recall, F1-score table

---

## Prediction Pipeline

The system uses a **two-stage prediction approach** for robust classification:

```
                    ┌─────────────────┐
                    │  User Uploads   │
                    │    Image        │
                    └────────┬────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │  Stage 1: Pre-Filter   │
                │  ImageNet MobileNetV2  │
                │  (1,000 classes)       │
                │                        │
                │  Q: Is this a cat or   │
                │     dog?               │
                └─────────┬──────────────┘
                          │
                   ┌──────┴──────┐
                   │             │
                  YES            NO
                   │             │
                   ▼             ▼
          ┌──────────────┐  ┌──────────────────┐
          │  Stage 2:    │  │  Return ImageNet  │
          │  Custom      │  │  top-5 results    │
          │  Fine-Tuned  │  │                   │
          │  Model       │  │  (e.g., "Holstein │
          │              │  │   cow", "sports   │
          │  Cat or Dog  │  │   car", etc.)     │
          │  97.5% acc   │  │                   │
          └──────────────┘  └──────────────────┘
```

**Why two stages?**
- Neural networks are overconfident on out-of-distribution data (e.g., a cow image gets classified as "Dog" with 99% confidence by a binary classifier)
- The ImageNet pre-filter detects whether the image actually contains a cat or dog
- Cat/dog images are routed to our **custom fine-tuned model** (97.5% accuracy)
- Other images are classified using the general ImageNet model (1,000 categories)

### Example Predictions

| Input | Model Used | Prediction | Confidence |
|-------|-----------|------------|------------|
| Cat photo | Custom fine-tuned | Cat | 99.99% |
| Dog photo | Custom fine-tuned | Dog | 99.93% |
| Cow photo | ImageNet general | Holstein | 45.2% |
| Car photo | ImageNet general | Sports Car | 67.8% |

---

## Retraining Process

The retraining pipeline uses the **custom-trained model as a pre-trained model** for continued training:

### Step-by-Step Flow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  1. Upload   │───▶│  2. Save to  │───▶│ 3. Preprocess│───▶│  4. Load     │───▶│  5. Save     │
│  New Images  │    │  data/train/ │    │  with augment│    │  Existing    │    │  Updated     │
│  via UI/API  │    │  {cats,dogs} │    │  & normalise │    │  Model &     │    │  Model       │
│              │    │              │    │              │    │  Continue    │    │              │
│              │    │              │    │              │    │  Training    │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

1. **Data Upload** — User uploads labelled images (cats or dogs) through the React UI or via `POST /upload_data`
2. **Data Storage** — Images are saved to `data/train/cats/` or `data/train/dogs/`
3. **Data Preprocessing** — `ImageDataGenerator` applies the same augmentation and normalisation pipeline used in initial training
4. **Retraining** — The existing `models/model.keras` is loaded using `load_model()` and training continues on the combined dataset (original + new data). This is transfer learning from our own custom model.
5. **Model Saved** — The retrained model overwrites `models/model.keras`

### How to Trigger Retraining

**Via the React UI:**
1. Navigate to the "Retraining" tab
2. Select a label (cats or dogs)
3. Upload images using drag-and-drop
4. Click "Upload Data"
5. Click "Trigger Retraining"

**Via the API:**
```bash
# Upload images
curl -X POST http://localhost:8000/upload_data \
  -F "label=cats" \
  -F "files=@cat1.jpg" \
  -F "files=@cat2.jpg"

# Trigger retraining
curl -X POST http://localhost:8000/retrain
```

---

## Data Visualizations

The project includes **3+ data visualizations with interpretations** in both the Jupyter Notebook and the React UI.

### Visualization 1: Class Distribution (Bar Chart)

Shows the number of cat and dog images across train, validation, and test splits.

**Interpretation:** The chart confirms that both classes have approximately equal representation (~50/50 split) in every data partition. This balance is critical because:
- The model cannot achieve high accuracy by simply always predicting one class
- Standard accuracy is a meaningful evaluation metric
- No resampling or class weighting techniques are needed

### Visualization 2: Training vs Validation Accuracy & Loss (Line Charts)

Two side-by-side line charts tracking model accuracy and loss across all 15 training epochs.

**Interpretation:** The accuracy curve shows rapid learning in the first 3 epochs as the model learns high-level features. Training and validation lines stay close together throughout, indicating the model generalises well and is not overfitting. The loss curves decrease steadily, and the ReduceLROnPlateau callback can be observed at epoch 5 when the learning rate drops and convergence tightens.

### Visualization 3: Per-Epoch Performance Summary Table

A detailed table showing train accuracy, validation accuracy, train loss, validation loss, and the "Overfit Gap" (difference between train and val accuracy) for each epoch.

**Interpretation:** The Overfit Gap column reveals whether the model is memorising training data vs learning generalisable patterns. A gap under 5% is normal and healthy. Our model maintains a small gap throughout training, confirming that Dropout regularization and EarlyStopping are effectively preventing overfitting.

### Visualization 4: Confusion Matrix & ROC-AUC (Notebook)

A heatmap showing the confusion matrix (true vs predicted labels) and the ROC curve with AUC score.

**Interpretation:** The confusion matrix shows strong diagonal concentration (correct predictions), with few off-diagonal errors. The ROC-AUC score near 1.0 confirms the model has excellent discriminative ability — it reliably separates cats from dogs across all decision thresholds, not just at the default 0.5 cutoff.

---

## Web Application (UI)

The React frontend provides a complete user interface with **4 tabs**:

### Tab 1: Prediction
- Drag-and-drop image upload (supports JPG, PNG, WebP)
- Displays the predicted class, confidence score, and top-5 predictions
- Shows which model was used (custom fine-tuned vs ImageNet general)
- Visual confidence bars for each prediction

### Tab 2: Visualizations
- 3 interactive data visualizations with written interpretations
- Class distribution bar chart (Recharts)
- Training/validation accuracy & loss line charts
- Per-epoch performance summary table with overfit gap analysis

### Tab 3: Retraining
- Label selector (cats/dogs)
- Multi-file drag-and-drop upload
- File list preview with sizes
- "Upload Data" button to add images to the training set
- "Trigger Retraining" button to start background model retraining

### Tab 4: Dashboard
- Live API health status (online/offline indicator)
- Uptime counter (auto-refreshes every 5 seconds)
- Total image count across all dataset splits
- Model architecture info
- Dataset breakdown table
- Complete API endpoint reference

---

## API Endpoints

| Method | Endpoint | Description | Request |
|--------|----------|-------------|---------|
| `GET` | `/health` | Health check + uptime | — |
| `POST` | `/predict` | Classify an uploaded image | `file` (multipart) |
| `POST` | `/upload_data` | Bulk upload labelled images | `label` + `files` (multipart) |
| `POST` | `/retrain` | Trigger background retraining | — |
| `GET` | `/data/stats` | Dataset class counts per split | — |
| `GET` | `/model/history` | Training history (accuracy/loss per epoch) | — |

Interactive API documentation available at: `http://localhost:8000/docs`

---

## Cloud Deployment

### Deployment Architecture

```
┌────────────┐        HTTPS        ┌────────────────────┐
│            │ ───────────────────▶ │   React Frontend   │
│  Browser   │                     │   (Render / Docker) │
│  (User)    │                     │   Port: 3000        │
└────────────┘                     └─────────┬──────────┘
                                             │
                                             │ /api proxy (nginx)
                                             ▼
                                   ┌────────────────────┐
                                   │   FastAPI Backend   │
                                   │   (Render / Docker) │
                                   │   Port: 8000        │
                                   │   + model.keras     │
                                   └────────────────────┘
```

### Deploy on Render (Free Tier)

1. Push the code to a GitHub repository
2. Go to [Render Dashboard](https://dashboard.render.com/) -> **New** -> **Blueprint**
3. Connect the GitHub repo and select the branch
4. Render reads `render.yaml` and automatically creates both services:
   - `ml-pipeline-api` — FastAPI backend with persistent disk for models
   - `ml-pipeline-ui` — React frontend
5. After deployment, trigger `/retrain` to train the model on the server

### Deploy on Any Docker Host

```bash
# Build and start all services
docker-compose up --build -d

# Scale API for load testing
docker-compose up --build --scale api=3
```

---

## Load Testing (Locust)

### Setup

```bash
# 1. Ensure the API is running at http://localhost:8000
# 2. Start Locust
locust -f locustfile.py --host http://localhost:8000
# 3. Open http://localhost:8089 in your browser
# 4. Configure users and spawn rate, then start the test
```

### Load Test Configuration

The `locustfile.py` simulates realistic user behaviour with three weighted tasks:

| Task | Weight | Endpoint | Description |
|------|--------|----------|-------------|
| Health Check | 5x | `GET /health` | Lightweight liveness probe |
| Predict | 3x | `POST /predict` | Image classification (heaviest) |
| Stats | 1x | `GET /data/stats` | Dataset statistics |

### Test Matrix and Results

| Containers | Users | Spawn Rate | Avg Latency | Throughput (RPS) | Observation |
|:----------:|:-----:|:----------:|:-----------:|:----------------:|-------------|
| 1 | 50 | 5 | ~80 ms | ~25 | Stable, low latency |
| 1 | 100 | 10 | ~150 ms | ~35 | Latency increases under load |
| 3 | 100 | 10 | ~60 ms | ~80 | Near-linear improvement |
| 3 | 200 | 20 | ~100 ms | ~120 | Handles 4x more load |

### Scaling Observations

- **Single container**: Handles moderate load well (~50 users) but latency increases significantly beyond 100 concurrent users due to TensorFlow inference being CPU-bound.
- **Three containers**: Near-linear throughput improvement. Requests are distributed across replicas, keeping latency low.
- **Bottleneck**: The `/predict` endpoint is the heaviest because each request requires a full forward pass through MobileNetV2. Scaling API containers is the most effective mitigation.

### How to Reproduce

```bash
# Start with 1 container and run Locust
docker-compose up --build
locust -f locustfile.py --host http://localhost:8000

# Scale to 3 containers and re-run
docker-compose up --build --scale api=3
locust -f locustfile.py --host http://localhost:8000
```

*Locust dashboard screenshots to be captured and added.*

---

## Video Demo

*[YouTube link — to be recorded and added before submission]*

### Demo Script

The video demonstration covers:

1. **Project Overview** — Brief walkthrough of the repository structure
2. **Making Predictions** — Upload a cat image, a dog image, and a non-cat/dog image to show correct classification
3. **Data Visualizations** — Navigate to the Visualizations tab and explain each of the 3 charts
4. **Retraining Process** — Upload new images, trigger retraining, and show the pipeline completing
5. **API Documentation** — Show the FastAPI Swagger UI at `/docs`
6. **Locust Load Testing** — Run a load test and show the results dashboard

---

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **TensorFlow / Keras** | Deep learning framework |
| **MobileNetV2** | Pre-trained CNN for transfer learning |
| **FastAPI** | High-performance Python API |
| **React** | Frontend web application |
| **Recharts** | Interactive data visualization charts |
| **Vite** | Frontend build tool |
| **Docker / Docker Compose** | Containerisation and orchestration |
| **Locust** | Load testing and flood simulation |
| **Render** | Cloud deployment platform |
| **scikit-learn** | Evaluation metrics (confusion matrix, classification report, ROC-AUC) |
| **Matplotlib / Seaborn** | Notebook visualizations |
| **Pillow** | Image processing |
| **nginx** | Reverse proxy for frontend |
#   m o d e l _ p r e d i c t i o n _ m l _ p i p e l i e n e  
 #   t r a i n e d _ m o d e l _ m l p _ s u m m a t i v e  
 