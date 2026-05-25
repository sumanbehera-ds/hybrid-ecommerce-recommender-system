# Hybrid E-commerce Recommendation System

A production-style hybrid recommendation system built using collaborative filtering, deep learning, and sequential recommendation techniques for personalized e-commerce product recommendations.

---

# Project Overview

This project implements multiple recommendation approaches:

- Popularity-Based Recommendation
- Item-Based Collaborative Filtering
- Neural Collaborative Filtering (NCF)
- GRU4Rec Sequential Recommendation
- Hybrid Recommendation System

The system was optimized for lightweight cloud deployment using:

- FastAPI
- Streamlit
- Docker
- Render

---

# Tech Stack

## Machine Learning / Deep Learning
- Python
- PyTorch
- Scikit-learn
- Pandas
- NumPy

## Backend
- FastAPI
- Uvicorn

## Frontend
- Streamlit

## Deployment
- Docker
- Render

## Experimentation
- MLflow

---

# Dataset

Dataset used:

RetailRocket E-commerce Dataset

Files:
- events.csv
- item_properties_part1.csv
- item_properties_part2.csv
- category_tree.csv

Main recommendation training was performed using:
- events.csv

because it contains:
- user interactions
- clicks
- views
- add-to-cart events
- transactions

---

# Recommendation Models

## 1. Popularity-Based Recommender

Recommends globally popular products.

### Purpose
- baseline model
- fallback recommendation system

---

## 2. Item-Based Collaborative Filtering

Uses item-item similarity based on user interaction patterns.

### Features
- cosine similarity
- nearest-neighbor style recommendations

---

## 3. Neural Collaborative Filtering (NCF)

Deep learning recommender model for learning user-item interactions.

### Architecture
- embedding layers
- dense neural network
- implicit feedback learning

---

## 4. GRU4Rec Sequential Recommender

Session-based recommendation using GRU networks.

### Features
- sequence-aware recommendation
- next-item prediction
- session learning

---

## 5. Hybrid Recommendation System

Combines:
- GRU4Rec predictions
- popularity fallback

Optimized for lightweight production deployment.

---

# Project Structure

```text
hybrid-ecommerce-recommender-system/
│
├── app.py
├── app_lightweight.py
├── streamlit_app.py
│
├── deploy_models/
│   ├── gru4rec_model.pth
│   └── popularity_baseline.pkl
│
├── src/
│   └── models/
│       ├── train_gru4rec.py
│       ├── train_ncf.py
│       ├── train_item_cf.py
│       ├── hybrid_recommender.py
│       └── lightweight_recommender.py
│
├── notebooks/
├── Dockerfile.api
├── Dockerfile.streamlit
├── requirements.txt
├── requirements-docker.txt
└── README.md
```

---

# Live Deployments

## FastAPI API

Live API:
https://ecommerce-recommender-api-slg9.onrender.com

### Endpoints

#### Health Check
```text
GET /health
```

#### Recommendation Endpoint
```text
POST /recommend
```

Example request:
```json
{
  "item_sequence": [325215, 259884, 216305],
  "top_n": 10
}
```

---

## Streamlit Frontend

Live App:
https://ecommerce-recommender-ui.onrender.com

Features:
- interactive recommendation generation
- API integration
- lightweight frontend deployment

---

# Docker Deployment

## Build API

```bash
docker build -f Dockerfile.api -t ecommerce-recommender-api .
```

## Run API

```bash
docker run -p 8000:8000 ecommerce-recommender-api
```

## Build Streamlit

```bash
docker build -f Dockerfile.streamlit -t ecommerce-recommender-ui .
```

---

# Production Optimization

Initial hybrid deployment was too large due to:
- item similarity matrix
- full hybrid artifacts
- large PyTorch layers

Optimizations performed:
- lightweight GRU4Rec deployment
- popularity fallback strategy
- CPU-only PyTorch
- reduced Docker image size

---

# Key Learning Outcomes

- recommender system architecture
- sequential recommendation systems
- collaborative filtering
- deep learning recommendation
- production deployment
- Docker optimization
- lightweight inference systems
- FastAPI deployment
- Streamlit integration
- Render cloud deployment

---

# Future Improvements

- content-based recommendation
- category-aware recommendation
- vector search retrieval
- ANN similarity search
- Redis caching
- Kubernetes deployment
- real-time streaming recommendations

---

# Author

## Suman Behera

GitHub:
https://github.com/sumanbehera-ds