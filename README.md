# 🧠 GRAYBOX — Explainable AI for Iris Classification

**GRAYBOX** is a FastAPI-based microservice for training, predicting, and explaining a neural network on the Iris dataset.  
It allows users to customize training of an MLPClassifier and returns both predictions and human-readable explanations using SHAP.

---

## 🚀 Features

- Train a multilayer perceptron (MLP) with custom hyperparameters
- Batch prediction for multiple Iris samples
- SHAP-based explanations with feature importance summaries
- Fully async, modular FastAPI backend
- Ready for extension or deployment

---

## 📊 Example Use Case

Use GRAYBOX as a demo of how to turn any ML model into a transparent, explainable API.  
Send a batch of flower measurements → get a class prediction and explanation for each sample.

---

## 📂 Project Structure

├── LICENSE
└── README.md

---

## 🔧 Quickstart

1. Create and activate a virtual environment

```bash
make create_venv
```

This creates a Python virtual environment in `.venv/` and installs `uv`.
You'll see a message telling you how to activate the environment:

```bash
source .venv/bin/activate
```

2. Install project dependencies 

```bash
make requirements
```

Installs all required Python packages using `uv` from the `requirements.txt` file.

---

## 📦 API Overview

Run the API using : 

```bash
uvicorn mlp_iris_api:app
```

The available endpoints once the API is running are :

| Endpoint     | Method | Description                              |
|--------------|--------|------------------------------------------|
| `/train`     | POST   | Train a custom MLP model on the Iris data|
| `/predict`   | POST   | Predict classes for a batch of samples    |
| `/explain`   | POST   | Explain predictions using SHAP values     |
| `/docs`      | GET    | Interactive API docs via Swagger UI       |

🧪 Once running, go to [http://localhost:8000/docs](http://localhost:8000/docs) to explore the API.
