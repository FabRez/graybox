import joblib
import logging
import pandas as pd
import numpy as np
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from contextlib import asynccontextmanager


# -----------------------------
# Config & Setup
# -----------------------------

MODEL_PATH = Path("models/mlp_model.pkl")
EXPLAINER_PATH = Path("models/shap_explainer.pkl")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load dataset
iris = load_iris()

# sanitize column names of the iris dataset
# - remove (cm) from the column names
# - replace spaces with underscores
feature_names = [
    name.replace(" (cm)", "").replace(" ", "_") for name in iris.feature_names
]
df_iris = pd.DataFrame(iris.data, columns=feature_names)

# add a new column with the species names, this is what we are going to try to predict
df_iris["species"] = np.array([iris.target_names[i] for i in iris.target])


# -----------------------------
# Utility Functions
# -----------------------------
def categorize_importance(shap_val: float, max_shap: float) -> str:
    """
    Categorize the importance of a feature based on its SHAP value.
    """
    if max_shap == 0:
        raise ValueError("max_shap cannot be zero.")
    norm_val = abs(shap_val) / max_shap
    if norm_val > 0.7:
        return "High"
    elif norm_val > 0.3:
        return "Medium"
    elif norm_val > 0.001:
        return "Low"
    else:
        return "Null"


# -----------------------------
# Data Models
# -----------------------------
class TrainConfig(BaseModel):
    hidden_layer_sizes: Optional[List[int]] = [32, 16]
    activation: Optional[str] = "relu"
    max_iter: Optional[int] = 30
    test_size: Optional[float] = 0.2
    random_state: Optional[int] = 42
    save_model: Optional[bool] = False  # Flag to save model


class IrisFeaturesBatch(BaseModel):
    sepal_length: List[float]
    sepal_width: List[float]
    petal_length: List[float]
    petal_width: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": [5.1, 5.4, 6.5, 1.5, 2.5],
                "sepal_width": [3.5, 3.0, 3.2, 0.4, 1.2],
                "petal_length": [1.4, 3.5, 5.7, 2.7, 0.8],
                "petal_width": [0.2, 1.58, 2.2, 4.0, 0.3],
            }
        }

        @staticmethod
        def field_validator(v, field):
            """Validate that fields are within reasonable bounds for Iris dataset."""
            if not all(0 <= x for x in v):
                raise ValueError(f"{field.name} values must be positive.")
            return v


# -----------------------------
# Application Startup and Shutdown
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, explainer

    if MODEL_PATH.exists():
        logger.info("Loading pre-trained model...")
        model = joblib.load(MODEL_PATH)

    if EXPLAINER_PATH.exists():
        logger.info("Loading SHAP explainer...")
        explainer = joblib.load(EXPLAINER_PATH)

    yield  # The application runs during this phase

    # Cleanup if necessary
    logger.info("Shutting down, releasing resources.")
    model, explainer = None, None  # Reset to free up memory


app = FastAPI(
    title="MLP Iris Classifier with Explainability",
    version="2.0.0",
    lifespan=lifespan,
)


# -----------------------------
# API Endpoints
# -----------------------------


@app.post("/train")
def train_model(cfg: TrainConfig):
    """
    Train an MLPClassifier on the Iris dataset with configurable hyperparameters.
    """
    global model, explainer

    try:
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            df_iris.drop(columns="species"),
            iris.target,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
        )

        # Initialize and fit model
        model = MLPClassifier(
            hidden_layer_sizes=tuple(cfg.hidden_layer_sizes),
            activation=cfg.activation,
            max_iter=cfg.max_iter,
            random_state=cfg.random_state,
        )
        model.fit(X_train, y_train)

        # Initialize SHAP explainer
        explainer = shap.KernelExplainer(
            model.predict, X_train
        )  # Uses DeepExplainer if possible

        # Save model and explainer if required
        if cfg.save_model:
            joblib.dump(model, MODEL_PATH)
            joblib.dump(explainer, EXPLAINER_PATH)
            logger.info("Model and explainer saved successfully.")

        test_accuracy = model.score(X_test, y_test)
        return {
            "message": "Model trained successfully",
            "test_accuracy": test_accuracy,
        }

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Training failed: {str(e)}"
        )


@app.post("/predict")
def predict(data: IrisFeaturesBatch):
    """
    Return predictions for batch Iris samples.
    """
    global model

    if model is None and MODEL_PATH.exists():
        logger.info("Loading pre-trained model from disk.")
        model = joblib.load(MODEL_PATH)

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Please call /train first.",
        )

    try:
        df_pred = pd.DataFrame(data.model_dump())
        preds = model.predict(df_pred).tolist()
        return {"predictions": [iris.target_names[i] for i in preds]}

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )


@app.post("/explain")
def explain_sample(data: IrisFeaturesBatch):
    """
    Provide a clear and structured explanation for a batch of iris sample.
    The explanation is based on SHAP feature importance values and feature statistics.
    """
    global explainer, model

    if model is None and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)

    if explainer is None and EXPLAINER_PATH.exists():
        explainer = joblib.load(EXPLAINER_PATH)

    if explainer is None or model is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained yet. Please call /train first.",
        )

    try:
        # Convert input features to a DataFrame
        df_samples = pd.DataFrame(data.model_dump())
        pred_classes = model.predict(df_samples)
        pred_class_names = [iris.target_names[pred] for pred in pred_classes]
        pred_probs = [prob.max() for prob in model.predict_proba(df_samples)]

        # calculate min, max, and mean values for each feature
        feature_stats = df_iris.describe().loc[["min", "max", "mean"]].T

        batch_explanations = {}

        # Calculate SHAP values for the entire batch
        shap_values = explainer.shap_values(df_samples)

        for i, (pred_class_name, pred_prob) in enumerate(
            zip(pred_class_names, pred_probs)
        ):
            explanation = [
                f"The model predicted class {pred_class_name.upper()} with {pred_prob:.2%} confidence."
            ]

            # Get most important features for this sample
            sample_shap_values = abs(shap_values[i])
            shap_dict = dict(zip(feature_names, sample_shap_values))
            sorted_shap = sorted(
                shap_dict.items(), key=lambda x: x[1], reverse=True
            )
            max_shap = max(sample_shap_values)
            explanation.append("Feature Importance:")
            for feature, shap_value in sorted_shap:
                # Get the current sample's feature value
                feature_value = df_samples[feature].iloc[i]

                shap_importance = categorize_importance(shap_value, max_shap)

                explanation.append(
                    f" - {feature.replace('_', ' ').title()}: \n"
                    f"   Input Value: {feature_value:.2f} \n"
                    f"   Importance: {shap_importance} \n"
                    f"   Feature stats: mean={feature_stats['mean'][feature]:.2f}, "
                    f"range=[{feature_stats['min'][feature]:.2f} - {feature_stats['max'][feature]:.2f}]"
                )
            batch_explanations[f"Sample {i+1}"] = "\n".join(explanation)

        return {"explanations": batch_explanations}

    except Exception as e:
        logger.error(f"SHAP explanation failed: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"SHAP explanation failed: {str(e)}"
        )
