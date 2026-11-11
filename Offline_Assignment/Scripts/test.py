from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "Outputs"
MODEL_DIR = ROOT / "Model"

class BankMarketingMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

def predict_new_customer(model, threshold):
    sample_path = OUTPUT_DIR / "new_customer.json"

    if not sample_path.exists():
        print("\nNew sample prediction skipped: JSON file not found.")
        return

    with open(sample_path, "r", encoding="utf-8") as file:
        sample = json.load(file)

    preprocessor = joblib.load(OUTPUT_DIR / "preprocessor.pkl")

    expected_features = list(preprocessor.feature_names_in_)
    missing = [col for col in expected_features if col not in sample]
    extra = [col for col in sample if col not in expected_features]

    if missing or extra:
        raise ValueError(
            f"Missing features: {missing}; Extra features: {extra}"
        )

    customer = pd.DataFrame([
        {col: sample[col] for col in expected_features}
    ])

    transformed = preprocessor.transform(customer)

    with torch.no_grad():
        inputs = torch.tensor(transformed, dtype=torch.float32)
        probability = torch.sigmoid(model(inputs)).item()

    prediction = "Yes" if probability >= threshold else "No"

    print("\n--- New Customer Prediction ---")
    print("Sample:", sample_path.name)
    print(f"Probability of subscription (Yes): {probability:.4f}")
    print(f"Prediction: {prediction}")
    print(f"Decision threshold: {threshold:.2f}")

def main():
    X_test = np.load(OUTPUT_DIR / "X_test.npy")
    y_test = np.load(OUTPUT_DIR / "y_test.npy")

    checkpoint = torch.load(
        MODEL_DIR / "best_mlp_model.pth",
        map_location="cpu",
        weights_only=True
    )

    model = BankMarketingMLP(checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold = checkpoint["classification_threshold"]

    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32)
        logits = model(inputs)
        probabilities = torch.sigmoid(logits).numpy().ravel()
        predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    print("Best Experiment:", checkpoint["best_experiment"])
    print("Test Samples:", len(y_test))
    print("\nFinal Test Results")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    pd.DataFrame(
        list(metrics.items()),
        columns=["Metric", "Value"]
    ).to_csv(
        OUTPUT_DIR / "docker_test_metrics.csv",
        index=False
    )

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No", "Yes"]
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Docker Testing - Confusion Matrix")

    fig.savefig(
        OUTPUT_DIR / "docker_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    print("\nConfusion Matrix:")
    print(cm)
    print("\nTesting completed successfully!")
    predict_new_customer(model, threshold)

if __name__ == "__main__":
    main()
