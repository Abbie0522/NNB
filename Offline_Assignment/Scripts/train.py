
import os
import random
import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Dataset"
OUTPUT_DIR = ROOT / "Outputs"
MODEL_DIR = ROOT / "Model"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

device = torch.device("cpu")


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


def prepare_data():
    df = pd.read_csv(DATA_DIR / "bank-full.csv", sep=";")
    df = df.drop_duplicates()
    df = df.drop(columns=["duration"])

    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50,
        random_state=SEED, stratify=y_temp
    )

    categorical = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    preprocessor = ColumnTransformer([
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ))
        ]), categorical),

        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical)
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    joblib.dump(preprocessor, OUTPUT_DIR / "preprocessor.pkl")

    for name, data in {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy()
    }.items():
        np.save(OUTPUT_DIR / f"{name}.npy", data)

    print("Data preprocessing completed.")
    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    return X_train, X_val, y_train.to_numpy(), y_val.to_numpy()


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    labels_all = []
    predictions_all = []

    with torch.no_grad():
        for inputs, labels in loader:
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(inputs)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).int()

            labels_all.extend(labels.numpy().ravel())
            predictions_all.extend(predictions.numpy().ravel())

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "precision": precision_score(
            labels_all, predictions_all, zero_division=0
        ),
        "recall": recall_score(
            labels_all, predictions_all, zero_division=0
        ),
        "f1": f1_score(
            labels_all, predictions_all, zero_division=0
        )
    }


def train_experiment(name, learning_rate, batch_size, train_ds, val_ds, input_size):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = BankMarketingMLP(input_size)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, generator=generator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    history = []
    best_f1 = -1.0
    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in range(30):
        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        train_metrics = evaluate(model, train_loader, criterion)
        val_metrics = evaluate(model, val_loader, criterion)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"]
        })

        if (val_metrics["f1"] > best_f1 or
            (val_metrics["f1"] == best_f1 and
             val_metrics["loss"] < best_loss)):
            best_f1 = val_metrics["f1"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"{name} | Epoch {epoch+1:02d}/30 | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}",
            flush=True
        )

    model.load_state_dict(best_state)

    return {
        "name": name,
        "model": model,
        "history": pd.DataFrame(history),
        "best_epoch": best_epoch,
        "best_f1": best_f1,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }


def main():
    X_train, X_val, y_train, y_val = prepare_data()

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    )

    experiments = [
        ("Baseline", 0.001, 32),
        ("Experiment 1", 0.0005, 32),
        ("Experiment 2", 0.001, 64)
    ]

    results = []
    comparison = []

    for name, lr, batch in experiments:
        result = train_experiment(
            name, lr, batch, train_ds, val_ds, X_train.shape[1]
        )
        results.append(result)

        filename = name.lower().replace(" ", "_")
        result["history"].to_csv(
            OUTPUT_DIR / f"{filename}_history.csv", index=False
        )

        row = result["history"].iloc[result["best_epoch"] - 1]

        comparison.append({
            "Experiment": name,
            "Learning Rate": lr,
            "Batch Size": batch,
            "Best Epoch": result["best_epoch"],
            "Val Accuracy": row["val_accuracy"],
            "Val Precision": row["val_precision"],
            "Val Recall": row["val_recall"],
            "Val F1": row["val_f1"]
        })

    pd.DataFrame(comparison).to_csv(
        OUTPUT_DIR / "task4_experiment_comparison.csv",
        index=False
    )

    best = max(results, key=lambda x: x["best_f1"])

    torch.save({
        "model_state_dict": best["model"].state_dict(),
        "input_size": X_train.shape[1],
        "hidden_layers": [64, 32],
        "best_experiment": best["name"],
        "best_epoch": best["best_epoch"],
        "validation_f1": best["best_f1"],
        "classification_threshold": 0.5
    }, MODEL_DIR / "best_mlp_model.pth")

    print("\nTraining completed successfully!")
    print("Best Experiment:", best["name"])
    print("Best Epoch:", best["best_epoch"])
    print("Best Validation F1:", best["best_f1"])
    print("Model saved:", MODEL_DIR / "best_mlp_model.pth")


if __name__ == "__main__":
    main()
