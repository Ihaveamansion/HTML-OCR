import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =====================================================
# Configuration
# =====================================================

NPZ_PATH = "imgs.npz"
SAVE_SPLIT_PKL = "dataset_split.pkl"
SAVE_HISTORY_JSON = "training_history"
SAVE_MODEL = "model"
SAVE_PLOT = "loss_curve"

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3


# =====================================================
# Load Dataset
# =====================================================

print(f"Loading {NPZ_PATH}")

data = np.load(NPZ_PATH)

X = data["imgs"]
y = data["labels"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# =====================================================
# Train / Val / Test Split (80/10/10)
# =====================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
)

print("\nDataset split:")
print("Train:", X_train.shape)
print("Val  :", X_val.shape)
print("Test :", X_test.shape)


# =====================================================
# Scaling
# =====================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("\nSaving split and scaler...")

with open(SAVE_SPLIT_PKL, "wb") as f:
    pickle.dump(
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "scaler": scaler,
        },
        f,
    )

print("Saved:", SAVE_SPLIT_PKL)


# =====================================================
# Optional: Read Existing PKL Scaler
# =====================================================

load_existing_scaler = (
    input(
        "\nLoad scaler from another .pkl? (1/0): "
    )
    .strip()
    .lower()
)

if load_existing_scaler == "1":

    scaler_path = input(
        "Enter scaler .pkl path: "
    ).strip()

    with open(scaler_path, "rb") as f:
        loaded = pickle.load(f)

    if hasattr(loaded, "transform"):
        scaler = loaded

    elif isinstance(loaded, dict) and "scaler" in loaded:
        scaler = loaded["scaler"]

    else:
        raise ValueError(
            "Could not locate scaler in pkl file."
        )

    print("Scaler loaded successfully.")


# =====================================================
# DataLoaders
# =====================================================

def make_loader(X, y, batch_size=64, shuffle=True):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


train_loader = make_loader(
    X_train,
    y_train,
    BATCH_SIZE,
    True,
)

val_loader = make_loader(
    X_val,
    y_val,
    BATCH_SIZE,
    False,
)

test_loader = make_loader(
    X_test,
    y_test,
    BATCH_SIZE,
    False,
)


# =====================================================
# Dynamic Model
# =====================================================

class DynamicNet(nn.Module):

    def __init__(self, layer_spec):
        super().__init__()

        layers = []

        for i, (in_features, out_features) in enumerate(layer_spec):

            layers.append(
                nn.Linear(
                    int(in_features),
                    int(out_features),
                )
            )

            if i != len(layer_spec) - 1:
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# =====================================================
# Example 3D Architecture Array
# =====================================================
#
# Shape:
# [model][layer][in,out]
#
# Example:
# input -> 256 -> 128 -> 1
#
# =====================================================

model_dims = np.array(
    [
        [
            [X.shape[1], 256],
            [256, 128],
            [128, 10],
        ]
    ]
)

for MODEL_INDEX in range(len(model_dims)):
    model = DynamicNet(
        model_dims[MODEL_INDEX]
    )

    print("\nModel:")
    print(model)


    # =====================================================
    # Loss / Optimizer
    # =====================================================

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )


    # =====================================================
    # Evaluation
    # =====================================================

    def evaluate(loader):

        model.eval()

        total_loss = 0.0

        with torch.no_grad():

            for X_batch, y_batch in loader:

                preds = model(X_batch)

                loss = criterion(
                    preds,
                    y_batch,
                )

                total_loss += loss.item()

        return total_loss / len(loader)


    # =====================================================
    # Training
    # =====================================================

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for X_batch, y_batch in train_loader:

            preds = model(X_batch)

            loss = criterion(
                preds,
                y_batch,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss
            / len(train_loader)
        )

        val_loss = evaluate(
            val_loader
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS} | "
            f"Train: {train_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )


    # =====================================================
    # Test Evaluation
    # =====================================================

    test_loss = evaluate(
        test_loader
    )

    history["test_loss"] = float(
        test_loss
    )

    print(
        f"\nFinal Test Loss: "
        f"{test_loss:.6f}"
    )


    # =====================================================
    # Save History JSON
    # =====================================================

    with open(
        SAVE_HISTORY_JSON,
        "w",
    ) as f:

        json.dump(
            history,
            f,
            indent=4,
        )

    print(
        "Saved history:",
        SAVE_HISTORY_JSON,
    )


    # =====================================================
    # Save Plot
    # =====================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        history["train_loss"],
        label="Train Loss",
    )

    plt.plot(
        history["val_loss"],
        label="Validation Loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        MODEL_INDEX/SAVE_PLOT,
        dpi=300,
    )

    plt.close()

    print(
        "Saved plot:",
        MODEL_INDEX/SAVE_PLOT,
    )


    # =====================================================
    # Save Model
    # =====================================================

    torch.save(
        model.state_dict(),
        MODEL_INDEX/SAVE_MODEL,
    )

    print(
        "Saved model:",
        MODEL_INDEX/SAVE_MODEL,
    )


    # =====================================================
    # Example Loading Saved Scaler Later
    # =====================================================
    #
    # with open("dataset_split.pkl", "rb") as f:
    #     data = pickle.load(f)
    #
    # scaler = data["scaler"]
    # X_new = scaler.transform(X_new)
    #
    # =====================================================

    print("\nTraining complete.")