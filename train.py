import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# =====================================================
# Configuration
# =====================================================

NPZ_PATH = "100000.npz"
SAVE_HISTORY_JSON = "training_history"
SAVE_MODEL = "model"
SAVE_PLOT = "loss_curve"

for path in [SAVE_HISTORY_JSON, SAVE_MODEL, SAVE_PLOT]:
    os.makedirs(path, exist_ok=True)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
BETA=(0.9, 0.999)

Y_SHAPE=20

model_dims =[
    [1024,1024],
    [1024,512],
    [1024,256],
    [512,512],
    [512,256],
    [512,128],
    [256,256],
    [256,128],
    [256,64],
    [2048],
    [1024],
    [512],
]


# =====================================================
# Load Dataset
# =====================================================



# =====================================================
# Train / Val / Test Split (80/10/10)
# =====================================================

if input("\nSplit and scale dataset(1) Load existing(0): ").strip().lower() == "1":

    print(f"Loading {NPZ_PATH}")

    data = np.load(NPZ_PATH)

    X = data["imgs"]
    y = data["labels"]

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    print("Splitting dataset...")

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
    print("X Train:", X_train.shape)
    print("y Train:", y_train.shape)
    print("X Val  :", X_val.shape)
    print("y Val  :", y_val.shape)
    print("X Test :", X_test.shape)
    print("y Test :", y_test.shape)

    print("\nScaling dataset...")

# =====================================================
# Scale Dataset
# =====================================================

    scaler = StandardScaler()
    X_train = X_train.astype("float32") / 255.0
    X_val = X_val.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    """X_train = X_train.reshape(len(X_train), -1)
    X_val   = X_val.reshape(len(X_val), -1)
    X_test  = X_test.reshape(len(X_test), -1)"""


        
    np.savez_compressed(
        "dataset_split.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

else:
    data=np.load("dataset_split.npz")
    X_train = data['X_train']
    y_train = data['y_train']

    X_val = data['X_val']
    y_val = data['y_val']

    X_test = data['X_test']
    y_test = data['y_test']

    print("\nDataset loaded from split npz:")
    print("Train:", X_train.shape)
    print("Val  :", X_val.shape)
    print("Test :", X_test.shape)


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
        
        layers = [nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        ),]
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.MaxPool2d(2))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(20000,layer_spec[0]))
        layers.append(nn.ReLU())
        for i in range(len(layer_spec)-1):

            layers.append(
                nn.Linear(
                    int(layer_spec[i]),
                    int(layer_spec[i+1]),
                )
            )
            
            layers.append(nn.ReLU())

        layers.append(nn.Linear(int(layer_spec[-1]),Y_SHAPE))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# =====================================================
# Evaluation
# =====================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)

            loss = criterion(
                preds,
                y_batch,
            )

            total_loss += loss.item()

    return total_loss / len(loader)


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



for m in range(len(model_dims)):
    model = DynamicNet(
        model_dims[m]
    )
    MODEL_INDEX=f"E-{EPOCHS}-LR-{LEARNING_RATE}-B1-{str(BETA[0]).split('.')[1]}-B2-{str(BETA[1]).split('.')[1]}"

    for dim in model_dims[m]:
        MODEL_INDEX += f"-{dim}"

    if MODEL_INDEX+".pkl" in os.listdir(SAVE_MODEL):
        continue

    print("\nModel:")
    print(model)

    model.to(device)
    # =====================================================
    # Loss / Optimizer
    # =====================================================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=BETA
    )




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
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(
                preds,
                y_batch,
            )

            optimizer.zero_grad(set_to_none=True)
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
        os.path.join(SAVE_HISTORY_JSON, MODEL_INDEX)+".json",
        "w",
    ) as f:
        
        json.dump(
            history,
            f,
            indent=4,
        )
    print(
        "Saved history:",
        os.path.join(SAVE_HISTORY_JSON, MODEL_INDEX)+".json",
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
        os.path.join(SAVE_PLOT, MODEL_INDEX),
        dpi=300,
    )
    plt.close()
    print(
        "Saved plot:",
        os.path.join(SAVE_PLOT, MODEL_INDEX)+".png",
    )


    # =====================================================
    # Save Model
    # =====================================================

    torch.save(
        model.state_dict(),
        os.path.join(SAVE_MODEL, MODEL_INDEX)+".pkl",
    )

    print(
        "Saved model:",
        os.path.join(SAVE_MODEL, MODEL_INDEX)+".pkl",
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