import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import bisect

class ShardedDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

        self.lengths = []
        self.cumulative = []

        total = 0
        for path in paths:
            with np.load(path) as data:
                n = len(data["imgs"])
            self.lengths.append(n)
            total += n
            self.cumulative.append(total)

        self.loaded_shard = None
        self.loaded_index = -1

    def __len__(self):
        return self.cumulative[-1]

    def _load_shard(self, shard_idx):
        if shard_idx == self.loaded_index:
            return

        data = np.load(self.paths[shard_idx])

        self.loaded_shard = (
            data["imgs"].astype(np.float32) / 8.0,
            data["labels"],
        )
        self.loaded_index = shard_idx

    def __getitem__(self, idx):
        shard = bisect.bisect_right(self.cumulative, idx)

        if shard == 0:
            local = idx
        else:
            local = idx - self.cumulative[shard - 1]

        self._load_shard(shard)

        imgs, labels = self.loaded_shard

        return (
            torch.from_numpy(imgs[local]),
            torch.from_numpy(labels[local]).long(),
        )


# =====================================================
# Configuration
# =====================================================

NPY_PATH = "./npy"
SAVE_HISTORY_JSON = "training_history"
SAVE_MODEL = "model"
SAVE_PLOT = "loss_curve"
NUM_CLASSES = 94
WEIGHT_DECAY = 1e-6


for path in [SAVE_HISTORY_JSON, SAVE_MODEL, SAVE_PLOT]:
    os.makedirs(path, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
BETA=(0.9, 0.999)

Y_SHAPE=20

model_dims =[
    [[1024, 512, 256],[0, 0, 0]],
]


# =====================================================
# Load Dataset
# =====================================================



# =====================================================
# Train / Val / Test Split (80/10/10)
# =====================================================

if input("\nSplit and scale dataset(1) Load existing(0): ").strip().lower() == "1":

    print(f"Loading from {NPY_PATH}")

    paths=os.listdir(NPY_PATH)
    input("Shards: "+len(paths))
    train_size=int(input("Enter train shards: "))
    validation_size=int(input("Enter validation shards: "))
    split1=train_size
    split2=train_size+validation_size

    with open('split.json','w') as f:
        json.dump([split1, split2],f)


else:
    data=json.load('split.json')
    split1, split2=data
    paths=os.listdir(NPY_PATH)

shards=[]
for path in paths:
    path=path[:-4]
    shards.append(path.split('-'))
shards.sort(key=lambda x: x[0])
paths=[]
for shard in shards:
    paths.append(NPY_PATH+'/'+str(shard[0])+'-'+str(shard[1])+'.npz')

train = paths[:split1]
val = paths[split1:split2]
test = paths[split2:]

# =====================================================
# DataLoaders
# =====================================================

train_dataset = ShardedDataset(train)
val_dataset = ShardedDataset(val)
test_dataset = ShardedDataset(test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
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
        layers.append(nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        ))
        layers.append(nn.ReLU())
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(64,layer_spec[0][0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(layer_spec[1][0]))
        for i in range(len(layer_spec[0])-1):

            layers.append(
                nn.Linear(
                    int(layer_spec[0][i]),
                    int(layer_spec[0][i+1]),
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(layer_spec[1][i+1]))

        self.net = nn.Sequential(*layers[:-1])
        self.final_layer = nn.Linear(int(layer_spec[0][-1]), NUM_CLASSES * Y_SHAPE)

    def forward(self, x):
        x = self.net(x)
        x = self.final_layer(x)
        return x.view(-1, NUM_CLASSES, Y_SHAPE)

# =====================================================
# Evaluation
# =====================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.reshape(-1)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            preds = preds.reshape(-1, NUM_CLASSES)
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
        betas=BETA,
        weight_decay=WEIGHT_DECAY
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
        os.path.join(SAVE_PLOT, MODEL_INDEX)+'.png',
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