import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import gc

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

# =====================================================
# Configuration
# =====================================================

NPZ_PATH = "./npz"
SAVE_HISTORY_JSON = "training_history"
SAVE_MODEL = "model"
SAVE_PLOT = "loss_curve"
NUM_CLASSES = 53
WEIGHT_DECAY = 1e-6
AAP_OUT = 4


for path in [SAVE_HISTORY_JSON, SAVE_MODEL, SAVE_PLOT]:
    os.makedirs(path, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
BETA=(0.9, 0.999)

Y_SHAPE=20

# This is a quick way to change model layer dimensions. Each entry in the list is a model configuration. The first sublist specifies the dimensions of the convolution layers, the second sublist is the dense layer dimensions, and the third sublist is the dropout rates for each dense layer.

model_dims =[
    [[32,64],[1024, 512, 256],[0, 0]],
]

# =====================================================
# Loader loads data from npz files on disk to RAM and scales it to [0,1] as training progresses. Labels are also converted to 0-52 range for training.
# =====================================================

class NPZDataset(Dataset):
    def __init__(self, path):

        with np.load(path) as data:

            self.imgs = data["imgs"].astype(np.float32) / 8.0

            labels = data["labels"]

            labels = labels.astype(np.int64)
            upper = labels < 91
            labels[upper] -= 64
            labels[~upper] -= 81

            self.labels = labels

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]




# =====================================================
# Dynamic Model
# =====================================================

class DynamicNet(nn.Module):

    def __init__(self, layer_spec):
        super().__init__()
        layers=[]
        for i in range(len(layer_spec[0])):
            layers.append(nn.Conv2d(
                in_channels=3 if i==0 else int(layer_spec[0][i-1]),
                out_channels=int(layer_spec[0][i]),
                kernel_size=3,
                padding=1
            ))
            layers.append(nn.ReLU())
            if i!=len(layer_spec[0])-1:
                layers.append(nn.MaxPool2d(2))
        layers.append(nn.AdaptiveAvgPool2d((AAP_OUT, AAP_OUT)))
        layers.append(nn.Flatten())
        for i in range(len(layer_spec[1])):
            layers.append(
                nn.Linear(
                    in_features = layer_spec[0][-1]*(AAP_OUT**2) if i == 0 else int(layer_spec[1][i-1]),
                    out_features=int(layer_spec[1][i]),
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(layer_spec[2][i+1]))

        self.net = nn.Sequential(*layers[:-1])
        self.final_layer = nn.Linear(int(layer_spec[1][-1]), NUM_CLASSES * Y_SHAPE)

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
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            y_batch = y_batch.reshape(-1)
            preds = model(X_batch)
            preds = preds.reshape(-1, NUM_CLASSES)
            loss = criterion(
                preds,
                y_batch,
            )

            total_loss += loss.item()

    return total_loss / len(loader)



if __name__=='__main__':
    

    # =====================================================
    # Load Dataset
    # =====================================================



    # =====================================================
    # Train / Val / Test Split by shard
    # =====================================================

    if input("\nSplit and scale dataset(1) Load existing(0): ").strip().lower() == "1":

        print(f"Loading from {NPZ_PATH}")

        paths=os.listdir(NPZ_PATH)
        print("Shards: "+str(len(paths)))
        train_size=int(input("Enter train shards: "))
        validation_size=int(input("Enter validation shards: "))
        split1=train_size
        split2=train_size+validation_size

        with open('split.json','w') as f:
            json.dump([split1, split2],f)


    else:
        with open("split.json", "r") as f:
            data = json.load(f)
        split1, split2=data
        paths=os.listdir(NPZ_PATH)

    shards=[]
    for path in paths:
        path=path[:-4]
        shards.append(path.split('-'))
    shards.sort(key=lambda x: int(x[0]))
    paths=[]
    for shard in shards:
        paths.append(NPZ_PATH+'/'+str(shard[0])+'-'+str(shard[1])+'.npz')
    
    for path in paths:
        print(path)

    train = paths[:split1]
    val = paths[split1:split2]
    test = paths[split2:]


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

        model.to(device, non_blocking=True)
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
            random.shuffle(train)
            model.train()
            running_loss = 0.0
            num_batches = 0
            for shard_path in train:
                train_dataset = NPZDataset(shard_path)

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=12,
                    pin_memory=True,
                )

                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)

                    y_batch = y_batch.reshape(-1)

                    preds = model(X_batch)
                    preds = preds.reshape(-1, NUM_CLASSES)
                    loss = criterion(
                        preds,
                        y_batch,
                    )

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    num_batches += 1

                del train_dataset
                del train_loader
            gc.collect()

            val_loss=0.0
            for shard in val:
                dataset = NPZDataset(shard)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)
                val_loss+=evaluate(loader)

            train_loss = (
                running_loss
                / num_batches
            )
            history["train_loss"].append(
                train_loss
            )
            history["val_loss"].append(
                val_loss/len(val)
            )
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f}"
            )

        with open(os.path.join(SAVE_HISTORY_JSON, MODEL_INDEX)+".json", "w", ) as f:
            json.dump(history, f, indent=4)
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
        print("\nTraining complete.")