HTML OCR

Description:
This is a convolutional neural network for recognizing characters trained on a dataset rendered using HTML.

Features

- Trains on millions of generated images
- Supports dataset sharding
- Fast multiprocessing data loader
- Scalability, running on strong and weak systems

Project structure

project/
├── train.py
├── model.py
├── image_gen.py
├── image_gen_props.py
├── requirements.txt
└── README.md

Usage

Run
python image_gen.py, changing min_ln, and max_ln, the maximum and minumum length of the text generated, respectively.
Workers dictates the maximum number of workers dedicated to the task, shard_start and shard_end allow control over which shards are generated, meaning one can only regenerate failed shards.

Model architecture

Conv2d(3→32)
MaxPool
Conv2d(32→64)
AdaptiveAvgPool
Linear(64→1024→512→256→classes)

Training

Optimizer: Adam
Loss: CrossEntropyLoss
Batch size: 1024
Epochs: 20

Requirements

Python 3.12
PyTorch
NumPy
Playwright chromium
Pillow
