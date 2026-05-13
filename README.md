# Deep Learning Course Project: Handwritten Digit Recognition using CNN

## 1. Problem Description
This project implements a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The goal is to classify grayscale images of handwritten digits (0-9) accurately.

## 2. Dataset Link
The MNIST dataset is used for this project. It can be downloaded automatically by the provided Python script. More information about the dataset can be found [here](http://yann.lecun.com/exdb/mnist/).

## 3. Results (Comparison Table)
Two experiments were conducted to compare the performance of different optimizers: Adam and SGD. The models were trained for 5 epochs with a batch size of 64.

| Model     | Accuracy (%) | Loss   |
|-----------|--------------|--------|
| Model_Adam| 98.36        | 0.0523 |
| Model_SGD | 97.26        | 0.0862 |

## 4. Visualizations
Training and validation loss and accuracy curves are provided in the `plots/` directory. A comparison plot for both models is available as `comparison_plots.png`.

## 5. Instructions for Running the Project

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Installation
1. Clone the repository (or download the files).
2. Navigate to the project directory:
   ```bash
   cd mnist_project
   ```
3. Install the required Python packages:
   ```bash
   sudo pip3 install torch torchvision matplotlib pandas seaborn
   ```

### Running the Experiments and Generating Results
To train the models, run the `train.py` script. This script will download the MNIST dataset, train two CNN models (one with Adam optimizer and one with SGD optimizer), and save their training histories in the `results/` directory.

```bash
python3.11 train.py
```

### Generating Visualizations
After running the `train.py` script, execute the `visualize.py` script to generate plots of training/validation loss and accuracy curves. The plots will be saved in the `plots/` directory.

```bash
python3.11 visualize.py
```

### Project Structure
```
mnist_project/
├── train.py             # Main script for model training and experimentation
├── visualize.py         # Script for generating visualizations
├── results/             # Directory to store training history JSON files
│   ├── Model_Adam_history.json
│   └── Model_SGD_history.json
└── plots/               # Directory to store generated plots
    ├── comparison_plots.png
    ├── Model_Adam_plots.png
    └── Model_SGD_plots.png
```
