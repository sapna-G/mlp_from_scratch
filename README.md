# Multi-Layer Perceptron (MLP) from Scratch in NumPy

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://handwrittentextdetector.streamlit.app/)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![NumPy Only](https://img.shields.io/badge/core-NumPy%20Only-013243.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

**[Try the Live Interactive Demo](https://handwrittentextdetector.streamlit.app/)**

A modular, pure-NumPy neural network framework built from scratch without PyTorch, TensorFlow, or JAX. Implements vectorized forward and backward propagation, analytical gradient checking, mini-batch training on MNIST, and a live Streamlit inference web app.

## Key Highlights
- **Zero Deep Learning Frameworks:** Matrix operations, activations, loss functions, and stateful optimizers implemented purely in NumPy.
- **Verified Backpropagation:** Analytical gradients verified against central finite differences (`max relative error < 1e-6`).
- **97.68% Validation Accuracy:** Achieved in 10 epochs on MNIST (`784 -> 128 -> 64 -> 10`) using He initialization, ReLU, Fused Softmax Cross-Entropy, and Adam.
- **Interactive Inference UI:** Draw digits directly on a canvas or upload images with automatic grayscale/inversion preprocessing.

## Project Structure
- `activations.py` - `ReLU`, `Sigmoid`, and numerically stable `Softmax`
- `layers.py` - `Dense` layer with He & Xavier weight initialization
- `losses.py` - `MSELoss`, `CrossEntropyLoss`, and fused `SoftmaxCrossEntropy`
- `optimizers.py` - Stateful `SGD` (with momentum) and `Adam`
- `model.py` - `Sequential` container
- `grad_check.py` - Numerical vs. analytical gradient verification
- `train.py` - Mini-batch MNIST training pipeline (exports `weights.npz`)
- `app.py` - Streamlit interactive digit classifier
- `weights.npz` - Pre-trained model parameters (97.68% val acc)
- `requirements.txt` - Project dependencies

## Quickstart
1. Install dependencies: `pip install -r requirements.txt`
2. Verify backpropagation math: `python grad_check.py`
3. Train from scratch (optional): `python train.py`
4. Launch the interactive UI: `streamlit run app.py`
