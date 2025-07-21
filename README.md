# 🚀 Deep Learning Journey from Scratch

This repository documents my step-by-step journey to becoming an AI/ML Engineer. Everything here is coded from **scratch using only NumPy** — no high-level libraries like TensorFlow or PyTorch. This is a hands-on implementation of the core building blocks of deep learning.

---

## 🧠 Project Structure

### 1. Linear Regression (01_Linear_Regression/)
- Implemented a simple linear regression model from scratch.
- Used gradient descent for optimization.

### 2. Shallow Neural Network (02_Shallow_Neural_Network/)
- 2-layer neural network (1 hidden layer).
- Implemented forward and backward propagation manually.

### 3. Deep Neural Network for Binary Image Classification (03_Deep_Neural_Network/)
- Trained on a **Cat vs. Dog** dataset (5000 images each).
- Implemented:
  - Deep architectures with customizable layers
  - Sigmoid/ReLU activations
  - Vectorized forward/backward propagation
  - Mini-batch gradient descent
  - L2 regularization
  - Dropout
  - Adam and GD optimizers
- Achieved **76% Test Accuracy** without any high-level libraries.

📸 **Result:**
![76% Accuracy](03_Deep_Neural_Network/results/test_accuracy_76_percent.png)

---

## 📦 Dataset

Images were resized and converted to `.npy` format for efficiency. You can plug your own datasets into the pipeline.

This project uses the **Cat vs. Dog** dataset from [Kaggle](https://www.kaggle.com/competitions/dogs-vs-cats/data).

#### 🔻 How to set up the dataset

1. Go to the [Kaggle competition page](https://www.kaggle.com/competitions/dogs-vs-cats/data)
2. Download the `train.zip` file (you'll need a Kaggle account).
3. Extract the contents — you’ll get image files like `cat.1.jpg`, `dog.1234.jpg`, etc.
4. Place the extracted files in a folder named `dataset/train/` inside this project directory:

---

## 🛠 Why I Did This

- To deeply understand how ML and DL models work internally.
- To build solid intuition for training dynamics, overfitting, regularization, and optimization.
- To prepare myself for more advanced architectures like CNNs, RNNs, and Transformers.

---

## ✅ Next Steps

- Add cost function plots and accuracy graphs
- Start CNN implementation from scratch
- Continue the DeepLearning.ai specialization (currently completed Course 2)

---

> Built with patience, sweat, bugs, and NumPy 🧪

