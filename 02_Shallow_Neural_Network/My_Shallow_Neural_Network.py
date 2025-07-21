# X: shape (12288, m)
# Y: shape (1, m)
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Loading images to X 


def load_images_from_folders(folder_paths, label_values, image_size=(64, 64)):
    X_list = []
    Y_list = []

    for folder_path, label in zip(folder_paths, label_values):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                img_array = np.array(img).reshape(-1, 1) / \
                    255.0  # Normalize to [0,1]
                X_list.append(img_array)
                Y_list.append(label)

    X = np.hstack(X_list)               # Shape: (features, m)
    Y = np.array(Y_list).reshape(1, -1)  # Shape: (1, m)
    return X, Y


X, Y = load_images_from_folders(
    folder_paths=[r"C:\Users\HP\Desktop\Learning\AI\main.py\BC_Images\1",
                  r"C:\Users\HP\Desktop\Learning\AI\main.py\BC_Images\0"],
    label_values=[1, 0],
    image_size=(64, 64)
)

X_test, Y_test = load_images_from_folders(
    folder_paths=[r"C:\Users\HP\Desktop\Learning\AI\main.py\BC_Images\Test\1",
                  r"C:\Users\HP\Desktop\Learning\AI\main.py\BC_Images\Test\0"],
    label_values=[1, 0],
    image_size=(64, 64)
)

print("Loaded training images are", X.shape[1])
print("Loaded testing images are", X_test.shape[1])
print("Input per image are", X.shape[0])

# Shuffle data
perm = np.random.permutation(X.shape[1])
X = X[:, perm]
Y = Y[:, perm]

m = X.shape[1]


# Model

n_h = 10
n_x = X.shape[0]
n_y = Y.shape[0]


def sigmoid(Z):
    return 1/(1+np.exp(-Z))


def nReLU(Z):
    return np.maximum(0, Z)


def nReLU_derivative(Z):
    return (Z < 0).astype(float)


def initialize(n_h, n_x, n_y):
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))

    params = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return params


def forward(params, X):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1, X)+b1
    A1 = np.tanh(Z1)

    # A1 = nReLU(Z1)
    Z2 = np.dot(W2, A1)+b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return cache


def compute_cost(Y, cache):
    m = Y.shape[1]
    A2 = cache['A2']

    cost = -(1/m)*(np.sum(Y*np.log(A2)+(1-Y)*np.log(1-A2)))
    return float(np.squeeze(cost))


def backward(cache, params, Y, X):
    m = X.shape[1]
    W2 = params['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    Z1 = cache['Z1']

    dZ2 = A2-Y
    dW2 = 1/m*(np.dot(dZ2, A1.T))
    db2 = 1/m*np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)*(1-(np.power(A1, 2)))
    # dZ1 = np.dot(W2.T, dZ2) * nReLU_derivative(Z1)
    dW1 = 1/m*(np.dot(dZ1, X.T))
    db1 = 1/m*np.sum(dZ1, axis=1, keepdims=True)

    grads = {'dW2': dW2,
             'db2': db2,
             'dW1': dW1,
             'db1': db1}

    return grads


def optimize(params, grads, lr):
    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']
    dW2, dW1, db1, db2 = grads['dW2'], grads['dW1'], grads['db1'], grads['db2']

    W1 = W1-lr*dW1
    W2 = W2-lr*dW2
    b1 = b1-lr*db1
    b2 = b2-lr*db2

    params = {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}
    return params


def model_v228(X, Y, n_h, lr, iterations, print_cost):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    params = initialize(n_h, n_x, n_y)
    costs = []

    tic = time.time()
    for i in range(iterations):
        cache = forward(params, X)
        cost = compute_cost(Y, cache)
        grads = backward(cache, params, Y, X)
        params = optimize(params, grads, lr)

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 2 == 0:
            print(f"Cost after iteration {i}: {cost:.4f}")
    toc = time.time()

    print('Training time:', round((toc-tic)/60, 2), 'minutes')

    # Plot the cost
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Every 100 Iterations')
    plt.title(f"Learning rate = {lr}")
    plt.grid(True)
    plt.show()

    return params


def predict(X, params):
    cache = forward(params, X)
    A2 = cache['A2']
    prediction = (A2 > 0.5).astype(int)
    return prediction


params = model_v228(X, Y, n_h=34,
                    lr=0.03, iterations=6800, print_cost=True)

np.savez("best_model__79.npz", **params)

prediction = predict(X, params)
accuracy = np.mean(prediction == Y) * 100
print(f"Training Accuracy: {accuracy:.2f}%")

prediction = predict(X_test, params)
accuracy = np.mean(prediction == Y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

test_pred = predict(X_test, params)
print("Test Prediction Sample:", test_pred[:, :10])
print("Ground Truth Sample:   ", Y_test[:, :10])
