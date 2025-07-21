 # Import required libraries
import numpy as np
import copy
import matplotlib.pyplot as plt
from lr_utils import load_dataset

# Load the dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Get shapes and dimensions
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# Reshape the training and test set
train_set_x_flatten = train_set_x_orig.reshape(
    m_train, -1).T  # shape (12288, 209)
test_set_x_flatten = test_set_x_orig.reshape(
    m_test, -1).T     # shape (12288, 50)

# Standardize pixel values (0-255 to 0-1)
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

# ----------------------------- #
# Define sigmoid activation function


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# ----------------------------- #
# Initialize parameters to zero


def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b

# ----------------------------- #
# Forward and backward propagation


def propagate(w, b, X, Y):
    m = X.shape[1]

    # Forward propagation (activation)
    A = sigmoid(np.dot(w.T, X) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m  # Compute cost

    dw = (1 / m) * np.dot(X, (A - Y).T)          # Backward propagation
    db = (1 / m) * np.sum(A - Y)

    grads = {"dw": dw, "db": db}
    return grads, np.squeeze(cost)

# ----------------------------- #
# Optimize parameters using gradient descent


def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i}: {cost}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

# ----------------------------- #
# Predict labels using learned weights


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    return Y_prediction

# ----------------------------- #
# Combine everything into a model


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.005, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train,
                                    num_iterations=num_iterations,
                                    learning_rate=learning_rate,
                                    print_cost=print_cost)

    w = params["w"]
    b = params["b"]

    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    if print_cost:
        print("train accuracy: {} %".format(
            100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(
            100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations
    }

    return d


# ----------------------------- #
# Train the model
logistic_regression_model = model(
    train_set_x, train_set_y,
    test_set_x, test_set_y,
    num_iterations=2000,
    learning_rate=0.005,
    print_cost=True
)

# ----------------------------- #
# Plot the learning curve (cost over iterations)
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate = " + str(logistic_regression_model["learning_rate"]))
plt.show()
