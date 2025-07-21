import numpy as np
import copy
import time
from PIL import Image
import matplotlib.pyplot as plt
import os
import math 


# Loading images to X, Y, X_test, Y_test

X = np.load(
    r"C:\Users\HP\Desktop\Learning\AI\dataset\Cat_and_dog_dataset\Dataset_files\X_train.npy")
Y = np.load(
    r"C:\Users\HP\Desktop\Learning\AI\dataset\Cat_and_dog_dataset\Dataset_files\Y_train.npy")
X_test = np.load(
    r"C:\Users\HP\Desktop\Learning\AI\dataset\Cat_and_dog_dataset\Dataset_files\X_test.npy")
Y_test = np.load(
    r"C:\Users\HP\Desktop\Learning\AI\dataset\Cat_and_dog_dataset\Dataset_files\Y_test.npy")

# Check shapes
print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# Loaded Images
print("Loaded training images are", X.shape[1])
print("Loaded testing images are", X_test.shape[1])
print("Input per image are", X.shape[0])


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def relu(Z):
    A = np.maximum(0, Z)
    return A


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def adam_initialization(parameters):
    L = len(parameters)//2
    v = {}
    s = {}

    for l in range(1, L+1):
        v['dW'+str(l)] = np.zeros(parameters['W'+str(l)].shape)
        v['db'+str(l)] = np.zeros(parameters['b'+str(l)].shape)
        s['dW'+str(l)] = np.zeros(parameters['W'+str(l)].shape)
        s['db'+str(l)] = np.zeros(parameters['b'+str(l)].shape)

    return v, s


def update_parameters_with_adam(parameters, grads, v, s, t, beta1, beta2, learning_rate, epsilon):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L+1):
        v["dW" + str(l)] = beta1 * v['dW'+str(l)] + \
            (1-beta1)*grads["dW" + str(l)]
        v["db" + str(l)] = beta1 * v['db'+str(l)] + \
            (1-beta1)*grads["db" + str(l)]
        v_corrected["dW" + str(l)] = v["dW" + str(l)] / (1 - beta1**t)
        v_corrected["db" + str(l)] = v["db" + str(l)] / (1 - beta1**t)
        s["dW" + str(l)] = beta2 * s['dW'+str(l)] + \
            (1-beta2)*np.square(grads["dW" + str(l)])
        s["db" + str(l)] = beta2 * s['db'+str(l)] + \
            (1-beta2)*np.square(grads["db" + str(l)])
        s_corrected["dW" + str(l)] = s["dW" + str(l)] / (1 - beta2**t)
        s_corrected["db" + str(l)] = s["db" + str(l)] / (1 - beta2**t)

        parameters["W" + str(l)] = parameters["W" + str(l)] - learning_rate * (
            v_corrected["dW" + str(l)] / (np.sqrt(s_corrected["dW" + str(l)])+epsilon))
        parameters["b" + str(l)] = parameters["b" + str(l)] - learning_rate * (
            v_corrected["db" + str(l)] / (np.sqrt(s_corrected["db" + str(l)])+epsilon))

    return parameters, v, s


def mini_batch_div(X, Y, mini_batch_size, seed):
    np.random.seed(seed)
    mini_batches = []
    m = X.shape[1]

    # Shuffle data
    perm = np.random.permutation(X.shape[1])
    X = X[:, perm]
    Y = Y[:, perm]

    num_complete_minibatches = math.floor(m/mini_batch_size)

    for i in range(0, num_complete_minibatches):
        mini_batch_X = X[:, i*mini_batch_size:(i+1)*mini_batch_size]
        mini_batch_Y = Y[:, i*mini_batch_size:(i+1)*mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = X[:, num_complete_minibatches*mini_batch_size:m]
        mini_batch_Y = Y[:, num_complete_minibatches*mini_batch_size:m]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def initialize(layers_dims):

    np.random.seed(3)
    L = len(layers_dims)
    parameters = {}

    for i in range(1, L):
        parameters['W'+str(i)] = np.random.randn(layers_dims[i],
                                                 # He init
                                                 layers_dims[i-1]) * np.sqrt(2. / layers_dims[i-1])
        parameters['b'+str(i)] = np.zeros((layers_dims[i], 1))

    return parameters


def linear_forward(A, W, b):
    Z = np.dot(W, A)+b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    if activation == 'sigmoid':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = sigmoid(Z)
        activation_cache = Z

    elif activation == 'relu':
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = relu(Z)
        activation_cache = Z

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, keep_probs):
    caches = []
    L = len(parameters)//2
    A = X
    drop = {}
    np.random.seed(1)

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(
            A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation='relu')
        caches.append(cache)

        if keep_probs[l-1] < 1:
            d = np.random.rand(A.shape[0], A.shape[1])
            drop['d' + str(l)] = (d < keep_probs[l-1]).astype(int)
            A = (A * (drop['d'+str(l)])) / keep_probs[l-1]

    AL, cache = linear_activation_forward(
        A, parameters['W'+str(L)], parameters['b'+str(L)], activation='sigmoid')
    caches.append(cache)

    return AL, caches, drop


def compute_cost(AL, Y):

    m = Y.shape[1]
    epsilon = 1e-8

    cost = -(1/m*(np.dot(Y, np.log(AL+epsilon).T) +
             np.dot((1-Y), np.log(1-AL+epsilon).T)))
    cost = np.squeeze(cost)

    return cost


def compute_cost_with_regularization(AL, Y, parameters, lambd):
    m = Y.shape[1]
    entropy_cost = compute_cost(AL, Y)
    L2_cost = 0

    L = len(parameters)//2

    for i in range(1, L+1):
        L2_cost += np.sum(np.square(parameters['W' + str(i)]))

    L2_cost = lambd/(2*m) * L2_cost
    cost = entropy_cost + L2_cost

    cost = np.squeeze(cost)

    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward_with_regularization(AL, Y, caches, lambd, parameters, drop, keep_probs):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    epsilon = 1e-8

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL+epsilon))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
        dAL, current_cache, activation='sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp + (lambd/m) * parameters['W'+str(L)]
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l+1)], current_cache, activation='relu')

        if l > 0 and keep_probs[l-1] < 1:
            grads["dA" + str(l)] = dA_prev_temp * \
                drop['d' + str(l)] / keep_probs[l-1]
        else:
            grads["dA" + str(l)] = dA_prev_temp

        grads["dW" + str(l+1)] = dW_temp + (lambd/m) * parameters['W'+str(l+1)]
        grads["db" + str(l+1)] = db_temp

    return grads


def L_model_backward(AL, Y, caches, drop, keep_probs):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    epsilon = 1e-8

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL+epsilon))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
        dAL, current_cache, activation='sigmoid')
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
            grads["dA" + str(l+1)], current_cache, activation='relu')

        if l > 0 and keep_probs[l-1] < 1:
            grads["dA" + str(l)] = dA_prev_temp * \
                drop['d' + str(l)] / keep_probs[l-1]
        else:
            grads["dA" + str(l)] = dA_prev_temp

        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp

    return grads


def update_parameters(params, grads, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(L):

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - \
            learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - \
            learning_rate * grads["db" + str(l+1)]

    return parameters


def model_v228(X, Y, learning_rate, layers_dims, num_iterations, lambd, keep_probs, mini_batch_size,  print_cost, beta1, beta2, epsilon, optimizer, regularization=True):

    parameters = initialize(layers_dims)
    costs = []
    tic = time.time()
    mini_batches = mini_batch_div(X, Y, mini_batch_size, seed=0)
    v, s = adam_initialization(parameters)
    t = 0

    for i in range(num_iterations):
        epoch_cost = 0

        for mini_batch in mini_batches:

            (mini_batch_X, mini_batch_Y) = mini_batch

            AL, cache, drop = L_model_forward(
                mini_batch_X, parameters, keep_probs)

            if regularization:
                cost = compute_cost_with_regularization(
                    AL, mini_batch_Y, parameters, lambd)
                grads = L_model_backward_with_regularization(
                    AL, mini_batch_Y, cache, lambd, parameters, drop, keep_probs)

            else:
                cost = compute_cost(AL, mini_batch_Y)
                grads = L_model_backward(
                    AL, mini_batch_Y, cache, drop, keep_probs)

            if optimizer == 'grad':
                parameters = update_parameters(
                    parameters, grads, learning_rate)

            elif optimizer == 'adam':
                t = t+1
                parameters, v, s = update_parameters_with_adam(
                    parameters, grads, v, s, t, beta1, beta2, learning_rate, epsilon)

            epoch_cost += cost / len(mini_batches)

        if i % 100 == 0:
            costs.append(epoch_cost)

        if print_cost and i % 100 == 0:
            print(f"Cost after iteration {i}: {epoch_cost}")

    toc = time.time()

    print('Training time:', round((toc-tic)/60, 2), 'minutes')

    # Plot the cost
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Every 100 Iterations')
    plt.title(f"Learning rate = {learning_rate}")
    plt.grid(True)
    plt.show()

    return parameters, keep_probs


def predict(X, Y, parameters, type, keep_probs):
    AL, _, _ = L_model_forward(X, parameters, keep_probs)
    predictions = (AL > 0.5).astype(int)

    accuracy = np.mean(predictions == Y) * 100
    print(f"{type} Accuracy: {accuracy:.2f}%")


# ----- model_v228 ------
parameters, keep_probs = model_v228(X, Y, learning_rate=0.005, layers_dims=[
    X.shape[0], 32, 16, 8, 1], num_iterations=1000, lambd=0.1, keep_probs=[1, 1, 1], mini_batch_size=256, print_cost=True, beta1=0.9, beta2=0.999, epsilon=1e-8, optimizer='grad', regularization=False)

L = len(parameters) // 2
test_keep_probs = [1.0] * (L - 1)

predict(X, Y, parameters, "Traning", test_keep_probs)
predict(X_test, Y_test, parameters, "Test", test_keep_probs)
