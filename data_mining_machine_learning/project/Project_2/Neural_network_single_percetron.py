import scipy.io
data = scipy.io.loadmat('W:\Fau\data_min_mach_learn\project\Project_2\digits.mat')

import numpy as np

# Reshaping the training and testing data
train_images = data['train'].reshape(28, 28, -1, order='F')
test_images = data['test'].reshape(28, 28, -1, order='F')

# Checking the shape of reshaped training and test images
train_images.shape, test_images.shape

def initialize_weights(input_dim):
    # Initialize the weights for the perceptron.
    np.random.seed(42)  # For reproducibility
    return np.random.randn(input_dim, 1) * 0.01

def preprocess_labels(labels, target_digit):
    # Preprocess labels to +1 for target digit and -1 for others.
    return np.where(labels == target_digit, 1, -1)

def train_perceptron(X, y, learning_rate, epochs, report_every):
    # Train a single-layer perceptron.
    weights = initialize_weights(X.shape[0])  # Initialize weights
    n_samples = X.shape[1]
    error_history = []

    for epoch in range(epochs):
        total_error = 0
        for i in range(n_samples):
            xi = X[:, i:i+1]
            yi = y[i]
            output = np.sign(np.dot(weights.T, xi))
            update = learning_rate * (yi - output) * xi
            weights += update
            total_error += int(output != yi)

        if (epoch + 1) % report_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs}, Error: {total_error}")
            error_history.append(total_error)
            # Adjust learning rate
            learning_rate *= 0.9

    return weights, error_history


# Prepare data
X_train = train_images.reshape(-1, train_images.shape[2], order='F')
y_train = preprocess_labels(data['trainlabels'], 0)  # Preprocess labels for digit '0'
X_test = test_images.reshape(-1, test_images.shape[2], order='F')
y_test = preprocess_labels(data['testlabels'], 0)  # Preprocess labels for digit '0'

# Train the perceptron
weights, error_history = train_perceptron(X_train, y_train, learning_rate=1e-2, epochs=100, report_every=1)

print("Shape of weights for zero:",weights.shape)
print("Error history for dectecting 0:",error_history)

def evaluate_perceptron(X, y, weights):
    # Evaluate the perceptron model on a given dataset.
    outputs = np.sign(np.dot(weights.T, X))
    accuracy = np.mean(outputs == y) * 100
    return accuracy

# Evaluate the perceptron on the test set
test_accuracy = evaluate_perceptron(X_test, y_test, weights)

print("Test accuracy for dectecting 0:",test_accuracy)

import matplotlib.pyplot as plt


# Reshaping the weights to visualize them as an image
weights_image = weights.reshape(28, 28)

plt.figure(figsize=(6, 6))
plt.imshow(weights_image, cmap='gray')
plt.colorbar()
plt.title('Visualization of the Perceptron Weights')
plt.show()

# Prepare data
X_train_8 = train_images.reshape(-1, train_images.shape[2], order='F')
y_train_8 = preprocess_labels(data['trainlabels'], 8)  # Preprocess labels for digit '8'
X_test_8 = test_images.reshape(-1, test_images.shape[2], order='F')
y_test_8 = preprocess_labels(data['testlabels'], 8)  # Preprocess labels for digit '8'

# Train the perceptron
weights, error_history = train_perceptron(X_train_8, y_train_8, learning_rate=0.01, epochs=100, report_every=1)

print("Shape of weights for Eight:",weights.shape)
print("Error history for dectecting 8:",error_history)

def evaluate_perceptron(X, y, weights):
    # Evaluate the perceptron model on a given dataset.
    outputs = np.sign(np.dot(weights.T, X))
    accuracy = np.mean(outputs == y) * 100
    return accuracy

# Evaluate the perceptron on the test set
test_accuracy = evaluate_perceptron(X_test_8, y_test_8, weights)

print("Test accuracy for dectecting 8:",test_accuracy)

import matplotlib.pyplot as plt


# Reshaping the weights to visualize them as an image
weights_image = weights.reshape(28, 28)

plt.figure(figsize=(6, 6))
plt.imshow(weights_image, cmap='gray')
plt.colorbar()
plt.title('Visualization of the Perceptron Weights')
plt.show()

# Prepare data
X_train = train_images.reshape(-1, train_images.shape[2], order='F')
y_train = preprocess_labels(data['trainlabels'], 1)  # Preprocess labels for digit '1'
X_test = test_images.reshape(-1, test_images.shape[2], order='F')
y_test = preprocess_labels(data['testlabels'], 1)  # Preprocess labels for digit '1'

# Train the perceptron
weights, error_history = train_perceptron(X_train, y_train, learning_rate=0.01, epochs=100, report_every=1)

print("Shape of weights for One:",weights.shape)
print("Error history for dectecting 1:",error_history)

def evaluate_perceptron(X, y, weights):
    # Evaluate the perceptron model on a given dataset.
    outputs = np.sign(np.dot(weights.T, X))
    accuracy = np.mean(outputs == y) * 100
    return accuracy

# Evaluate the perceptron on the test set
test_accuracy = evaluate_perceptron(X_test, y_test, weights)

print("Test accuracy for dectecting 1:",test_accuracy)

import matplotlib.pyplot as plt


# Reshaping the weights to visualize them as an image
weights_image = weights.reshape(28, 28)  # Use all weights

plt.figure(figsize=(6, 6))
plt.imshow(weights_image, cmap='gray')
plt.colorbar()
plt.title('Visualization of the Perceptron Weights')
plt.show()


# Prepare data
X_train = train_images.reshape(-1, train_images.shape[2], order='F')
y_train = preprocess_labels(data['trainlabels'], 2)  # Preprocess labels for digit '2'
X_test = test_images.reshape(-1, test_images.shape[2], order='F')
y_test = preprocess_labels(data['testlabels'], 2)  # Preprocess labels for digit '2'

# Train the perceptron
weights, error_history = train_perceptron(X_train, y_train, learning_rate=0.01, epochs=100, report_every=1)

print("Shape of weights for Two:",weights.shape)
print("Error history for dectecting 2:",error_history)

def evaluate_perceptron(X, y, weights):
    # Evaluate the perceptron model on a given dataset.
    outputs = np.sign(np.dot(weights.T, X))
    accuracy = np.mean(outputs == y) * 100
    return accuracy

# Evaluate the perceptron on the test set
test_accuracy = evaluate_perceptron(X_test, y_test, weights)

print("Test accuracy for dectecting 2:",test_accuracy)

import matplotlib.pyplot as plt


# Correctly reshape the weights to visualize them as an image
weights_image = weights.reshape(28, 28)  # Use all weights

plt.figure(figsize=(6, 6))
plt.imshow(weights_image, cmap='gray')
plt.colorbar()
plt.title('Visualization of the Perceptron Weights')
plt.show()


