#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Loading the training data through drive

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import pandas as pd

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train.shape, type(x_train) ,y_train.shape , type(y_train)

# Splitting the data in two havles real and fake

x_train_split = x_train.shape[0] // 2
x_train_real = x_train[x_train_split:]
x_train_fake = x_train[:x_train_split]
print("X train real: ", x_train_real.shape)
print("X train fake: ", x_train_fake.shape)

y_train_real = y_train[x_train_split:]
y_train_fake = y_train[:x_train_split]
print("Y train real: ", y_train_real.shape)
print("Y train fake: ", y_train_fake.shape)

# Shuffling the y_train_fake labels to create fake data
np.random.shuffle(y_train_fake)

# Encoding the real labels to real data

import numpy as np

output_size = 10

def one_hot_encode(y_train_real):
    encoded_labels = np.random.randint(50, 201, size=(y_train_real.shape[0], output_size))
    for i in range(y_train_real.shape[0]):
        encoded_labels[i, y_train_real[i]] = 255
        # print(encoded_labels[0])
    return encoded_labels


print((one_hot_encode(y_train_real).shape))

y_train_onehot_real = one_hot_encode(y_train_real)


# Loop through each sample in x_train_real
for i in range(x_train_real.shape[0]):
    # Assign the one-hot encoded label from y_train_onehot_real to the first 10 elements of the first row of x_train_real
    x_train_real[i, 0, :10] = y_train_onehot_real[i]

print(x_train_real[0])

import matplotlib.pyplot as plt

plt.imshow(x_train_real[3000], cmap='gray')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Encoding the fake labels to fake data

output_size = 10

def one_hot_encode(y_train_fake):
    encoded_labels = np.random.randint(50, 201, size=(y_train_fake.shape[0], output_size))
    for i in range(y_train_fake.shape[0]):
        encoded_labels[i, y_train_fake[i]] = 255
        # print(encoded_labels[0])
    return encoded_labels


print((one_hot_encode(y_train_fake).shape))

y_train_onehot_fake = one_hot_encode(y_train_fake)

# Loop through each sample in x_train_real
for i in range(x_train_fake.shape[0]):
    # Assign the one-hot encoded label from y_train_onehot_fake to the first 10 elements of the first row of x_train_fake
    x_train_fake[i, 0, :10] = y_train_onehot_fake[i]

print(x_train_fake[0])

plt.imshow(x_train_fake[12313], cmap='gray')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Normalizing the data

x_train_real_norm = x_train_real/255
x_train_fake_norm = x_train_fake/255
y_train_norm = y_train/255

x_train_real_norm.shape , x_train_fake_norm.shape

x_train_rf_norm = np.concatenate((x_train_real,x_train_fake), axis=0 )
x_train_rf_norm.shape

# Defining the parameters and forward propogation

# Definin Hyperparameters
input_size = 784
hidden_size_one = 2000
hidden_size_two = 2000
output_size = 10
learning_rate = 0.01
num_epochs = 500

# Initialzae the weights
np.random.seed(36)
weights_input_hidden_1 = np.random.randn(hidden_size_one, input_size)
bais_input_hidden_1 = np.random.randn(hidden_size_one,1)
weights_hidden_1_hidden_2 = np.random.randn(hidden_size_two,hidden_size_one)
bais_hidden_1_hidden_2 = np.random.randn(hidden_size_two,1)
weights_hidden_2_output = np.random.randn(output_size,hidden_size_two)
bais_hidden_2_output = np.random.randn(output_size,1)

# Sigmoid function
# sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

import numpy as np
import matplotlib.pyplot as plt

# Define the sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the softmax function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Define the cross-entropy loss function
def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
    loss = np.sum(log_likelihood) / m
    return loss

# Define the forward pass function
def forward(x, w1, b1, w2, b2, w3, b3):
    z1 = np.dot(x, w1.T) + b1.T
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2.T) + b2.T
    a2 = sigmoid(z2)
    z3 = np.dot(a2, w3.T) + b3.T
    a3 = softmax(z3)
    return a3, a2, a1

# Define the backward pass function
def backward(x, y, w1, b1, w2, b2, w3, b3, a1, a2, a3, lr):
    m = x.shape[0]
    delta3 = a3 - y
    dw3 = np.dot(delta3.T, a2) / m
    db3 = np.sum(delta3, axis=0, keepdims=True).T / m
    delta2 = np.dot(delta3, w3) * a2 * (1 - a2)
    dw2 = np.dot(delta2.T, a1) / m
    db2 = np.sum(delta2, axis=0, keepdims=True).T / m
    delta1 = np.dot(delta2, w2) * a1 * (1 - a1)
    dw1 = np.dot(delta1.T, x) / m
    db1 = np.sum(delta1, axis=0, keepdims=True).T / m

    # Update weights and biases
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    w3 -= lr * dw3
    b3 -= lr * db3

    return w1, b1, w2, b2, w3, b3

# Train the neural network
def train(x_train, y_train, num_epochs, lr):
    # Initialize weights and biases
    input_size = x_train.shape[1]
    hidden_size_one = 2000
    hidden_size_two = 2000
    output_size = 10
    np.random.seed(36)
    w1 = np.random.randn(hidden_size_one, input_size)
    b1 = np.random.randn(hidden_size_one, 1)
    w2 = np.random.randn(hidden_size_two, hidden_size_one)
    b2 = np.random.randn(hidden_size_two, 1)
    w3 = np.random.randn(output_size, hidden_size_two)
    b3 = np.random.randn(output_size, 1)

    # Lists to store losses
    train_losses = []

    for epoch in range(num_epochs):
        # Forward pass
        a3, a2, a1 = forward(x_train, w1, b1, w2, b2, w3, b3)

        # Calculate loss
        loss = cross_entropy_loss(y_train, a3)
        train_losses.append(loss)

        # Backward pass
        w1, b1, w2, b2, w3, b3 = backward(x_train, y_train, w1, b1, w2, b2, w3, b3, a1, a2, a3, lr)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

    return w1, b1, w2, b2, w3, b3, train_losses

# Prepare the data
x_train_rf_norm = np.concatenate((x_train_real_norm, x_train_fake_norm), axis=0)
y_train_rf = np.concatenate((y_train_real, y_train_fake), axis=0)

# Reshape and flatten the input data
x_train_rf_norm = x_train_rf_norm.reshape(x_train_rf_norm.shape[0], -1)

# One-hot encode the target labels
def one_hot_encode(y, output_size):
    m = len(y)
    y_onehot = np.zeros((m, output_size))
    y_onehot[np.arange(m), y] = 1
    return y_onehot

y_train_onehot = one_hot_encode(y_train_rf, output_size)

# Train the neural network
num_epochs = 500
lr = 0.01
w1, b1, w2, b2, w3, b3, train_losses = train(x_train_rf_norm, y_train_onehot, num_epochs, lr)

# Plot the training loss
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Test the neural network
def predict(x, w1, b1, w2, b2, w3, b3):
    a3, _, _ = forward(x, w1, b1, w2, b2, w3, b3)
    return np.argmax(a3, axis=1)

# Prepare test data
x_test = x_test.reshape(x_test.shape[0], -1)
y_test_onehot = one_hot_encode(y_test, output_size)

# Make predictions
y_pred = predict(x_test, w1, b1, w2, b2, w3, b3)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print("Test Accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:




