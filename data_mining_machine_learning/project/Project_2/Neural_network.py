import numpy as np
import scipy.io

import scipy.io
data = scipy.io.loadmat('W:\Fau\data_min_mach_learn\project\Project_2\digits.mat')

# Extract variables
train_data = data['train']
train_labels = data['trainlabels']
test_data = data['test']
test_labels = data['testlabels']

# Normalizing pixel values
train_data_normalized = train_data / 255
test_data_normalized = test_data / 255

num_classes = 10
train_labels_onehot = np.zeros((num_classes, len(train_labels)))
for i in range(len(train_labels)):
    train_labels_onehot[train_labels[i], i] = 1

# Definin Hyperparameters
input_size = 784
hidden_size = 25
output_size = num_classes
learning_rate = 0.01
num_epochs = 100

# Initializing weights and biases
weights_input_hidden = np.random.randn(hidden_size, input_size)
biases_input_hidden = np.random.randn(hidden_size, 1)
weights_hidden_output = np.random.randn(output_size, hidden_size)
biases_hidden_output = np.random.randn(output_size, 1)

# Sigmoid function
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))

for epoch in range(num_epochs):
    epoch_errors = []
    num_correct = 0
    for sample in range(5000):
        # Forward propagation
        x = train_data_normalized[:, sample].reshape(-1, 1)
        hidden_layer_input = np.dot(weights_input_hidden, x) + biases_input_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        output_layer_input = np.dot(weights_hidden_output, hidden_layer_output) + biases_hidden_output
        output_layer_output = sigmoid(output_layer_input)
        output_error = output_layer_output - train_labels_onehot[:, sample].reshape(-1, 1)
        hidden_error = np.dot(weights_hidden_output.T, output_error) * (hidden_layer_output * (1 - hidden_layer_output))

        # Backpropagation
        weights_hidden_output -= learning_rate * np.dot(output_error, hidden_layer_output.T)
        biases_hidden_output -= learning_rate * output_error
        weights_input_hidden -= learning_rate * np.dot(hidden_error, x.T)
        biases_input_hidden -= learning_rate * hidden_error

        # Calculating epoch error (MSE)
        epoch_errors.append(np.mean(output_error ** 2))

        # Calculating accuracy
        predicted_class = np.argmax(output_layer_output)
        true_class = np.argmax(train_labels_onehot[:, sample])
        if predicted_class == true_class:
            num_correct += 1

    epoch_accuracy = num_correct / 5000
    mean_epoch_error = np.mean(epoch_errors)
    print(f'Epoch {epoch + 1}: MSE={mean_epoch_error:.4f}, Accuracy={epoch_accuracy * 100:.2f}%')

# Testing
test_labels_onehot = np.zeros((num_classes, len(test_labels)))
for i in range(len(test_labels)):
    test_labels_onehot[test_labels[i], i] = 1

num_correct = 0
for sample in range(1000):  # Assuming 'test' has 1000 examples
    x_test = test_data_normalized[:, sample].reshape(-1, 1)
    hidden_layer_input_test = np.dot(weights_input_hidden, x_test) + biases_input_hidden
    hidden_layer_output_test = sigmoid(hidden_layer_input_test)
    output_layer_input_test = np.dot(weights_hidden_output, hidden_layer_output_test) + biases_hidden_output
    output_layer_output_test = sigmoid(output_layer_input_test)

    predicted_class = np.argmax(output_layer_output_test)
    true_class = np.argmax(test_labels_onehot[:, sample])

    if predicted_class == true_class:
        num_correct += 1

# Calculating accuracy
accuracy = num_correct / 1000
print(f'Testing Accuracy: {accuracy * 100:.2f}%')
