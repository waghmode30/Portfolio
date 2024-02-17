import numpy as np
import matplotlib.pyplot as plt

# Given data
x1 = np.array([0.5, 0.8, 0.9, 1.0, 1.1, 2.0, 2.2, 2.5, 2.8, 3.0])
x2 = np.array([0.5, 0.2, 0.9, 0.8, 0.3, 2.5, 3.5, 1.8, 2.1, 3.2])
d = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Creating Vandermonde matrix
A = np.column_stack((np.ones(len(x1)), x1, x2))
print("Vandermonde matrix:\n",A)

# Initializing weights
w = np.zeros(3)
print("Weights:\n",w)

# Logistic regression function (sigmoidal function)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Applying Gradient descent
lr = 0.01  # Learning rate
epooch = 10000

for _ in range(epooch):
    z = A @ w
    h = sigmoid(z)
    gradient = A.T @ (h - d) / len(d)
    #Gradient descent = (Vandermonde matrix_Transope(Predicted value - Desired value))/length of desired value
    w -= lr * gradient

print("Final Weights:\n",w)


# Ploting data points and decision boundary(Threshold)
plt.scatter(x1, x2, c=d, cmap=plt.cm.Paired, label='Data Points')
x_boundary = np.linspace(0, 3, 100)
y_boundary = (-w[0] - w[1] * x_boundary) / w[2]
plt.plot(x_boundary, y_boundary, color='red', label='Decision Boundary (Order 1)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Logistic Regression (Order 1)')
plt.show()

# Increasing the order of the decision boundary by 2

# Creating Vandermonde matrix
A_poly2 = np.column_stack((A, x1**2, x2**2, x1 * x2))
print("Vandermonde matrix:\n",A_poly2)

lr2 = 0.001  # Learning rate
epooch2 = 1500

# Initialize weights
w_poly2 = np.zeros(6)
print("Weights:\n",w_poly2)

for _ in range(epooch2):
    z_poly2 = A_poly2 @ w_poly2
    h_poly2 = sigmoid(z_poly2)
    gradient_poly2 = A_poly2.T @ (h_poly2 - d) / len(d)
    w_poly2 -= lr2 * gradient_poly2

print("Final Weights:\n",w_poly2)

# Plot data points and decision boundary for the second-order boundary
plt.scatter(x1, x2, c=d, cmap=plt.cm.Paired, label='Data Points')
x_boundary_poly2 = np.linspace(0, 3.0, 100)
y_boundary_poly2 = (-w_poly2[0] - w_poly2[1] * x_boundary_poly2 - w_poly2[3] * x_boundary_poly2**2 - w_poly2[5] * x_boundary_poly2**2) / w_poly2[2]
plt.plot(x_boundary_poly2, y_boundary_poly2, color='blue', label='Decision Boundary (Order 2)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Logistic Regression (Order 2)')
plt.show()

