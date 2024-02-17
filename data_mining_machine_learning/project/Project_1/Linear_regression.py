import numpy as np
import matplotlib.pyplot as plt

# Given data
d = np.array([6.0532, 7.3837, 10.0891, 11.0829, 13.2337, 12.6710, 12.7972, 11.6371])
x = np.array([1, 1.7143, 2.4286, 3.1429, 3.8571, 4.5714, 5.4857, 6])

n = len(x)
x_dash = np.mean(x)
d_dash = np.mean(d)
m = np.sum((x - x_dash) * (d - d_dash)) / np.sum((x - x_dash)**2)
b = d_dash - m * x_dash

print("Slope:",m)
print("Y-intercept:",b)

d_pred = b + m*x
mse = np.mean((d - d_pred)**2)
mse = np.round(mse,4)
print("MSE(for simple linear regression):", mse)

plt.scatter(x , d , label = 'Data Points' , color = 'orange')
plt.plot(x , d_pred , label = 'Linear Regression Line' , color = 'green')
plt.xlabel('x: input data')
plt.ylabel('d: genrated output')
plt.legend()
plt.title(f'Simple Linear Regression (MSE:{mse})')
plt.show()

A = np.column_stack((np.ones(n) , x , x**2))
print("Vandermonde matrix:\n",A)

beta = np.linalg.inv(A.T @ A) @ A.T @ d
print("Coefficients of the quadratic equation:\n", beta)

d_pred_poly2 = A @ beta
mse_poly2 = np.mean((d - d_pred_poly2)**2)
mse_poly2 = np.round(mse_poly2, 4)
print("MSE(mean squared error) for 2nd order polynomial: ", mse_poly2)

plt.scatter(x , d , label="Data Points" , color = 'orange')
plt.plot(x ,  d_pred_poly2 , label = "2nd Order Polynomial" , color = 'green')
plt.xlabel('x: input data')
plt.ylabel('d: genrated output')
plt.legend()
plt.title(f'2nd Order Polynomial Regression (MSE:{mse_poly2})')
plt.show()

A_poly6 = np.column_stack((np.ones(n) , x , x**2 , x**3 , x**4 , x**5 , x**6 ))
print("Vandermonde matrix for 6th order polynomial:\n", A_poly6)

beta_poly6 = np.linalg.inv(A_poly6.T @ A_poly6) @ A_poly6.T @ d
print("Coefficients of 6th order polynomial:\n",beta_poly6)

d_pred_poly6 = A_poly6 @ beta_poly6
mse_poly6 = np.mean((d - d_pred_poly6)**2)
mse_poly6 = np.round(mse_poly6, 4)
print("MSE for 6th order polynomial:", mse_poly6)

plt.scatter(x , d , label = "Data Points" , color = 'orange')
plt.plot(x , d_pred_poly6 , label = '6th Order Polynomial' , color='green')
plt.xlabel('x: input data')
plt.ylabel('d: genrated output')
plt.legend()
plt.title(f'6th Order Polynomial Regression (MSE:{mse_poly6})')
plt.show()

# Removing the data point
x_remove = np.delete(x , 6)
d_remove = np.delete(d , 6)

# Calculating the Vandermonde matrix when the one data point is removed
A_poly6_remove = np.column_stack((np.ones(n-1) , x_remove , x_remove**2 , x_remove**3 , x_remove**4 , x_remove**5 , x_remove**6))
print("Removed data point Vandermonde matrix:\n", A_poly6_remove)

# Calculating the coefficient when the one data point is removed
beta_poly6_remove = np.linalg.inv(A_poly6_remove.T @ A_poly6_remove) @ A_poly6_remove.T @ d_remove
print("Removed data point Coefficient:\n",beta_poly6_remove)

# Calculating the predicted values of d and mse when the data point is removed
d_pred_poly6_remove = A_poly6 @ beta_poly6_remove
mse_poly6_remove = np.mean((d_pred_poly6_remove - d)**2)
mse_poly6_remove = np.round(mse_poly6_remove)
print("MSE for 6th polynomial order when one data point is removed: ", mse_poly6_remove)

# Adding the removed data point back
x_add = np.insert(x_remove, 6, 5.2857)
d_add = np.insert(d_remove, 6, 12.7772)

# Calculating the Vandermonde matrix when the one data point is added back
A_poly6_add = np.column_stack((np.ones(n), x_add, x_add**2, x_add**3, x_add**4, x_add**5, x_add**6))
print("Added data point Vandermonde matrix:\n", A_poly6_add)

# Here we are using the the 'beta_poly6' as the calculation would be
# the same for the coefficient when the data point is added back

# Calculating the predicted values of d and mse when the data point is added back
d_pred_poly6_add = A_poly6_add @ beta_poly6
mse_poly6_add = np.mean((d_pred_poly6_add - d_add)**2)
mse_poly6_add = np.round(mse_poly6_add,4)
print("MSE for 6th polynomial order when one data point is added back: ", mse_poly6_add)

# Now to calculate the differnce between the MSE when the data was removed and added back

change_in_mse = mse_poly6_remove - mse_poly6_add
change_in_mse = np.round(change_in_mse, 4)
print("Change in mse when one data point is removed and added back: ", change_in_mse)

plt.scatter(x_add, d_add, label='Data Points' , color = 'orange')
plt.plot(x_add, d_pred_poly6_remove, label='6th Order Polynomial (with point removed)', color='green')
plt.xlabel('x')
plt.ylabel('d')
plt.legend()
plt.title(f'6th Order Polynomial Regression (MSE: {mse_poly6_remove})')
plt.show()

orders = range(1, 11)
mse_values = []

for order in orders:
    # Calculating the Vandermonde matrix
    A_order = np.column_stack([x**i for i in range(order + 1)])  # Including constant term so adding one to order

    # Calculating coefficients for the current order using the normal equation
    beta_order = np.linalg.lstsq(A_order, d, rcond=None)[0]

    # Calculating predicted values
    d_pred_order = A_order @ beta_order

    # Calculate Mean Squared Error (MSE)
    mse_order = np.mean((d_pred_order - d)**2)
    mse_values.append(mse_order)

# Ploting Cost Function (MSE) v/s polynomial order
plt.plot(orders, mse_values, marker='o', linestyle='-')
plt.xlabel('Polynomial Order')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Cost Function vs. Polynomial Order')
plt.grid(True)
plt.show()
