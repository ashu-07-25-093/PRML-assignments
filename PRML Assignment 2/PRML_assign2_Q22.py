import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# read the dataset
df = pd.read_csv('D:\Downloads\PRML Assignment 2\A2Q2Data_train.csv', header=None)

X_data = df.drop([100], axis=1)
y_act = df[100]

# converting dataframe to array
data_arr = np.array(X_data)

# calculate wml
def calculate_wml(data_arr, y_act):
    mul = np.matmul(data_arr.T, data_arr)
    pinv = np.linalg.pinv(mul)
    x_y = np.matmul(data_arr.T, y_act)
    w_star = np.matmul(pinv, x_y)

    return w_star

w_ml = calculate_wml(data_arr, y_act)

# implementation of gradient descent
def gradientDescent(X_data, y_act, learning_rate, w_ml):

    w_curr = np.random.rand(X_data.shape[1])

    iterations = []
    iteration = 0
    diff_values = []
    diff_val = float(sys.maxsize)

    while (diff_val > 0.0001):

        iteration += 1

        # gradient calculation starts
        y_pred = np.dot(X_data, w_curr)
        diff = y_pred - y_act
        dw = (np.dot(diff, X_data) * 2) / len(y_act)
        # gradient calculation ends

        w_next = w_curr - learning_rate * dw

        sub = w_next - w_ml
        diff_val = np.sqrt(np.sum(sub ** 2))

        iterations.append(iteration)
        diff_values.append(diff_val)

        diff = abs(w_next - w_curr)
        diff_val = np.sqrt(np.sum(diff ** 2))

        w_curr = w_next

    print("iterations = ", iteration)
    plt.title("iterations vs ||wt - wml|| as a function of t")
    plt.xlabel('iterations')
    plt.ylabel('||wt - wml||')
    plt.plot(iterations, diff_values)

# calling gradient descent
gradientDescent(data_arr, y_act, 0.0001, w_ml)  # curve getting bigger curvature if learning rate is more
print(w_ml)