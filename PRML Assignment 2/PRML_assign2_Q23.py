import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random

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

# performing stochastic gradient for a given batch of data
def stochasticGradientDescent(X_data, y_act, learning_rate, w_ml):

    w_curr = np.random.rand(X_data.shape[1])

    iterations = []
    iteration = 0
    diff_values = []
    diff_val = float(sys.maxsize)
    w_sgd = []

    while (diff_val > 0.0001):

        iteration += 1
        index = np.random.choice(data_arr.shape[0], 100, replace=False)

        # taking data from 100 random indexes
        X_data_t = X_data[index]
        y_act_t = y_act[index]

        # gradient calculation starts
        y_pred = np.dot(X_data_t, w_curr)
        diff = y_pred - y_act_t
        dw = (np.dot(X_data_t.T, diff) * 2) / len(y_act_t)
        # gradient calculation ends

        # updation on w
        w_next = w_curr - learning_rate * dw

        w_sgd.append(w_next)

        sub = w_next - w_ml

        diff_val = np.sqrt(np.sum(sub ** 2))

        iterations.append(iteration)
        diff_values.append(diff_val)

        diff = abs(w_next - w_curr)
        diff_val = np.sqrt(np.sum(diff ** 2))

        w_curr = w_next

        print("iteration -> ", iteration, ", diff_val -> ", diff_val)

    print("iterations = ", iteration)
    plt.title("iterations vs ||wt - wml|| as a function of t")
    plt.xlabel('iterations')
    plt.ylabel('||wt - wml||')
    plt.plot(iterations, diff_values)

    return w_sgd, iteration


w_sgd, num_of_iterations = stochasticGradientDescent(data_arr, y_act, 0.00001, w_ml)
# calculation avg of all w's to get actual w_sgd
w_sgd = np.array(w_sgd)
w_s = np.sum(w_sgd, axis=0)
w_s = w_s / num_of_iterations

# reading the test data
test_df = pd.read_csv("D:\Downloads\PRML Assignment 2\A2Q2Data_test.csv",header=None)

X_test = test_df.drop([100], axis=1)
y_test = test_df[100]

# converting test data to array
data_arr_test = np.array(X_test)

# finding test error
y_pred_test = np.dot(data_arr_test, w_s)
diff = y_pred_test - y_test

test_error = np.sqrt(np.sum(diff**2))
print(test_error)
