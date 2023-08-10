import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import random

# reading the dataset
df = pd.read_csv('D:\Downloads\PRML Assignment 2\A2Q2Data_train.csv', header=None)

X_data = df.drop([100], axis=1)
y_act = df[100]

# converting dataframe to array
data_arr = np.array(X_data)

def get_wml(data_arr, y_act):
    mul = np.matmul(data_arr.T, data_arr)
    pinv = np.linalg.pinv(mul)
    x_y = np.matmul(data_arr.T, y_act)
    w_star = np.matmul(pinv, x_y)

    return w_star

w_ml = get_wml(data_arr, y_act)

# finding training error
y_pred = np.dot(data_arr, w_ml)

diff = y_pred - y_act

error_val = np.sqrt(np.sum(diff**2))
print("w_ml training error : ", error_val)

# performing ridge gradient on a given fold of data
def ridgeGradient(X_data, y_act, learning_rate, lambda_):

    w_curr = np.random.rand(X_data.shape[1])

    iterations = []
    iteration = 0
    diff_values = []
    diff_val = float(sys.maxsize)

    while (diff_val > 0.0001):

        iteration += 1
        y_pred = np.dot(X_data, w_curr)

        diff = y_pred - y_act

        dw = (np.dot(X_data.T, diff) * 2 + 2 * w_curr * lambda_) / len(y_act)
        w_next = w_curr - learning_rate * dw

        sub = abs(w_next - w_curr)
        diff_val = np.sqrt(np.sum(sub ** 2))
        w_curr = w_next

        iterations.append(iteration)
        diff_values.append(diff_val)

    return w_curr

# prediction on new data
def predictionOnNew(data_arr_validation, y_act_validation, w_train):

  y_pred = np.dot(data_arr_validation, w_train)

  diff = y_pred - y_act_validation

  error_val = np.sqrt(np.sum(diff**2))

  return error_val

# taking values of lambda between 0 to 5
lambdas = []
for i in range(5):
    lambdas.append(round(random.uniform(0, 5), 2))

# taking different values of fold
k_folds = [3, 5, 7, 10]
error_list = []
best_k_fold_list = []

# performing k-fold cross validation for different values of lambda
for lambda_ in lambdas:

    min_error = float(sys.maxsize)
    best_k_fold = 0
    for k in k_folds:

        num_of_training_samples = (int)(data_arr.shape[0] / k) * (k - 1)
        num_of_validation_samples = data_arr.shape[0] - num_of_training_samples

        index = np.random.choice(data_arr.shape[0], num_of_training_samples, replace=False)

        data_arr_train = data_arr[index]
        y_act_train = y_act[index]

        validation_index = []
        for i in range(data_arr.shape[0]):
            if i not in index:
                validation_index.append(i)

        random.shuffle(validation_index)
        data_arr_validation = data_arr[validation_index]
        y_act_validation = y_act[validation_index]

        w_train = ridgeGradient(data_arr_train, y_act_train, 0.002, lambda_)

        validation_error = predictionOnNew(data_arr_validation, y_act_validation, w_train)

        if validation_error < min_error:
            min_error = validation_error
            best_k_fold = k

    error_list.append(min_error)      # storing min validation error for a lambda
    best_k_fold_list.append(best_k_fold)    # storing the best k-fold for a lambda

# plotting
plt.title("Graph lambda vs validation-error")
plt.xlabel("lambda")
plt.ylabel("validation-error")
plt.scatter(lambdas, error_list)

# reading test error
df = pd.read_csv('D:\Downloads\PRML Assignment 2\A2Q2Data_test.csv', header=None)

data_test = df.drop([100], axis=1)
y_test_act = df[100]

data_arr_test = np.array(data_test)

# finding prediction on test data
test_error_wml = predictionOnNew(data_arr_test, y_test_act, w_ml)

# finding test errors for lambda with best k-fold corresponding to that lambda
for lambda_ in lambdas:
    wr = ridgeGradient(data_arr, y_act, 0.002, lambda_)
    test_error_wr = predictionOnNew(data_arr_test, y_test_act, wr)

    print("wr_test_error : ", test_error_wr, " and wml_test_error : ", test_error_wml)
