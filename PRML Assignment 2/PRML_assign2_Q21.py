import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# read the dataset
df = pd.read_csv('D:\Downloads\PRML Assignment 2\A2Q2Data_train.csv', header=None)

X_data = df.drop([100], axis=1)
y = df[100]

# converting dataframe to array
data_arr = np.array(X_data)

mul = np.matmul(data_arr.T, data_arr)
pinv = np.linalg.pinv(mul)
x_y = np.matmul(data_arr.T, y)
w_star = np.matmul(pinv, x_y)

# finding training error
y_pred = np.dot(data_arr, w_star)
diff = y_pred - y

error_val = np.sum(diff**2)

print(error_val)
print(np.sqrt(error_val))