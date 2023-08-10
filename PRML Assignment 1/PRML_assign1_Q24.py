import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# reading dataset file
data = pd.read_csv('Dataset.csv', names=['X', 'Y'], float_precision=None)

# plotting the data distribution by pyplot
plt.xlabel('X-coordinates')
plt.ylabel('Y-coordinates')
plt.title('Data distribution of original data by scatter plot')
plt.plot(data['X'], data['Y'], 'g+')

# converting dataframe to np array
data_arr = np.array(data)
print("data in array form : ",data_arr)

## Kernel Centering
def find_center(P):

    I = np.identity(P.shape[0])
    iden_min_ones = I - (1 / P.shape[0])
    intermediate = np.matmul(iden_min_ones, P)
    P_centered = np.matmul(intermediate, iden_min_ones)
    # print(P_centered)
    # print(np.allclose(P_centered, P_centered.T))
    return P_centered

## Computing eigen values and eigen vectors
def center_and_eigen(P, k):

    P_centered = find_center(P)
    eigen_vals, beta_eigen_vecs = np.linalg.eigh(P_centered)
    # print(eigen_vals)
    top_eigen_val_ind = eigen_vals.argsort()[-k:][::-1]

    top_eigen_vals = []
    for i in range(0, k):
        top_eigen_vals.append(eigen_vals[top_eigen_val_ind[i]])

    top_eigen_vecs = []
    for i in range(0, k):
        top_eigen_vecs.append(beta_eigen_vecs[:, top_eigen_val_ind[i]])

    return top_eigen_val_ind, top_eigen_vals, top_eigen_vecs

## Polynomial Kernel
def poly_kernel(data_arr, degree, k):
    P = np.dot(data_arr, data_arr.T)
    P = P + 1
    P = np.power(P, degree)

    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = center_and_eigen(P, k)

    return top_eigen_val_ind, top_eigen_vals, top_eigen_vecs

## Radial Basis Kernel
def radial_basis(data_arr, sigma, k):
    P_k = np.zeros((data_arr.shape[0], data_arr.shape[0]))
    sig_sq = 2 * ((sigma) ** 2)

    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[0]):
            diff = np.subtract(data_arr[i], data_arr[j])

            sum_ = 0
            for r in range(len(diff)):
                sum_ += diff[r] ** 2

            sum_ = -sum_
            div = sum_ / sig_sq
            P_k[i][j] = np.exp(div)
    # print(P_k)

    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = center_and_eigen(P_k, k)

    return top_eigen_val_ind, top_eigen_vals, top_eigen_vecs

## Gaussian Kernel
def gaussian_kernel(data_arr, sigma, k):
    P_k = np.zeros((data_arr.shape[0], data_arr.shape[0]))
    sig_sq = 2 * ((sigma) ** 2)

    # mean_ = np.mean(data_arr, axis=0)

    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[0]):
            diff = np.subtract(data_arr[i], data_arr[j])
            diff = np.dot(diff.T, diff)
            diff = -(diff * sigma)
            P_k[i][j] = np.exp(diff)

    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = center_and_eigen(P_k, k)

    return top_eigen_val_ind, top_eigen_vals, top_eigen_vecs

## Calculating H_Normalize
def calculating_HNormalize(top_eigen_vecs):

    H_normalize = np.ndarray((1000, 4))

    for i in range(0, len(top_eigen_vecs)):  # Normalize the row
        sum_ = 0
        for j in range(0, len(top_eigen_vecs[0])):
            sum_ += top_eigen_vecs[i][j] ** 2
        sum_ = sum_ ** (1 / 2)
        for j in range(0, len(top_eigen_vecs[0])):
            H_normalize[i][j] = top_eigen_vecs[i][j] / sum_

    return H_normalize

def cluster_assignment(H_normalize, str1, hyperparam):

    # cluster assignment according to the maximum value in a row of H_Normalize
    clusters = {}
    for i in range(H_normalize.shape[0]):
        clusters["x" + str(i + 1)] = [data_arr[i], np.argmax(H_normalize[i]) + 1]

    # plotting the datapoints
    color = ["red", "blue", "green", "purple"]
    str1 = str1 + str(hyperparam)
    fig, ax = plt.subplots(1,1, figsize=(6,6))
    ax.set_xlabel('X-coordinates')
    ax.set_ylabel('Y-coordinates')
    ax.set_title(f'Cluster assignment by taking maximum value in a row of H_Normalize \n {str1}')

    for i in range(len(clusters)):
        ax.scatter(clusters["x" + str(i + 1)][0][0], clusters["x" + str(i + 1)][0][1],
                    c=color[clusters["x" + str(i + 1)][1] - 1])

    plt.show()

num_of_clusters = 4
rand = np.random.randint(50)
lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]

# By polynomial kernel
str1 = "By polynomial kernel for degree "
print("By polynomial kernel")
for i in range(2,4):
    print("for drgree : ",i)
    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = poly_kernel(data_arr, i, 4)
    top_eigen_vecs = np.array(top_eigen_vecs)
    top_eigen_vecs = top_eigen_vecs.T
    H_normalize = calculating_HNormalize(top_eigen_vecs)

    cluster_assignment(H_normalize,str1, i)

# By Radian Basis kernel
## calling for different values of sigma
str1 = "By Radial Basis kernel for sigma "
print("By Radial Basis kernel")
for i in range(len(lst)):
    print("for sigma : ", lst[i])

    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = radial_basis(data_arr, lst[i], 4)
    top_eigen_vecs = np.array(top_eigen_vecs)
    top_eigen_vecs = top_eigen_vecs.T
    H_normalize = calculating_HNormalize(top_eigen_vecs)

    cluster_assignment(H_normalize, str1, lst[i])

# By Gaussian kernel
## calling for different values of sigma
str1 = "By Gaussian kernel for sigma "
print("By Gaussian kernel")
for i in range(len(lst)):
    print("for sigma : ", lst[i])

    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = gaussian_kernel(data_arr, lst[i], 4)
    top_eigen_vecs = np.array(top_eigen_vecs)
    top_eigen_vecs = top_eigen_vecs.T
    H_normalize = calculating_HNormalize(top_eigen_vecs)

    cluster_assignment(H_normalize, str1, lst[i])


