import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# reading dataset file
data = pd.read_csv('Dataset.csv', names=['X', 'Y'], float_precision=None)

# plotting the data distribution by pyplot
plt.xlabel('X-coordinates')
plt.ylabel('Y-coordinates')
plt.title('Distribution of original data')
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

## Lloyds' algo functions
# assigning datapoints to random cluster and finding sum at iteration 0
def old_values(H_normalize, rand, num_of_clusters):
    clusters_old = {}  # for mapping of datapoints to cluster  i.e. "x1" -> [datapoint, cluster_num]

    sum_ = {}  # for storing datapoints of same cluster

    for i in range(1, num_of_clusters + 1):
        sum_['s' + str(i)] = []

    np.random.seed(42)

    for i in range(H_normalize.shape[0]):
        c = np.random.randint(1,
                              num_of_clusters + 1)  # generating random number between 1 to 4 and assigning to data point
        clusters_old["x" + str(i + 1)] = [H_normalize[i], c]
        sum_["s" + (str(c))].append(H_normalize[i])

    # print(len(sum_["s1"]))
    mean_old = {}
    for i in range(1, num_of_clusters + 1):
        mean_old["u" + str(i)] = (np.sum(sum_["s" + str(i)], axis=0)) / len(
            sum_["s" + str(i)])  # finding mean of each cluster

    # print(mean_old)
    # sum_of_diff_old calculation
    sum_of_diff_old = 0

    for i in range(H_normalize.shape[0]):  # finding distance of datapoint to it's respective mean, and then sum it
        c = clusters_old["x" + str(i + 1)][1]

        diff = clusters_old["x" + str(i + 1)][0] - mean_old["u" + str(c)]
        diff = np.dot(diff.T, diff)

        sum_of_diff_old += diff

    # print(sum_of_diff_old)

    return sum_of_diff_old, mean_old


# from iteration 1 onwards, we will find clusters and sum of distance from mean to data points until convergence
def convergence_calculation(H_normalize, sum_of_diff_new, mean_new, num_of_clusters, str1, hyperparam):
    count = 1
    run_time = [1]
    error_list = [sum_of_diff_new]

    sum_of_diff_old = float(sys.maxsize)

    sum_new = {}
    clusters_new = {}

    # we will continue to perform reassignment till sum of distance of datapoint to it's mean is less then the previous sum
    while sum_of_diff_new < sum_of_diff_old:

        sum_of_diff_old = sum_of_diff_new

        # cluster reassignment calculation, finding distance from one datapoint to all cluster mean
        for i in range(H_normalize.shape[0]):
            min_diff = float(sys.maxsize)
            c = 0
            for j in range(1, num_of_clusters + 1):  # upto num of clusters
                diff = (H_normalize[i] - mean_new["u" + str(j)])
                diff = np.dot(diff.T, diff)
                if (diff < min_diff):
                    min_diff = diff
                    c = j
            clusters_new["x" + str(i + 1)] = [H_normalize[i], c]

        for i in range(1, num_of_clusters + 1):
            sum_new['s' + str(i)] = []

        for i in range(H_normalize.shape[0]):  # storing datapoints of same cluster
            c = clusters_new["x" + str(i + 1)][1]
            sum_new["s" + str(c)].append(H_normalize[i])

        # new mean calculation
        for i in range(1, num_of_clusters + 1):
            mean_new["u" + str(i)] = (np.sum(sum_new["s" + str(i)], axis=0)) / len(sum_new["s" + str(i)])

        # print(mean_new)

        # sum of difference of data points to its own cluster mean
        sum_of_diff_new = 0

        for i in range(H_normalize.shape[0]):
            c = clusters_new["x" + str(i + 1)][1]

            diff = clusters_new["x" + str(i + 1)][0] - mean_new["u" + str(c)]
            diff = np.dot(diff.T, diff)

            sum_of_diff_new += diff

        count += 1
        run_time.append(count)
        error_list.append(sum_of_diff_new)

        # print("sum of diff old : ", sum_of_diff_old, "sum of diff new : ", sum_of_diff_new)
    color = ["red", "blue", "green", "purple"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.set_xlabel('X-coordinates')
    ax1.set_ylabel('Y-coordinates')
    ax1.set_title(str1+str(hyperparam))
    for i in range(len(clusters_new)):  # plotting datapoints
        ax1.scatter(data_arr[i][0], data_arr[i][1], c=color[clusters_new["x" + str(i + 1)][1] - 1])

    #     for i in range(1, num_of_clusters+1):
    #         ax1.plot(mean_new["u"+str(i)][0], mean_new["u"+str(i)][1], marker="+", color="black")

    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.title("Graph of iterations vs error")
    ax2.plot(run_time, error_list)
    plt.show()

num_of_clusters = 4
rand = np.random.randint(50)
lst = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
# # By polynomial kernel
str1 = "Spectral clustering by polynomial kernel for degree "
print("By polynomial kernel")
for i in range(2,4):
    print("for drgree : ",i)
    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = poly_kernel(data_arr, i, 4)
    top_eigen_vecs = np.array(top_eigen_vecs)
    top_eigen_vecs = top_eigen_vecs.T
    H_normalize = calculating_HNormalize(top_eigen_vecs)
    sum_of_diff_new, mean_new = old_values(H_normalize, rand, num_of_clusters)
    convergence_calculation(H_normalize, sum_of_diff_new, mean_new, num_of_clusters, str1, i)


# By Radian Basis kernel
## calling for different values of sigma
str1 = "Spectral clustering by Radial Basis kernel for sigma "
print("By Radial Basis kernel")
for i in range(len(lst)):
    print("for sigma : ", lst[i])

    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = radial_basis(data_arr, lst[i], 4)
    top_eigen_vecs = np.array(top_eigen_vecs)
    top_eigen_vecs = top_eigen_vecs.T
    H_normalize = calculating_HNormalize(top_eigen_vecs)

    sum_of_diff_new, mean_new = old_values(H_normalize, rand, num_of_clusters)
    convergence_calculation(H_normalize, sum_of_diff_new, mean_new, num_of_clusters, str1, lst[i])

# By Gaussian kernel
## calling for different values of sigma
str1 = "Spectral clustering by Gaussian kernel for sigma "
print("By Gaussian kernel")
for i in range(len(lst)):
    print("for sigma : ", lst[i])

    top_eigen_val_ind, top_eigen_vals, top_eigen_vecs = gaussian_kernel(data_arr, lst[i], 4)
    top_eigen_vecs = np.array(top_eigen_vecs)
    top_eigen_vecs = top_eigen_vecs.T
    H_normalize = calculating_HNormalize(top_eigen_vecs)

    sum_of_diff_new, mean_new = old_values(H_normalize, rand, num_of_clusters)
    convergence_calculation(H_normalize, sum_of_diff_new, mean_new, num_of_clusters, str1, lst[i])
