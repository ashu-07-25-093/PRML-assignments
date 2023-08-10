import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy import linalg
import random

# reading dataset
df = pd.read_csv('D:\Downloads\PRML Assignment 2\A2Q1.csv', header=None)

# converting dataframe to array
data_arr = np.array(df)


# assigning datapoints to random cluster and finding sum at iteration 0
def old_values(data_arr, rand, num_of_clusters):
    clusters_old = {}  # for mapping of datapoints to cluster  i.e. "x1" -> [datapoint, cluster_num]

    sum_ = {}  # for storing datapoints of same cluster

    for i in range(1, num_of_clusters + 1):
        sum_['s' + str(i)] = []

    np.random.seed(rand)

    for i in range(data_arr.shape[0]):
        c = np.random.randint(1,
                              num_of_clusters + 1)  # generating random number between 1 to 4 and assigning to data point
        clusters_old["x" + str(i + 1)] = [data_arr[i], c]
        sum_["s" + (str(c))].append(data_arr[i])

    mean_old = {}  # keeping mean of each cluster
    for i in range(1, num_of_clusters + 1):
        mean_old["u" + str(i)] = (np.sum(sum_["s" + str(i)], axis=0)) / len(
            sum_["s" + str(i)])  # finding mean of each cluster

    # sum_of_diff_old calculation
    sum_of_diff_old = 0

    for i in range(data_arr.shape[0]):  # finding distance of datapoint to it's respective mean, and then sum it
        c = clusters_old["x" + str(i + 1)][1]

        diff = clusters_old["x" + str(i + 1)][0] - mean_old["u" + str(c)]
        diff = np.dot(diff.T, diff)

        sum_of_diff_old += diff

    return sum_of_diff_old, mean_old


# from iteration 1 onwards, we will find clusters and sum of distance from mean to data points until convergence
def convergence_calculation(data_arr, sum_of_diff_new, mean_new, num_of_clusters):
    count = 1
    run_time = [1]
    error_list = [sum_of_diff_new]

    sum_of_diff_old = float(sys.maxsize)

    sum_new = {}

    # we will continue to perform reassignment till sum of distance of datapoint to it's mean is less then the previous sum
    while sum_of_diff_new < sum_of_diff_old:

        sum_of_diff_old = sum_of_diff_new

        # cluster reassignment calculation, finding distance from one datapoint to all cluster mean
        clusters_new = {}
        for i in range(data_arr.shape[0]):
            min_diff = float(sys.maxsize)
            c = 0
            for j in range(1, num_of_clusters + 1):  # upto num of clusters
                diff = (data_arr[i] - mean_new["u" + str(j)])
                diff = np.dot(diff.T, diff)
                if (diff < min_diff):
                    min_diff = diff
                    c = j
            clusters_new["x" + str(i + 1)] = [data_arr[i], c]

        for i in range(1, num_of_clusters + 1):
            sum_new['s' + str(i)] = []

        for i in range(data_arr.shape[0]):  # storing datapoints of same cluster
            c = clusters_new["x" + str(i + 1)][1]
            sum_new["s" + str(c)].append(data_arr[i])

        # new mean calculation
        for i in range(1, num_of_clusters + 1):
            mean_new["u" + str(i)] = (np.sum(sum_new["s" + str(i)], axis=0)) / len(sum_new["s" + str(i)])

        # sum of difference of data points to its own cluster mean
        sum_of_diff_new = 0

        for i in range(data_arr.shape[0]):
            c = clusters_new["x" + str(i + 1)][1]

            diff = clusters_new["x" + str(i + 1)][0] - mean_new["u" + str(c)]
            diff = np.dot(diff.T, diff)

            sum_of_diff_new += diff

        count += 1
        run_time.append(count)
        error_list.append(sum_of_diff_new)

    pi = []

    for i in range(num_of_clusters):
        pi.append(len(sum_new["s" + str(i + 1)]) / data_arr.shape[0])

    return pi, sum_new


# performing clustering operation for 5 different initialization
num_of_clusters = 4
all_mus = []     # 100 elements, one element is a list(4 elements of (50,) dim)
all_covs = []    # 100 elements, one element is a list(4 elements of (50,50) dim)
all_pis = []     # 100 elements, one element is a list(4 scaler elements)

for i in range(100):
  rand = np.random.randint(50)

  sum_of_diff_new, mean_new = old_values(data_arr, rand, num_of_clusters)

  pi, init_clusters = convergence_calculation(data_arr, sum_of_diff_new, mean_new, num_of_clusters)

  all_pis.append(pi)

  mean_vecs = []
  # mean calculation
  for key in init_clusters:
    array = np.array(init_clusters[key])
    mean_vec = np.mean(array, axis=0)
    mean_vecs.append(mean_vec)

  all_mus.append(mean_vecs)

  # covariance calculation
    # cluster calculation
  clusters = []
  for x in init_clusters:
      clusters.append(np.array(init_clusters[x]))

  covariances = [np.cov(x.T) for x in clusters]
  all_covs.append(covariances)

# calculating value of gaussian equation
def multivariateGaussian(data_vec, mean_vec, covariance_mat):

    eigen_vals = np.linalg.eigvals(covariance_mat)
    eigen_vals = [i for i in eigen_vals if abs(i) >= 1e-4]

    det = np.prod(eigen_vals)

    const = ((2 * np.pi) ** (-(len(data_vec)) / 2)) * (abs(det) ** (-1 / 2))
    pinv = linalg.pinv(covariance_mat)
    diff = data_vec - mean_vec
    pow_val = (-1 / 2) * (np.dot(np.dot(diff.T, pinv), diff))

    myValue = const * (np.exp(pow_val))
    return myValue

# calculating log-likelihood of gaussian distribution
def logLiklihood(data_arr, pi, mean_vec, covariances, numOfMixtures):
    log_liklihood = 0
    for i in range(data_arr.shape[0]):

        sum_ = 0
        for k in range(numOfMixtures):
            multivariate_gaussian = multivariateGaussian(data_arr[i], mean_vec[k], covariances[k])

            sum_ += pi[k] * multivariate_gaussian

        log_val = np.log(sum_)
        log_liklihood += log_val

    return log_liklihood


numOfMixtures = 4

final_iterations = []
final_log_liklihood_vals = []
final_diff_vals = []
final_iteration = 0
avg_log_liklihood = 0
max_log_liklihood_val = -float(sys.maxsize)
idx = 0

# iterating over 100 times
for iter in range(100):

    lambda_mat = np.zeros((data_arr.shape[0], numOfMixtures))

    new_log_liklihood = logLiklihood(data_arr, all_pis[iter], all_mus[iter], all_covs[iter], numOfMixtures)
    old_log_liklihood = float(sys.maxsize)

    iterations = []
    iteration = 0
    diff_vals = []
    log_liklihood_vals = []
    diff_val = abs(old_log_liklihood - new_log_liklihood)

    while diff_val > 0.0001:

        iteration += 1

        old_log_liklihood = new_log_liklihood

        #############################  E - step ###########################
        for i in range(data_arr.shape[0]):

            for k in range(numOfMixtures):

                multivariate_gaussian = multivariateGaussian(data_arr[i], all_mus[iter][k], all_covs[iter][k])
                lambda_mat[i][k] = all_pis[iter][k] * multivariate_gaussian

                sum_list = []
                for l in range(numOfMixtures):
                    sum_list.append(
                        all_pis[iter][l] * multivariateGaussian(data_arr[i], all_mus[iter][l], all_covs[iter][l]))

                sum_val = sum(sum_list)
                lambda_mat[i][k] /= sum_val

        #############################  M - step ###########################
        N = np.sum(lambda_mat, axis=0)

        for k in range(numOfMixtures):
            sum_val = 0
            for i in range(data_arr.shape[0]):
                sum_val += lambda_mat[i][k] * data_arr[i]

            all_mus[iter][k] = sum_val / N[k]

            sum_val = np.zeros((data_arr.shape[1], data_arr.shape[1]))
            for j in range(data_arr.shape[0]):
                diff = data_arr[j] - all_mus[iter][k]

                sum_val += (lambda_mat[j][k] * (np.dot(diff, diff.T)))

            all_covs[iter][k] = sum_val / N[k]

        for k in range(numOfMixtures):
            all_pis[iter][k] = N[k] / data_arr.shape[0]

        new_log_liklihood = logLiklihood(data_arr, all_pis[iter], all_mus[iter], all_covs[iter], numOfMixtures)
        diff_val = abs(old_log_liklihood - new_log_liklihood)
        print("new_log_liklihood", new_log_liklihood)

        iterations.append(iteration)
        diff_vals.append(diff_val)
        log_liklihood_vals.append(new_log_liklihood)

        if iter == 99:
            final_iterations.append(iteration)
            final_log_liklihood_vals.append(new_log_liklihood)
            final_diff_vals.append(diff_val)

    avg_log_liklihood += new_log_liklihood

    if new_log_liklihood > max_log_liklihood_val:
        max_log_liklihood_val = new_log_liklihood
        idx = iter

    print("iter : ", iter)

avg_log_liklihood /= 100

# plotting iterations vs log-likelihood graph
plt.xlabel("iterations")
plt.ylabel("log-liklihood")
plt.plot(final_iterations, final_log_liklihood_vals)

# plotting iterations vs error graph
plt.xlabel("iterations")
plt.ylabel("error")
plt.plot(final_iterations, final_diff_vals)

print(avg_log_liklihood)

comp_names = [f"comp{index}" for index in range(numOfMixtures)]

# prediction to which datapoint is assigned to which cluster
def predict(X, numOfMixtures):
    probas = []
    for n in range(len(X)):
        probas.append([multivariateGaussian(X[n], all_mus[idx][k], all_covs[idx][k]) for k in range(numOfMixtures)])
    cluster = []
    for proba in probas:
        cluster.append(comp_names[proba.index(max(proba))])
    return cluster

clusters = predict(data_arr,4)
print(clusters)
