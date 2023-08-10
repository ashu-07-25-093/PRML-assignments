import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

# read the dataset
df = pd.read_csv('A2Q1.csv', header=None)

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

    # plotting iterationg vs error graph
    plt.title("Graph of iterations vs error")
    plt.xlabel('iterations')
    plt.ylabel('error')

    plt.plot(run_time, error_list)  # plotting iteration vs error
    plt.show()

    return sum_new


num_of_clusters = 4

rand = np.random.randint(100)

sum_of_diff_new, mean_new = old_values(data_arr, rand, num_of_clusters)

clusters = convergence_calculation(data_arr, sum_of_diff_new, mean_new, num_of_clusters)

for i in range(4):
    print(len(clusters['s'+str(i+1)]))