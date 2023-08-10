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

# finding mean (mean of each column and subtract mean from data points)
mean = data_arr.mean(axis=0)
data_arr = data_arr - mean

## Clustering functions
def dis(x1,y1,x2,y2):
    return ((x1-x2)**2 + (y1-y2)**2)

# assigning datapoints to random cluster and finding sum at iteration 0
def old_values(data_arr, rand, num_of_clusters):
    clusters_old = {}  # for mapping of datapoints to cluster  i.e. "x1" -> [datapoint, cluster_num]

    sum_ = {}  # for storing datapoints of same cluster

    for i in range(1, num_of_clusters + 1):
        sum_['s' + str(i)] = []

    np.random.seed(42)

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
# from iteration 1 onwards, we will find clusters and sum of distance from mean to data points until convergence
def convergence_calculation(data_arr, sum_of_diff_new, mean_new, num_of_clusters):
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

        # print(mean_new)

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

        # print("sum of diff old : ", sum_of_diff_old, "sum of diff new : ", sum_of_diff_new)

    # print(mean_new)
    color = ["red", "blue", "green", "purple", "yellow"]
    for i in np.arange(-10.0, 10.5, 0.5):
        for j in np.arange(-10.0, 10.5, 0.5):
            c = 0
            min_ = float(sys.maxsize)
            for k in range(1, num_of_clusters+1):
                dist = (i-mean_new["u" + str(k)][0])**2 + (j-mean_new["u" + str(k)][1])**2
                if min_ > dist:
                    min_ = dist
                    c = k
            plt.scatter(i, j, c=color[c-1])

    color = ["red", "blue", "green", "purple", "yellow"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    ax1.set_xlabel('X-coordinates')
    ax1.set_ylabel('Y-coordinates')
    str1 = "for number of clusters " + str(num_of_clusters)
    ax1.set_title('Clustered dataset, ' + str1)

    # plt.title('cluster')
    for i in range(len(clusters_new)):  # plotting datapoints
        ax1.scatter(clusters_new["x" + str(i + 1)][0][0], clusters_new["x" + str(i + 1)][0][1],
                    c=color[clusters_new["x" + str(i + 1)][1] - 1])

    for i in range(1, num_of_clusters + 1):  # plotting mean of each cluster
        ax1.plot(mean_new["u" + str(i)][0], mean_new["u" + str(i)][1], marker="+", color="black")

    plt.xlabel('iterations')
    plt.ylabel('error')
    plt.title('Graph of iterations vs error, ' + str1)

    ax2.plot(run_time, error_list)  # plotting iteration vs error
    plt.show()


## calling for 2,3,4 and 5 clusters
rand = np.random.randint(7)
for num_of_clusters in range(2, 6):
    print("for number of clusters : ", num_of_clusters)

    sum_of_diff_new, mean_new = old_values(data_arr, rand, num_of_clusters)
    # print(mean_new)
    convergence_calculation(data_arr, sum_of_diff_new, mean_new, num_of_clusters)
