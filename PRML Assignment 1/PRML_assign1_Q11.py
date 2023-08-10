import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Dataset.csv', names = ['X', 'Y'])

# # plotting the data distribution by seaborn
# ans = sns.displot(data, x="X", y="Y", kind="kde", bw_adjust=2)
# ans.fig.suptitle("Distribution of original Data")

# plotting the data distribution by pyplot
plt.xlabel('X-coordinates')
plt.ylabel('Y-coordinates')
plt.title('Distribution of original Data')
plt.plot(data['X'], data['Y'], 'g+')

# converting dataframe to np array
data_arr = np.array(data)
print("data in array form : ",data_arr)

# pca function to find principle components
def pca(data_):

    P = data_.T @ data_ / len(data_)  # covariance matrix 2x2

    eigen_vals, eigen_vecs = np.linalg.eig(P)  # finding eigen values and eigen vectors

    #print(eigen_vecs)
    #print(eigen_vals)
    #print(np.sum(eigen_vals))

    print("max eigen value is ", max(eigen_vals))
    # print(eigen_vals[1])
    print("variance explained by eigen vector corresponding to eigen value ", max(eigen_vals), " is : ",
          format(np.divide(eigen_vals[1], np.sum(eigen_vals)) * 100), "%")

    print("second max eigen valus is ", eigen_vals[0])
    print("variance explained by eigen vector corresponding to eigen value ", eigen_vals[0], " is : ",
          format(np.divide(eigen_vals[0], np.sum(eigen_vals)) * 100), "%")

    # origin = [0,0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.set_xlabel('X-coordinates')
    ax.set_ylabel('Y-coordinates')
    x = np.linspace(0, 20, 1000)
    ax.plot(data['X'], data['Y'], 'g+')
    #     plt.quiver([0,0], [0,0], eigen_vecs[:,0], eigen_vecs[:, 1], scale=3 ,color=['red', 'blue'])
    color = ['red', 'blue']
    # for i  in range(len(eigen_vecs)):
    ax.axline([0, 0], eigen_vecs[:, 1], color=color[1], label=f"Principle Compenent 1")
    ax.axline([0, 0], eigen_vecs[:, 0], color=color[0], label=f"Principle Compenent 2")
    ax.set_title('Principle components for centered dataset')

    plt.legend()
    #     plt.plot(eigen_vecs[:,0])
    #     plt.plot(eigen_vecs[:,1])
    plt.show()

# Q-1.1
# PCA with data centering
pca(data_arr)