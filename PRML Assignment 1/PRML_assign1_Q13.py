import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Dataset.csv', names = ['X', 'Y'])

# plotting the data distribution by pyplot
plt.xlabel('X-coordinates')
plt.ylabel('Y-coordinates')
plt.title('Distribution of original Data')
plt.plot(data['X'], data['Y'], 'g+')

# converting dataframe to np array
data_arr = np.array(data)
print("data in array form : ",data_arr)

## Kernel PCA
# find eigen_values and eigen vectors
def find_eigen_quantities(P_centered, k):

    eigen_vals, beta_eigen_vecs = np.linalg.eigh(P_centered)
    print("beta eigen vectors : ", beta_eigen_vecs)

    # higher_dim_eigen_vals = eigen_vals[0:6]
    top_eigen_val_ind = eigen_vals.argsort()[-k:][::-1]

    top_eigen_vals = []
    for i in range(0, k):
        top_eigen_vals.append(eigen_vals[top_eigen_val_ind[i]])

    top_eigen_vecs = []
    for i in range(0, k):
        top_eigen_vecs.append(beta_eigen_vecs[:, top_eigen_val_ind[i]])

    alpha = []
    for i in range(0, k):  # calculating alpha eigen vectors by dividing beta eigen vectors by (eigen_val)**(1/2)
        temp = (top_eigen_vals[i]) ** (1 / 2)
        alpha.append(top_eigen_vecs[i] / temp)

    alpha_mat = np.array(alpha)
    alpha_mat = alpha_mat.T

    coeff = np.matmul(P_centered, alpha_mat)  # calculating the coefficients

    return alpha_mat, coeff

# centering the kernel matrix
def find_center(P):
    I = np.identity(P.shape[0])

    iden_min_ones = I - (1 / P.shape[0])

    intermediate = np.matmul(iden_min_ones, P)

    P_centered = np.matmul(intermediate, iden_min_ones)
    # print(P_centered)
    # print(np.allclose(P_centered, P_centered.T))
    return P_centered

## By Polynomial Kernel
# this function will compute eigen values and eigen vectors (polynomial kernel)
def kernel_pca_poly(data_arr, degree, k):
    # P = (x.T*y + 1)**d
    P = np.dot(data_arr, data_arr.T)
    P = P + 1
    P = np.power(P, degree)

    P_centered = find_center(P)

    alpha_mat, coeff = find_eigen_quantities(P_centered, k)

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].set_xlabel('X-coordinates')
    ax[0].set_ylabel('Y-coordinates')
    ax[0].set_title('Kernel PCA scatter plot by polynomial kernel for degree {}'.format(degree))
    ax[0].scatter(coeff[:, 0], coeff[:, 1])  # plotting the coefficients of datapoints

    ax[1].set_xlabel('X-coordinates')
    ax[1].set_ylabel('Y-coordinates')
    plt.title('Kernel PCA plot by polynomial kernel for degree {}'.format(degree))
    ax[1].plot(coeff[:, 0], coeff[:, 1])

    plt.show()

## By Radial Basis Kernel
# this function will compute eigen values and eigen vectors (radial basis kernel)
def kernel_pca_radial_basis(sigma, l):
    # P = exp( (-||x - x'||**2) / 2*(sigma)**2 )
    P_k = np.zeros((data_arr.shape[0], data_arr.shape[0]))
    sig_sq = 2 * ((sigma) ** 2)

    for i in range(data_arr.shape[0]):
        for j in range(data_arr.shape[0]):
            diff = np.subtract(data_arr[i], data_arr[j])
            sum_ = 0
            for k in range(len(diff)):
                sum_ += diff[k] ** 2
            sum_ = -sum_
            div = sum_ / sig_sq
            P_k[i][j] = np.exp(div)

    P_k_centered = find_center(P_k)  # centering the kernel matrix

    alpha_mat, coeff = find_eigen_quantities(P_k_centered, l)

    plt.xlabel('X-coordinates')
    plt.ylabel('Y-coordinates')
    str1 = "Kernel PCA by Radial basis kernel for sigma : "+str(sigma)
    plt.title(str1)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    plt.scatter(coeff[:, 0], coeff[:, 1])
    # ax2.plot(coeff[:,0], coeff[:, 1])
    plt.show()

## Q-1.3A
# calling polynomial kernel pca for degree 2 and 3
for i in range(2,4):
    kernel_pca_poly(data_arr, i, 2)      # calling polynomial kernel function

# Q-1.3B
# calling radial basis kernel pca for 10 different values of sigma
sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for i in range(len(sigma)):
    kernel_pca_radial_basis(sigma[i], 2)      # calling radial basis function



