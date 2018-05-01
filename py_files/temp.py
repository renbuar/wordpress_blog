import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def apply_PCA_all_components(X):
    from sklearn.decomposition import PCA
    pca = PCA(n_components = None)
    X_pca = pca.fit_transform(X_train_std)
    
    return pca, X_pca
    
#%% Load data
from sklearn import datasets
data_set = datasets.load_breast_cancer()
X=data_set.data
y=data_set.target

# Split into training and tets sets
X_train,X_test,y_train,y_test = (
        train_test_split(X,y,test_size=0.3, random_state=0))

#%% Normalise data
sc=StandardScaler() 
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

# Apply pca, accesing all components
pca = PCA(n_components = None)
X_train_pca=pca.fit_transform(X_train_std)

explained_variance = pca.explained_variance_ratio_
explained_variance_cum_sum = np.cumsum(explained_variance)

# Plot principle components explained variance

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
x = np.arange(1, len(explained_variance)+1)
ax.plot(x, explained_variance, label = 'Inidividual contribution')
ax.plot(x, explained_variance_cum_sum, label = 'Cumulative contribution')
ax.set_xlabel('Number of principal components')
ax.set_ylabel('Proportion of variance explained')
ax.legend()
ax.grid()
plt.show()

# Plot covariance matrix of original data
cov_original = np.cov(X_train_std.T)
plt.imshow(cov_original, interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.title('Covariance of orginal features')
plt.show()

# Plot covariance of principal components
cov_pca = np.cov(X_train_pca.T)
plt.imshow(cov_pca, interpolation='nearest', cmap=cm.Greys)
plt.colorbar()
plt.title('Covariance of principal components')

plt.show()
