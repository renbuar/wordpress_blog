import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

    
#%% Load data
from sklearn import datasets
data_set = datasets.load_breast_cancer()
X=data_set.data[:,0:2]


#%% Normalise data
sc=StandardScaler() 
sc.fit(X)
X_std=sc.transform(X)


#%% Fitting k means with 3 clusters
kmeans = KMeans(n_clusters=3)  
kmeans.fit(X_std)  

print('\nCluster centres:')
print(kmeans.cluster_centers_)
labels = kmeans.labels_

plt.scatter(X_std[:,0],X_std[:,1],
            c=labels, cmap=plt.cm.rainbow)
plt.xlabel('Normalised feature 1')
plt.ylabel('Normalised feature 2')
plt.show() 

#%% Effect of changing the number of clusters on distance to cluster centre

distance_to_closter_cluster_centre = []
for k in range(1,100):
    kmeans = KMeans(n_clusters=k)  
    kmeans.fit(X_std)
    distance = np.min(kmeans.transform(X_std),axis=1)
    average_distance = np.mean(distance)
    distance_to_closter_cluster_centre.append(average_distance)

clusters = np.arange(len(distance_to_closter_cluster_centre))+1
plt.plot(clusters, distance_to_closter_cluster_centre)
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average distance to closest cluster centroid')
plt.ylim(0,1.2)
plt.show()