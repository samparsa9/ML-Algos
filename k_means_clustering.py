import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import time as tm
from sklearn.datasets import make_blobs

class KMeans():
    def __init__(self, k):
        self.num_clusters = k
        self.cluster_dict = {}
        self.centroid_mean_diff_list = []
        self.data = None

    def fit(self, data_coords):
        self.data = data_coords
        for cluster in range(self.num_clusters):
            # Cluster dictionary will have the key being the cluster # and the value being a list of a tuple of the coordiantes
            # of the cluster point, and a list of the data points in that cluster, initialized to an empty list for now
            #self.cluster_dict[cluster][0] will give the (x,y) coordinate of the cluster centroid
            # self.cluster_dict[cluster][1] will return a list of [x y] cordinates
            x_max = np.max(self.data[:, 0])
            y_max = np.max(self.data[:, 1])
            self.cluster_dict[cluster] = [np.array([x_max/2, y_max/2]), []]
        
        plt.ion()
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(111)
        # Assiging datapoints to clusters
        while np.sum(self.centroid_mean_diff_list) > 0.01 or self.centroid_mean_diff_list == []:
            self.live_plotting(ax1)
            # refreshing points in the clusters
            for i in list(self.cluster_dict.keys()):
                self.cluster_dict[i][1] = []
            # refreshing the list
            self.centroid_mean_diff_list = []
            # For every point in our dataset
            for data_coord in data_coords:
                # the min cluster is set to nothing
                closest_cluster_distance = float('inf')
                closest_cluster = None
                # For each cluster, see how far away it is, if it is closer than our current min distance, update it to be the new
                # closest cluster
                for i in list(self.cluster_dict.keys()):
                    cluster_coord = self.cluster_dict[i][0]
                    distance = np.sqrt( ((cluster_coord[0] - data_coord[0]) ** 2) + ((cluster_coord[1] - data_coord[1]) ** 2))
                    if distance < closest_cluster_distance:
                        closest_cluster_distance = distance
                        closest_cluster = i
                self.cluster_dict[closest_cluster][1].append(data_coord)


            # Recalculate the mean of the cluster centroids
            print(f"Redefining centroids based on mean of datapoints currently in cluster...")
            for i in list(self.cluster_dict.keys()):
                old_centroid_mean = np.array(self.cluster_dict[i][0]) # This is an (x,y) point
                if len(self.cluster_dict[i][1]) > 0:
                    self.cluster_dict[i][0] = np.mean(self.cluster_dict[i][1], axis=0) # The new coordinate of our centroid based on the mean of the data points in it
                print(f"New Centroid Point: {self.cluster_dict[i][0]}")
                new_centroid_mean = np.array(self.cluster_dict[i][0])# This is an (x,y) point [4 5]
                print(f"New Centroid Mean: {new_centroid_mean}")
                try:
                    mean_diff = np.sqrt(((new_centroid_mean[0] - old_centroid_mean[0]) ** 2) + ((new_centroid_mean[1] - old_centroid_mean[1]) ** 2))
                except:
                    mean_diff = 0
                print(f"mean_diff: {mean_diff}")
                self.centroid_mean_diff_list.append(mean_diff)
                #tm.sleep(1)
            print("Finished updating means of centroids")
            print(f"List of new differences of means: {self.centroid_mean_diff_list}")
            print(f"sum: {np.sum(self.centroid_mean_diff_list)}")
            

        print("DONE")
        # plt.ioff()
        # plt.show()
        return self.cluster_dict
    
    def live_plotting(self, ax1):
        ax1.cla()
        ax1.plot(self.data[:, 0], self.data[:, 1], 'o')
        cluster_x_coords = []
        cluster_y_coords = []
        for i in list(self.cluster_dict.keys()):
            x_coord = self.cluster_dict[i][0][0]
            y_coord = self.cluster_dict[i][0][1]
            cluster_x_coords.append(x_coord)
            cluster_y_coords.append(y_coord)
        ax1.plot(cluster_x_coords, cluster_y_coords, 'o')

        plt.axis([-15, 15, -15, 15])  
        plt.draw()
        plt.pause(0.5)
            
        
        


def main():
#     data = np.array([
#     [5, 3],     # Eeach row is a datapoint, each column is a feature, since this is unsurpervised learning both the x and y
#     [1, 5],     # coordinates will be features
#     [2, 8], 
#     [3, 6],
#     [6, 2], 
#     [8, 1], 
#     [7, 3], 
#     [9, 4],
#     [7, 7], 
#     [6, 6]
# ])
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=4) # random_state 1 doesnt work
    data = X


    # Plot the dataset
    model = KMeans(3)
    print(model.fit(data))

    # plt.figure(figsize=(12,8))
    # plt.scatter(data[:, 0], data[:, 1], marker='o')
    # plt.show()


if __name__ == "__main__":
    main()

