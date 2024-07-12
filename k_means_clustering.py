import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import time as tm

class KMeans():
    def __init__(self, num_clusters=4):
        self.k = num_clusters
        self.cluster_dict = {}
        for cluster in range(num_clusters):
            # Cluster dictionary will have the key being the cluster # and the value being a list of a tuple of the coordiantes
            # of the cluster point, and a list of the data points in that cluster, initialized to zero for now
            self.cluster_dict[cluster] = [np.array([random.randint(10), random.randint(10)]), []]

        self.centroid_mean_diff_list = []
        self.data = None

    def fit(self, data_coords):
        self.data = data_coords
        plt.ion()
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(111)
        # Assiging datapoints to clusters
        while np.sum(self.centroid_mean_diff_list) > 0.000001 or self.centroid_mean_diff_list == []:
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
                    print(f"Cluster X - coordinate: {cluster_coord[0]}")
                    distance = np.sqrt( ((cluster_coord[0] - data_coord[0]) ** 2) + ((cluster_coord[1] - data_coord[1]) ** 2))
                    print(f"Distance: {distance}")
                    if distance < closest_cluster_distance:
                        closest_cluster_distance = distance
                        closest_cluster = i
                self.cluster_dict[closest_cluster][1].append(data_coord)


            # Recalculate the mean of the cluster centroids
            print(f"Redefining centroids based on mean of datapoints currently in cluster...")
            for i in list(self.cluster_dict.keys()):
                old_centroid_mean = np.mean(self.cluster_dict[i][0])
                self.cluster_dict[i][0] = np.mean(self.cluster_dict[i][1], axis=0) # The new coordiante of our centroid based on the mean of the data points in it
                new_centroid_mean = self.cluster_dict[i][0]
                mean_diff = abs(old_centroid_mean-new_centroid_mean)
                self.centroid_mean_diff_list.append(mean_diff)
            print("Finished updating means of centroids")
            print(f"List of new differences of means: {self.centroid_mean_diff_list}")
            self.live_plotting(ax1)

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

        plt.draw()
        plt.pause(1)
            
        
        


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
#     [6, 6],
#     [2, 5],
#     [2, 5],
#     [2, 5],
#     [2, 5],
#     [2, 5],
#     [2, 5],

# ])
    data1 = np.random.randint(1, 100, size=(100,2))


    model = KMeans(num_clusters=10)
    print(model.fit(data1))

    # plt.figure(figsize=(12,8))
    # plt.scatter(data[:, 0], data[:, 1], marker='o')
    # plt.show()


if __name__ == "__main__":
    main()

