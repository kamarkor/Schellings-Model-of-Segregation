# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class Schelling:
    def __init__(self, size, empty_ratio, similarity_threshold, n_neighbors):
        self.size = size 
        self.empty_ratio = empty_ratio
        self.similarity_threshold = similarity_threshold
        self.n_neighbors = n_neighbors
        p = [(1-empty_ratio)/2, (1-empty_ratio)/2, empty_ratio]
        city_size = int(np.sqrt(self.size))**2
        self.city = np.random.choice([-1, 1, 0], size=city_size, p=p)
        self.city = np.reshape(self.city, (int(np.sqrt(city_size)), int(np.sqrt(city_size))))
    def run(self):
        for (row, col), value in np.ndenumerate(self.city):
            race = self.city[row, col]
            if race != 0:
                neighborhood = self.city[row-self.n_neighbors:row+self.n_neighbors, col-self.n_neighbors:col+self.n_neighbors]
                neighborhood_size = np.size(neighborhood)
                n_empty_houses = len(np.where(neighborhood == 0)[0])
                if neighborhood_size != n_empty_houses + 1:
                    n_similar = len(np.where(neighborhood == race)[0]) - 1
                    similarity_ratio = n_similar / (neighborhood_size - n_empty_houses - 1.)
                    is_unhappy = (similarity_ratio < self.similarity_threshold)
                    if is_unhappy:
                        empty_houses = list(zip(np.where(self.city == 0)[0], np.where(self.city == 0)[1]))
                        random_house = random.choice(empty_houses)
                        self.city[random_house] = race
                        self.city[row,col] = 0
    def get_mean_similarity_ratio(self):
        count = 0
        similarity_ratio = 0
        for (row, col), value in np.ndenumerate(self.city):
            race = self.city[row, col]
            if race != 0:
                neighborhood = self.city[row-self.n_neighbors:row+self.n_neighbors, col-self.n_neighbors:col+self.n_neighbors]
                neighborhood_size = np.size(neighborhood)
                n_empty_houses = len(np.where(neighborhood == 0)[0])
                if neighborhood_size != n_empty_houses + 1:
                    n_similar = len(np.where(neighborhood == race)[0]) - 1
                    similarity_ratio += n_similar / (neighborhood_size - n_empty_houses - 1.)
                    count += 1
        return similarity_ratio / count

nb_of_sim = 100
nb_of_iter = 20
pop_size = 2000
empty_houses = 0.2
similarity = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
n_neighbors = 3

mean_similarity_ratio_all = np.ndarray((len(similarity), nb_of_sim, nb_of_iter))
for k in range(len(similarity)):
	for i in range(nb_of_sim):
		schelling = Schelling(pop_size, empty_houses, similarity[k], n_neighbors)
		for j in range(nb_of_iter):
			schelling.run()
			mean_similarity_ratio_all[k, i, j] = schelling.get_mean_similarity_ratio()

fig, ax = plt.subplots(1, figsize=(8, 5))
plt.title("Mean Similarity Ratio", fontsize=15)
ax.plot(np.mean(mean_similarity_ratio_all[3], axis=0), color="grey", label="0.3")
ax.plot(np.mean(mean_similarity_ratio_all[4], axis=0), color="red", label="0.4")
ax.plot(np.mean(mean_similarity_ratio_all[6], axis=0), color="b", label="0.6")
ax.plot(np.mean(mean_similarity_ratio_all[7], axis=0), color="green", label="0.7")
plt.legend(loc="center right", title="similarity\nthreshold", frameon=False)
plt.xlabel("Iterations")
plt.ylim((0.5,1))
plt.xlim((0,20))
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.grid(True);

threshold_vs_mean = []
for i in range(len(similarity)):
	threshold_vs_mean.append(np.mean(mean_similarity_ratio_all[i], axis=0)[19])

plt.figure(figsize=(8,5))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Similarity threshold vs. Mean Similarity Ratio", fontsize=15)
plt.xlabel("Similarity threshold")
plt.ylabel("Mean Similarity Ratio")
plt.scatter(similarity, threshold_vs_mean)
plt.grid(True);