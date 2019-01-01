import copy 

def create_similarity_matrix():
	m = []
	m.append([1, 0.7, 0.65, 0.4, 0.2, 0.05])
	m.append([0.7, 1, 0.95, 0.7, 0.5, 0.35])
	m.append([0.65, 0.95, 1, 0.75, 0.55, 0.4])
	m.append([0.4, 0.7, 0.75, 1, 0.8, 0.65])
	m.append([0.2, 0.5, 0.55, 0.8, 1, 0.85])
	m.append([0.05, 0.35, 0.4, 0.65, 0.85, 1])
	return m

def find_clusters_to_join(similarity_matrix, this_steps_clusters):
	max_similarity = -1
	cluster_numbers_to_join = [-1, -1]
	for clusterInd1 in range(len(this_steps_clusters)):
		for data1 in this_steps_clusters[clusterInd1]:
			for clusterInd2 in range(len(this_steps_clusters)):
				if not clusterInd1==clusterInd2:
					for data2 in this_steps_clusters[clusterInd2]:
						if similarity_matrix[data1][data2]>max_similarity:
							max_similarity = similarity_matrix[data1][data2]
							cluster_numbers_to_join = [clusterInd1, clusterInd2]
	return cluster_numbers_to_join

def perform_single_link_clustering(similarity_matrix):
	clusters_and_step = {1:[[0], [1], [2], [3], [4], [5]]}
	steps = 1
	while steps<6:
		clusters_to_join = find_clusters_to_join(similarity_matrix, clusters_and_step[steps])
		steps += 1
		clusters_and_step[steps] = copy.deepcopy(clusters_and_step[steps-1])
		clusters_and_step[steps][clusters_to_join[0]].extend(clusters_and_step[steps][clusters_to_join[1]])
		del clusters_and_step[steps][clusters_to_join[1]]
	print(clusters_and_step)
	return clusters_and_step

def perform_hierarchical_clustering_with_scipy(similarity_matrix):
	import matplotlib.pyplot as plt
	from scipy.cluster import hierarchy
	import numpy as np

	ytdist = np.array(similarity_matrix)
	Z = hierarchy.linkage(ytdist, 'single')
	plt.figure()
	dn = hierarchy.dendrogram(Z)
	plt.show()

def main():
	similarity_matrix = create_similarity_matrix()
	perform_single_link_clustering(similarity_matrix)
	perform_hierarchical_clustering_with_scipy(similarity_matrix)




