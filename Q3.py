from helper import *
from Q3_Clustering import main as clustering_main
import math 
import copy 

DATA_FILE_NAME = './Data/data.mat'
LABEL_FILE_NAME = './Data/labels.mat'

def calculateDist(ins, center):
	dist = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dist += (ins[i] - center[i]) ** 2
	dist = math.sqrt(dist)
	return dist

def sort_based_on_distance(distances, indexes):  
	sorted_distances, sorted_indexes = copy.deepcopy(distances), copy.deepcopy(indexes)
	for i in range(1, len(distances)): 
  
		key = sorted_distances[i] 
		index_val = sorted_indexes[i]
		j = i-1
		while j >=0 and key < sorted_distances[j] : 
				sorted_distances[j+1] = sorted_distances[j]
				sorted_indexes[j+1] = sorted_indexes[j] 
				j -= 1
		sorted_distances[j+1] = key 
		sorted_indexes[j+1] = index_val
	return sorted_distances, sorted_indexes

def find_k_closest(x, training_set, k):
	distances = []
	indexes = []
	for ind, ins in enumerate(training_set):
		if ind==x:
			continue
		dist = calculateDist(ins, training_set[x])
		distances.append(dist)
		indexes.append(ind)

	sorted_distances, sorted_indexes = sort_based_on_distance(distances, indexes)

	nearerst = []
	for i in range(k):
		nearerst.append(sorted_indexes[i])

	return nearerst

def find_most_seen(count_each_class):
	if count_each_class[1]>=count_each_class[2] and count_each_class[1]>=count_each_class[3]:
		return 1
	if count_each_class[2]>=count_each_class[1] and count_each_class[2]>=count_each_class[3]:
		return 2
	if count_each_class[3]>=count_each_class[1] and count_each_class[3]>=count_each_class[2]:
		return 3

def find_predicted_class(A, labels):
	count = {1:0, 2:0, 3:0}
	for ins in A:
		l = labels[ins]
		count[l[0]] += 1
	return find_most_seen(count)

def find_k_closest_when_using_clustering(x, training_set, k):
	distances = []
	indexes = []
	for ind, ins in enumerate(training_set):
		if ind==x:
			continue
		dist = calculateDist(ins[1], training_set[x][1])
		distances.append(dist)
		indexes.append(ins[0])

	sorted_distances, sorted_indexes = sort_based_on_distance(distances, indexes)

	nearerst = []
	for i in range(k):
		nearerst.append(sorted_indexes[i])

	return nearerst

def knn_before_clustering(data, labels, k):
	for k_num in k:
		all_instances = 0
		wrongly_classified = 0
		for xId in range(len(data)):
			all_instances += 1
			A = find_k_closest(xId, data, k_num)
			pred_class = find_predicted_class(A, labels)
			if not pred_class==labels[xId][0]:
				wrongly_classified += 1
		print('for k = {}, {} of {} instances wrongly classified => accuracy = {}%'.format(k_num, wrongly_classified, all_instances, (all_instances-wrongly_classified)/all_instances*100))


def knn_after_clustering(clusters, labels, k):
	for k_num in k:
		all_instances = 0
		wrongly_classified = 0
		for clusterId in clusters.keys():
			for thisId in range(len(clusters[clusterId])):
				all_instances += 1
				A = find_k_closest_when_using_clustering(thisId, clusters[clusterId], k_num)
				pred_class = find_predicted_class(A, labels)
				if not pred_class==labels[clusters[clusterId][thisId][0]][0]:
					wrongly_classified += 1
		print('for k = {}, {} of {} instances wrongly classified => accuracy = {}%'.format(k_num, wrongly_classified, all_instances, (all_instances-wrongly_classified)/all_instances*100))



def read_the_data_for_Q3():
	data = read_mat_file(DATA_FILE_NAME, 'data2')
	labels = read_mat_file(LABEL_FILE_NAME, 'labels')
	return data, labels

def main():
	data, labels = read_the_data_for_Q3()
	print('Accuracy before clustering')
	knn_before_clustering(data.tolist(), labels.tolist(), [3, 5, 7, 9])
	clusters = clustering_main(data.tolist(), labels.tolist())
	print('Accuracy after clustering')
	knn_after_clustering(clusters, labels.tolist(), [3, 5, 7, 9])


