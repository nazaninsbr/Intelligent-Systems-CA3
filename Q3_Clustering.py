from helper import *
import numpy as np 
import random 
import copy 
import sys
import math
import matplotlib.pyplot as plt
import os


ITERATION = 200
KNN_DATA_FILE = './HW#01_Datasets/KNN/data.mat'
KNN_LABELS_FILE = './HW#01_Datasets/KNN/labels.mat'

def get_the_data():
	data = read_mat_file(KNN_DATA_FILE, 'data2')
	labels = read_mat_file(KNN_LABELS_FILE, 'labels')
	return data, labels


def calculateDist(ins, center):
	dist = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dist += (ins[i] - center[i]) ** 2
	dist = math.sqrt(dist)
	return dist

def calculateManhattanDist(ins, center):
	dist = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dist += abs(ins[i] - center[i])
	return dist

def calculateCosineSimilarity(ins, center):
	dot = 0
	norm_1 = 0
	norm_2 = 0
	for i in range(0, len(ins)):
		if i==len(ins):
			break
		dot += ins[i]*center[i]
		norm_1 += (ins[i])**2
		norm_2 += (center[i])**2
	if norm_1==0 or norm_2==0:
		return 1
	return dot/(math.sqrt(norm_1)*math.sqrt(norm_2))

def findMeanOfEverything(cluster):
	if len(cluster)==0:
		return [0, 0, 0, 0]
	s = []
	for fieldId in range(len(cluster[0][1])):
		s.append(0)
		for ins in cluster:
			s[-1] += ins[1][fieldId]
		s[-1] /= len(cluster)
	return s

def clusterBasedOnEveryThingWithEuclideanDistance(server_data, k):
	resultingCenters = {}
	resultingClusters = {}
	for k_num in k:
		print("For k = "+str(k_num)+":")
		# create the initial clusters
		server_data_copy = copy.deepcopy(server_data)
		centers = {}
		clusters = {}
		for i in range(k_num):
			ind = random.randint(0, len(server_data_copy)-1)
			centers[i] = server_data_copy[ind]
			clusters[i] = []
			clusters[i].append([ind, server_data_copy[ind]])
			del server_data_copy[ind]

		for ind, ins in enumerate(server_data_copy):
			minDist = sys.maxsize
			clusterNum = -1
			for center in centers.keys():
				dist = calculateDist(ins, centers[center])
				if dist < minDist:
					minDist = dist
					clusterNum = center
			clusters[clusterNum].append([ind, ins])

		# run the clustering algo.
		for inter_num in range(ITERATION):
			# calculate new centers 
			for clusterNum in clusters.keys():
				mean = findMeanOfEverything(clusters[clusterNum])
				centers[clusterNum] = [mean[0], mean[1], mean[2], mean[3]]

			# reassign elements
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break
					minDist = sys.maxsize
					newClusterNum = -1
					for center in centers.keys():
						dist = calculateDist(clusters[clusterNum][insId][1], centers[center])
						if dist < minDist:
							minDist = dist
							newClusterNum = center

					clusters[newClusterNum].append(clusters[clusterNum][insId])
					del clusters[clusterNum][insId]

			# cost function 
			cost_func = 0
			for clusterNum in clusters.keys():
				for insId in range(len(clusters[clusterNum])):
					if len(clusters[clusterNum]) == insId:
						break 
					cost_func += (calculateDist(clusters[clusterNum][insId][1], centers[clusterNum]))**2

			cost_func = cost_func / len(server_data)
			plt.scatter(inter_num, cost_func, color='black')
			# print('Cost : ', cost_func)

		print('Cluster Centers: ')
		print(centers)
		resultingCenters[k_num] = centers
		resultingClusters[k_num] = clusters
		# print distances 
		inner_dist = 0
		outer_dist = 0
		for centerId in centers.keys():
			for val in clusters[centerId]:
				inner_dist += calculateDist(val[1], centers[centerId])

		inner_dist = inner_dist/len(server_data)

		for centerId in centers.keys():
			for val in clusters[centerId]:
				for centerId2 in centers.keys():
					if not centerId2==centerId:
						outer_dist += calculateDist(val[1], centers[centerId2])

		outer_dist = outer_dist/len(server_data)

		print("Inner dist: "+str(inner_dist))
		print("Outer dist: "+str(outer_dist))

		for clusterNum in clusters.keys():
			fileName = './Clusters/'+str(k_num)+'_'+str(clusterNum)+'_Kcluster.txt'
			try:
				os.remove(fileName)
			except OSError:
				pass
			with open(fileName, 'a') as the_file:
				for item in clusters[clusterNum]:
					the_file.write(str(item))
					the_file.write('\n')

		plt.show()

	return resultingCenters, resultingClusters


def find_cluster_centers(data, labels, k):
	seen_classes, centers = [], []
	for ind, ins in enumerate(data):
		if len(centers)==k:
			break
		if not labels[ind] in seen_classes:
			seen_classes.append(labels[ind])
			centers.append(ind)
	return centers


def find_most_seen(count_each_class):
	if count_each_class[0]>=count_each_class[1] and count_each_class[0]>=count_each_class[2]:
		return 1
	if count_each_class[1]>=count_each_class[0] and count_each_class[1]>=count_each_class[2]:
		return 2
	if count_each_class[2]>=count_each_class[1] and count_each_class[2]>=count_each_class[0]:
		return 3

def calculate_cluster_majority(myClusters, labels):
	majority = {}
	for kVal in myClusters.keys():
		majority[kVal] = {}
		for classNumber in myClusters[kVal].keys():
			count_each_class = [0, 0, 0]
			for ins in myClusters[kVal][classNumber]:
				count_each_class[labels[ins[0]][0]-1] +=1
			majority[kVal][classNumber] = find_most_seen(count_each_class)
	return majority

def calculate_how_many_wrongly_classified(myClusters, labels, majority):
	stats = {}
	for kVal in myClusters.keys():
		all_instances, wrongly_clustered = 0, 0
		for classNumber in myClusters[kVal].keys(): 
			for ins in myClusters[kVal][classNumber]:
				all_instances += 1
				if not labels[ins[0]][0]==majority[kVal][classNumber]:
					# print("ins: {}, label: {}, majority:{}".format(ins, labels[ins[0]][0], majority[kVal][classNumber]))
					wrongly_clustered += 1
		stats[kVal] = [all_instances, wrongly_clustered]
	return stats


def main(data, labels):
	myCenters, myClusters = clusterBasedOnEveryThingWithEuclideanDistance(data, [5])
	return myClusters[5]


