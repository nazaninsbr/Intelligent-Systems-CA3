from helper import *
import copy
import random
import sys
import math
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATA_FILE_NAME = "./Data/Iris.csv"
EPOCS = 150
NUMBER_OF_CLUSTERS = [3, 5, 7, 9]

def calculate_distance_from_center(center, instance):
	dist = 0
	for xInd in range(len(instance)):
		dist += (center[xInd] - instance[xInd])**2
	return math.sqrt(dist)

def pick_initial_clusters(iris_data, k_num):
	centers, clusters = {}, {}
	for i in range(k_num):
		ind = random.randint(0, len(iris_data)-1)
		centers[i] = (iris_data[ind][0], iris_data[ind][1], '')
		clusters[i] = []
		clusters[i].append(iris_data[ind])
		del iris_data[ind]

	for ins in iris_data:
		minDist = sys.maxsize
		clusterNum = -1
		for center in centers.keys():
			dist = calculate_distance_from_center(centers[center][1], ins[1])
			if dist < minDist:
				minDist = dist
				clusterNum = center
		clusters[clusterNum].append(ins)
	return centers, clusters

def calculate_mean_of_all_fields(instances):
	insts = [x[1] for x in instances]
	mean_of_the_fields = np.mean(np.array(insts), axis=0).tolist()
	return [-1, mean_of_the_fields, '-']

def calculate_new_centers(clusters):
	centers = {}
	for cluster_number in clusters.keys():
		
		centers[cluster_number] = calculate_mean_of_all_fields(clusters[cluster_number])
	return centers

def reassign_items_to_clusters(clusters, centers):
	for cluster_number in clusters.keys():
		for insId in range(len(clusters[cluster_number])):
			if insId==len(clusters[cluster_number]):
				break
			minDist = sys.maxsize
			newClusterNum = -1
			for center_number in centers.keys():
				dist = calculate_distance_from_center(clusters[cluster_number][insId][1], centers[center_number][1])
				if dist < minDist:
					minDist = dist
					newClusterNum = center_number
			clusters[newClusterNum].append(clusters[cluster_number][insId])
			del clusters[cluster_number][insId]
	return clusters

def calculate_this_epochs_cost(clusters, centers):
	cost = 0
	for center_number in centers.keys():
		for cluster_number in clusters.keys():
			for ins in clusters[cluster_number]:
				cost += calculate_distance_from_center(centers[center_number][1], ins[1])
	return cost 

def calculate_inner_and_outer_dist(clusters, centers, number_of_data_points):
	inner_dist = 0
	outer_dist = 0
	for centerId in centers.keys():
		for val in clusters[centerId]:
			inner_dist += calculate_distance_from_center(centers[centerId][1], val[1])

	inner_dist = inner_dist/number_of_data_points

	for centerId in centers.keys():
		for val in clusters[centerId]:
			for centerId2 in centers.keys():
				if not centerId2==centerId:
					outer_dist += calculate_distance_from_center(centers[centerId2][1], val[1])
		outer_dist = outer_dist/number_of_data_points

	return inner_dist, outer_dist

def plot_the_values(inner_dist_plot_vals, outer_dist_plot_vals, cost_plot_vals, iteration_plot_vals, k_num):
	plt.scatter(iteration_plot_vals, cost_plot_vals)
	plt.title('Cost Values '+str(k_num))

	plt.show()

	plt.scatter(iteration_plot_vals, inner_dist_plot_vals)
	plt.title('Inner distance Values '+str(k_num))

	plt.show()

	plt.scatter(iteration_plot_vals, outer_dist_plot_vals)
	plt.title('Outer Distance Values '+str(k_num))

	plt.show()

def find_the_majority_class(number_of_each_class):
	if number_of_each_class['Iris-setosa'] > number_of_each_class['Iris-versicolor'] and  number_of_each_class['Iris-setosa'] > number_of_each_class['Iris-virginica']:
		return 'Iris-setosa'
	elif number_of_each_class['Iris-setosa'] < number_of_each_class['Iris-versicolor'] and  number_of_each_class['Iris-versicolor'] > number_of_each_class['Iris-virginica']:
		return 'Iris-versicolor'
	else:
		return 'Iris-virginica'

def calculate_accuracy(clusters, number_of_data_points):
	wrongly_clustered = 0
	for clusterNumber in clusters.keys():
		number_of_each_class = {'Iris-setosa':0, 'Iris-versicolor':0, 'Iris-virginica':0}
		for ins in clusters[clusterNumber]:
			number_of_each_class[ins[2]] += 1

		print(str(clusterNumber)+':'+str(number_of_each_class))
		majority_class = find_the_majority_class(number_of_each_class)

		for ins in clusters[clusterNumber]:
			if not ins[2]==majority_class:
				wrongly_clustered += 1
	print('Accuracy is: '+str((number_of_data_points - wrongly_clustered)/number_of_data_points*100)+' %')

def calculate_separating_index(clusters):
	d_Sl_Sl = {}
	for clusterNumber in clusters.keys():
		min_dist = sys.maxsize
		for x1_id in range(len(clusters[clusterNumber])):
			for x2_id in range(len(clusters[clusterNumber])):
				if x1_id==x2_id:
					continue
				dist = calculate_distance_from_center(clusters[clusterNumber][x1_id][1], clusters[clusterNumber][x2_id][1])
				if dist<min_dist:
					min_dist = dist
		d_Sl_Sl[clusterNumber] = min_dist

	min_j_val = sys.maxsize
	min_j = -1
	for clusterNumber_1 in clusters.keys():
		min_i = -1
		min_i_val = sys.maxsize
		for clusterNumber_2 in clusters.keys():
			if clusterNumber_1 == clusterNumber_2:
				continue 
			min_dist = sys.maxsize
			for val1 in clusters[clusterNumber_1]:
				for val2 in clusters[clusterNumber_2]:
					dist = calculate_distance_from_center(val1[1], val2[1])
					if dist<min_dist:
						min_dist = dist
			if min_dist<min_i_val:
				min_i_val = min_dist
				min_i = clusterNumber_2
		if min_i_val<min_j_val:
			min_j_val = min_i_val
			min_j = clusterNumber_1

	return min_j


def run_k_means(data):
	for k_num in NUMBER_OF_CLUSTERS:
		inner_dist_plot_vals, outer_dist_plot_vals, cost_plot_vals, iteration_plot_vals = [], [], [], []
		iris_data = copy.deepcopy(data)
		centers, clusters = pick_initial_clusters(iris_data, k_num)

		for epoch_number in range(EPOCS):
			
			centers = calculate_new_centers(clusters)
			clusters = reassign_items_to_clusters(clusters, centers)

			cost = calculate_this_epochs_cost(clusters, centers)
			print('['+str(epoch_number)+'] cost: '+str(cost))

			inner_dist, outer_dist = calculate_inner_and_outer_dist(clusters, centers, len(data))
			print('['+str(epoch_number)+'] inner distance: '+str(inner_dist))
			print('['+str(epoch_number)+'] outer distance: '+str(outer_dist))

			inner_dist_plot_vals.append(inner_dist)
			outer_dist_plot_vals.append(outer_dist)
			cost_plot_vals.append(cost)
			iteration_plot_vals.append(epoch_number)
		
		plot_the_values(inner_dist_plot_vals, outer_dist_plot_vals, cost_plot_vals, iteration_plot_vals, k_num)
		print('SI: ',calculate_separating_index(clusters))
		calculate_accuracy(clusters, len(data))

def not_random_initial_clusters(iris_data, k_num):
	centers, clusters = {}, {}
	INITIAL_CENTERS = [5, 89, 143]
	for i in range(k_num):
		ind = INITIAL_CENTERS[i]
		centers[i] = (iris_data[ind][0], iris_data[ind][1], '')
		clusters[i] = []
		clusters[i].append(iris_data[ind])
		del iris_data[ind]

	for ins in iris_data:
		minDist = sys.maxsize
		clusterNum = -1
		for center in centers.keys():
			dist = calculate_distance_from_center(centers[center][1], ins[1])
			if dist < minDist:
				minDist = dist
				clusterNum = center
		clusters[clusterNum].append(ins)
	return centers, clusters


def run_k_means_with_set_initial_centers(data):
	k_num = 3
	inner_dist_plot_vals, outer_dist_plot_vals, cost_plot_vals, iteration_plot_vals = [], [], [], []
	iris_data = copy.deepcopy(data)
	centers, clusters = not_random_initial_clusters(iris_data, k_num)

	for epoch_number in range(EPOCS):
			
		centers = calculate_new_centers(clusters)
		clusters = reassign_items_to_clusters(clusters, centers)

		cost = calculate_this_epochs_cost(clusters, centers)
		print('['+str(epoch_number)+'] cost: '+str(cost))

		inner_dist, outer_dist = calculate_inner_and_outer_dist(clusters, centers, len(data))
		print('['+str(epoch_number)+'] inner distance: '+str(inner_dist))
		print('['+str(epoch_number)+'] outer distance: '+str(outer_dist))

		inner_dist_plot_vals.append(inner_dist)
		outer_dist_plot_vals.append(outer_dist)
		cost_plot_vals.append(cost)
		iteration_plot_vals.append(epoch_number)
		
	plot_the_values(inner_dist_plot_vals, outer_dist_plot_vals, cost_plot_vals, iteration_plot_vals, k_num)
	calculate_accuracy(clusters, len(data))



def plot_and_calculate_the_cost_values(all_cost_plot_vals, all_iteration_plot_vals, k_num):
	plt.scatter(all_iteration_plot_vals[0], all_cost_plot_vals[0], color='blue')
	plt.scatter(all_iteration_plot_vals[1], all_cost_plot_vals[1], color='red')
	plt.scatter(all_iteration_plot_vals[2], all_cost_plot_vals[2], color='green')
	plt.scatter(all_iteration_plot_vals[3], all_cost_plot_vals[3], color='silver')
	plt.scatter(all_iteration_plot_vals[4], all_cost_plot_vals[4], color='black')
	plt.title('cost for k='+str(k_num))
	plt.show()

	number_of_instances = 0
	mean = 0
	variance = 0
	for x in all_cost_plot_vals:
		for val in x:
			number_of_instances += 1
			mean += val

	for x in all_cost_plot_vals:
		for val in x:
			variance += (val - mean)**2

	mean = mean / number_of_instances
	print('k = {} mean = {} variance = {}'.format(k_num, mean, variance))

def run_k_means_5_times_for_each_number(data):
	for k_num in NUMBER_OF_CLUSTERS:
		all_cost_plot_vals, all_iteration_plot_vals = [], []
		
		for _ in range(5):
			cost_plot_vals, iteration_plot_vals = [], []
			iris_data = copy.deepcopy(data)
			centers, clusters = pick_initial_clusters(iris_data, k_num)

			for epoch_number in range(EPOCS):
				
				centers = calculate_new_centers(clusters)
				clusters = reassign_items_to_clusters(clusters, centers)

				cost = calculate_this_epochs_cost(clusters, centers)
				cost_plot_vals.append(cost)
				iteration_plot_vals.append(epoch_number)

			all_iteration_plot_vals.append(iteration_plot_vals)
			all_cost_plot_vals.append(cost_plot_vals)

		plot_and_calculate_the_cost_values(all_cost_plot_vals, all_iteration_plot_vals, k_num)


def main():
	data = fix_iris_data_types(read_file(DATA_FILE_NAME))
	run_k_means(data)
	run_k_means_5_times_for_each_number(data)
	run_k_means_with_set_initial_centers(data)

