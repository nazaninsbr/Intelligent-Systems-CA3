from helper import *
import copy 
import numpy as np

DATA_FILE_PATH = './Data/test_images.mat'
LABELS_FILE_PATH = './Data/test_labels.mat'

def read_the_data():
	data = read_mat_file(DATA_FILE_PATH, "test_images")
	labels = read_mat_file(LABELS_FILE_PATH, "test_labels")
	return data, labels

def	perform_pca_using_sklearn(data):
	from sklearn.decomposition import PCA
	cdata = copy.deepcopy(data)
	A = np.array(cdata)
	print('The images before transformation:')
	print(A)
	pca = PCA(2)
	pca.fit(A)
	print('PCA components:')
	print(pca.components_)
	print('PCA variance:')
	print(pca.explained_variance_)

	B = pca.transform(A)
	print('The images after transformation:')
	print(B)
	return B

def perform_pca_using_numpy_helper_functions(data):
	from numpy import mean
	from numpy import cov
	from numpy.linalg import eig

	cdata = copy.deepcopy(data)
	A = np.array(cdata)
	print('The images before transformation: ')
	print(A.shape)
	M = mean(A.T, axis=1)
	print('The mean matrix: ')
	print(M.shape)
	print('subtracting column means: ')
	C = A - M
	print(C)
	print('calculate covariance matrix: ')
	V = cov(C.T)
	print(V)
	print('eigendecomposition of covariance matrix: ')
	values, vectors = eig(V)
	print(vectors)
	print(values)
	print('project data onto the vector: ')
	P = vectors.T.dot(C.T)
	print(P.T)
	print('Removing the 0 features...')
	mat = []
	for x in P.T:
		row = []
		for yInd in range(len(x)):
			if yInd>667:
				break
			row.append(x[yInd])
		mat.append(row)
	return np.array(mat)

def print_ten_of_each_class(data, labels):
	from_each_class = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}
	for i in range(len(labels)):
		if len(from_each_class[labels[i]])<10:
			from_each_class[labels[i]].append(data[i])

	for k in from_each_class.keys():
		path = './Numbers/'+str(k)+'.txt'
		f = open(path, 'w')
		for img in from_each_class[k]:
			f.write(str(img)+'\n')
		f.close()

def main():
	data, labels = read_the_data()
	X1 = perform_pca_using_sklearn(data)
	print('SKlearn shape: ', X1.shape)
	X2 = perform_pca_using_numpy_helper_functions(data)
	print('Numpy shape: ', X2.shape)
	print_ten_of_each_class(X2, labels[0].tolist())

