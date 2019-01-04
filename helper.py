import csv
import scipy.io as sio

def read_mat_file(path, field_to_return):
	content = sio.loadmat(path)
	return content[field_to_return]

def read_file(file_name):
	data = []
	header_line = True
	with open(file_name) as tsvfile:
		reader = csv.reader(tsvfile, delimiter=',')
		for row in reader:
			if header_line==True:
				header_line = False			
				continue
			data.append(row)
	return data

def fix_iris_data_types(data):
	result = []
	for xInd in range(len(data)):
		if xInd == len(data):
			break
		result.append([xInd, [float(x) for x in data[xInd][:5]], data[xInd][5]])
	return result