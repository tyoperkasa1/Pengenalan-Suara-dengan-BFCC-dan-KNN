import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def klasifikasiKNN(x):
	path1 = x
	path2 = './model/newData.xlsx'

	dataset1 = pd.read_excel(path1, header=None)
	dataset2 = pd.read_excel(path2, header=None)

	x_train = dataset1.iloc[1:, :40].values
	y_train = dataset1.iloc[1:, 40].values

	x_test = dataset2.iloc[1:, :40].values
	knn = KNeighborsClassifier(n_neighbors=5, weights="distance", metric="euclidean")
	knn.fit(x_train,y_train)
	klasifikasiDataMentah = knn.predict(x_test)

	fileBaruMasuk = [f for f in os.listdir("./dataNew/") if os.path.isfile(os.path.join("./dataNew/", f))]
	fileBaruMasuk.sort()
	ti_c = os.path.getctime("./dataNew/"+fileBaruMasuk[-1])
	c_ti = time.ctime(ti_c)
	riwayatPemakaian = open('./historyLog.txt', 'a')
	riwayatPemakaian.write('\n')
	riwayatPemakaian.write(fileBaruMasuk[-1])
	riwayatPemakaian.write(' ')
	riwayatPemakaian.write(c_ti)
	riwayatPemakaian.write(' berprediksi penyakit ')
	riwayatPemakaian.write(str(klasifikasiDataMentah))
	riwayatPemakaian.close()

	print("Prediksi ", klasifikasiDataMentah)
	return klasifikasiDataMentah
