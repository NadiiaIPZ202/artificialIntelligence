import numpy as np
from sklearn import preprocessing

input_data = np.array([[4.6, 3.9, -3.5],
                       [-2.9, 4.1, 3.3],
                       [2.2, 8.8, -6.1],
                       [3.9, 1.4, 2.2]])

data_binarized = preprocessing.Binarizer(threshold=2.2).transform(input_data)
print("\n Бінаризація даних:\n", data_binarized)

print("\nДані: ")
print("Середнє значення =", input_data.mean(axis=0))
print("Cтандартне відхилення =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nВиключення середнього: ")
print("Mean =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

data_scaler_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaler_minmax.fit_transform(input_data)
print("\nМасштабування MinМax:\n", data_scaled_minmax)

data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 Нормалізація даних:\n", data_normalized_l1)
print("\nl2 Нормалізація даних:\n", data_normalized_l2)
