import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Створення SVM класифікатора
classifier_svm = SVC()
# Тренування класифікатора
classifier_svm.fit(X_train, y_train)

# Створення наївного байєсовського класифікатора
classifier_nb = GaussianNB()
# Тренування класифікатора
classifier_nb.fit(X_train, y_train)

num_folds = 3


def count_value(classifier):
    accuracy_values = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_folds)
    print("Accuracy: " + str(round(100 * accuracy_values.mean(), 5)) + "%")

    precision_values = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_folds)
    print("Precision: " + str(round(100 * precision_values.mean(), 5)) + "%")

    recall_values = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_folds)
    print("Recall: " + str(round(100 * recall_values.mean(), 5)) + "%")

    f1_values = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_folds)
    print("F1: " + str(round(100 * f1_values.mean(), 5)) + "%")


print("SVM:")
count_value(classifier_svm)

print("NB:")
count_value(classifier_nb)
