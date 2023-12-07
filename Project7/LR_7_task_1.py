import numpy as np
from numpy.random import choice as np_choice
import matplotlib

matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import networkx as nx

# Масив міст
cities = [
    'Вінниця', 'Дніпро', 'Донецьк', 'Житомир', 'Запоріжжя', 'Івано-Франківськ', 'Київ', 'Кропивницький',
    'Луганськ', 'Луцьк', 'Львів', 'Миколаїв', 'Одеса', 'Полтава', 'Рівне', 'Сімферополь', 'Суми', 'Тернопіль',
    'Ужгород', 'Харків', 'Херсон', 'Хмельницький', 'Черкаси', 'Чернівці', 'Чернігів'
]


class AntColony:
    def __init__(self, distances, n_ants, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self, n_iterations):
        best_path = None
        all_time_best_path = ("placeholder", np.inf)
        for i in range(n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.pheromone, self.all_inds, self.decay)
            self.intensify_pheronome(all_paths, self.pheromone)
            self.pheromone * self.decay
            current_best_path = min(all_paths, key=lambda x: x[1])
            if current_best_path[1] < all_time_best_path[1]:
                all_time_best_path = current_best_path
        return all_time_best_path

    # Розповсюдження феромону на всіх шляхах
    def spread_pheronome(self, all_paths, pheromone, all_inds, decay):
        pheromone * (1 - decay)
        for path, dist in all_paths:
            for move in path:
                pheromone[move] += 1.0 / self.distances[move]

    # Підсилення феромону на шляхах, відсортованих за відстанню
    def intensify_pheronome(self, all_paths, pheromone):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths:
            for move in path:
                pheromone[move] += 1.0 / self.distances[move]

    # Генерація всіх шляхів для всіх мурів
    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path, distance = self.gen_path_dist()
            all_paths.append((path, distance))
        return all_paths

    # Генерація шляху та відстані для одного мура
    def gen_path_dist(self):
        path = []
        visited_nodes = set()
        current_node = 17  # Індекс міста Тернопіль
        visited_nodes.add(current_node)
        total_distance = 0

        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[current_node], self.distances[current_node], visited_nodes)
            path.append((current_node, move))
            total_distance += self.distances[current_node, move]
            current_node = move
            visited_nodes.add(current_node)

        path.append((current_node, path[0][0]))
        total_distance += self.distances[current_node, path[0][0]]
        return path, total_distance

    # Вибір міста для переходу муром
    def pick_move(self, pheromone, dist, visited):
        row = pheromone ** self.alpha * ((1.0 / (dist + 1e-10)) ** self.beta)

        probabilities = row / row.sum()
        left_prob = probabilities[probabilities > 0]

        for city in visited:
            probabilities[city] = 0

        probabilities /= probabilities.sum()

        if np.isnan(probabilities).any():
            # Якщо ймовірності містять NaN, вибрати випадкове місто
            chosen_city = np.random.choice(self.all_inds)
        else:
            chosen_city = np.random.choice(self.all_inds, p=probabilities)
        return chosen_city

    # Візуалізація найкращого шляху
    def plot_graph(self, best_path):
        G = nx.Graph()
        for move in best_path:
            city1, city2 = move
            G.add_edge(city1, city2)

        plt.figure(figsize=(10, 8))
        nx.draw(G, with_labels=True, labels={i: cities[i] for i in range(len(cities))}, font_size=8,
                font_color='black', font_weight='bold', node_color='skyblue', node_size=800,
                edge_color='black', linewidths=1, alpha=0.9)

        plt.title('Найкращий шлях')
        plt.show()


# Матриця відстаней між точками
distances = np.array([
    [0, 645, 868, 125, 748, 366, 256, 316, 1057, 382, 360, 471, 428, 593, 311, 844, 602, 232, 575, 734, 521, 120, 343,
     312, 396],
    [645, 0, 252, 664, 81, 901, 533, 294, 394, 805, 975, 343, 468, 196, 957, 446, 430, 877, 1130, 213, 376, 765, 324,
     891, 672],
    [868, 252, 0, 858, 217, 1171, 727, 520, 148, 1111, 1221, 611, 731, 390, 1045, 591, 706, 1100, 1391, 335, 560, 988,
     547, 1141, 867],
    [125, 664, 858, 0, 738, 431, 131, 407, 1182, 257, 423, 677, 557, 468, 187, 803, 477, 298, 671, 690, 624, 185, 321,
     389, 271],
    [748, 81, 217, 738, 0, 1119, 607, 303, 365, 681, 833, 377, 497, 270, 925, 365, 477, 977, 1488, 287, 297, 875, 405,
     957, 747],
    [366, 901, 1171, 431, 1119, 0, 561, 618, 1402, 328, 135, 747, 627, 898, 296, 1070, 908, 134, 280, 1040, 798, 246,
     709, 143, 701],
    [256, 533, 727, 131, 607, 561, 0, 298, 811, 388, 550, 490, 489, 337, 318, 972, 346, 427, 806, 478, 551, 315, 190,
     538, 149],
    [316, 294, 520, 407, 303, 618, 298, 0, 668, 664, 710, 174, 294, 246, 627, 570, 506, 547, 883, 387, 225, 435, 126,
     637, 363],
    [1057, 394, 148, 1182, 365, 1402, 811, 668, 0, 1199, 1379, 857, 977, 474, 1129, 739, 253, 1289, 1539, 333, 806,
     1177, 706, 1292, 951],
    [382, 805, 1111, 257, 681, 328, 388, 664, 1199, 0, 152, 780, 856, 725, 70, 1052, 734, 159, 413, 866, 869, 263, 578,
     336, 949],
    [360, 975, 1221, 423, 833, 135, 550, 710, 1379, 152, 0, 850, 970, 891, 232, 1173, 896, 128, 261, 1028, 1141, 240,
     740, 278, 690],
    [471, 343, 611, 677, 377, 747, 490, 174, 857, 780, 850, 0, 120, 420, 864, 282, 681, 754, 999, 556, 51, 590, 300,
     642, 640],
    [428, 468, 731, 557, 497, 627, 489, 294, 977, 856, 970, 120, 0, 540, 741, 392, 800, 660, 1009, 831, 171, 548, 420,
     515, 529],
    [593, 196, 390, 468, 270, 898, 337, 246, 474, 725, 891, 420, 540, 0, 665, 635, 261, 825, 1149, 141, 471, 653, 279,
     892, 477],
    [311, 957, 1045, 187, 925, 296, 318, 627, 1129, 70, 232, 864, 741, 665, 0, 1157, 664, 162, 484, 805, 834, 193, 508,
     331, 458],
    [844, 446, 591, 803, 365, 1070, 972, 570, 739, 1052, 1173, 282, 392, 635, 1157, 0, 896, 1097, 1363, 652, 221, 964,
     696, 981, 1112],
    [602, 430, 706, 477, 477, 908, 346, 506, 253, 734, 896, 681, 800, 261, 664, 896, 0, 774, 1138, 190, 732, 662, 540,
     883, 350],
    [232, 877, 1100, 298, 977, 134, 427, 547, 1289, 159, 128, 754, 660, 825, 162, 1097, 774, 0, 338, 987, 831, 112, 575,
     176, 568],
    [575, 1130, 1391, 671, 1488, 280, 806, 883, 1539, 413, 261, 999, 1009, 1149, 484, 1363, 1138, 338, 0, 1299, 1065,
     455, 984, 444, 951],
    [734, 213, 335, 690, 287, 1040, 478, 387, 333, 866, 1028, 556, 831, 141, 805, 652, 190, 987, 1299, 0, 576, 854, 420,
     1036, 608],
    [521, 376, 560, 624, 297, 798, 551, 225, 806, 869, 1141, 51, 171, 471, 834, 221, 732, 831, 1065, 576, 0, 641, 351,
     713, 691],
    [120, 765, 988, 185, 875, 246, 315, 435, 1177, 263, 240, 590, 548, 653, 193, 964, 662, 112, 455, 854, 641, 0, 463,
     190, 455],
    [343, 324, 547, 321, 405, 709, 190, 126, 706, 578, 740, 300, 420, 279, 508, 696, 540, 575, 984, 420, 351, 463, 0,
     660, 330],
    [312, 891, 1141, 389, 957, 143, 538, 637, 1292, 336, 278, 642, 515, 892, 331, 981, 883, 176, 444, 1036, 713, 190,
     660, 0, 695],
    [396, 672, 867, 271, 747, 701, 149, 363, 951, 949, 690, 640, 529, 477, 458, 1112, 350, 568, 951, 608, 691, 455, 330,
     695, 0]
])

# Кількість мурах
n_ants = 10
# Коефіцієнт випаровування феромону
decay = 0.1
# Параметри для обчислення ймовірностей вибору міста мурахою
alpha = 1
beta = 2

ant_colony = AntColony(distances, n_ants, decay, alpha, beta)
n_iterations = 100

best_path, best_distance = ant_colony.run(n_iterations)

best_path_title = [(cities[i], cities[j]) for i, j in best_path]

# Виведення результатів
print("Найкращий шлях:")
path_str = " -> ".join([f"{cities[move[0]]}-{cities[move[1]]}" for move in best_path])
print(path_str)
print(f"Відстань найкращого шляху: {best_distance}")

# Виклик функції для візуалізації графа
ant_colony.plot_graph(best_path)
