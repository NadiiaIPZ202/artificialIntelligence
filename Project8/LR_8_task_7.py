import numpy as np
import cv2
from sklearn.cluster import KMeans

# Завантаження зображення
img = cv2.imread('coins_2.JPG')

# Перетворення зображення BGR в RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Розгортання зображення в одномірний масив
pixels = img_rgb.reshape((-1, 3)).astype(np.float32)

# Кількість кластерів (вартостей монет)
num_clusters = 3

# Використання k-середніх для кластеризації кольорів
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(pixels)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Створення маски для кожного кластеру
masks = [labels == i for i in range(num_clusters)]

# Зміна розміру масок
masks_resized = [mask.reshape(img_rgb.shape[:-1]) for mask in masks]

# Сегментація зображення за допомогою кольорових масок
segmented_image = np.zeros_like(img_rgb)
segmented_image2 = np.zeros_like(img_rgb)

# Різні кольори для кожної монети
cluster_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

for i in range(num_clusters):
    segmented_image[masks_resized[i]] = centers[i]
    segmented_image2[masks_resized[i]] = cluster_colors[i]

# Повернення вихідного розміру та BGR формату
segmented_image = cv2.cvtColor(segmented_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
segmented_image2 = cv2.cvtColor(segmented_image2.astype(np.uint8), cv2.COLOR_RGB2BGR)

# Конвертація в градації сірого кольору
gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

# Згладжування для зменшення шуму та підготовки до виявлення контурів
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Виявлення контурів
contours, hierarchy = cv2.findContours(blurred_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Фільтрація за розміром контуру (монети)
min_contour_area = 500  # Мінімальна площа контуру для розгляду

for contour in contours:
    contour_area = cv2.contourArea(contour)
    if contour_area > min_contour_area:
        # Виділення контуру на оригінальному зображенні
        cv2.drawContours(img_rgb, [contour], -1, (0, 255, 0), 2)

# Відображення результату
cv2.imshow('Segmented Image by Color 1', segmented_image)
cv2.imshow('Segmented Image by Color 2', segmented_image2)
cv2.imshow('Segmented Image by Color/Size 3', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()