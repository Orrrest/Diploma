import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import wiener


# Функція для додавання шуму Релея
def add_rayleigh_noise(image, scale=0.1):
    row, col, ch = image.shape
    rayleigh_noise = np.random.rayleigh(scale, (row, col, ch))
    noisy_image = image + rayleigh_noise * 255
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


# Реалізація фільтра Вінера
def wiener_filter(image, kernel_size=5, K=0.01):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
    if len(image.shape) == 3:
        restored_image = np.zeros_like(image, dtype=np.float32)
        for i in range(3):
            restored_image[:, :, i] = wiener(image[:, :, i].astype(np.float32), kernel, balance=K, clip=False)
        return np.clip(restored_image, 0, 255).astype(np.uint8)
    else:
        return np.clip(wiener(image.astype(np.float32), kernel, balance=K, clip=False), 0, 255).astype(np.uint8)


# Медіанний фільтр
def median_filter_color(image, kernel_size=3):
    if len(image.shape) == 3:
        restored_image = np.zeros_like(image)
        for i in range(3):
            restored_image[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
        return restored_image
    else:
        return cv2.medianBlur(image, kernel_size)


# Обчислення гістограми та середнього значення
def compute_histogram_and_mean(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram / np.sum(histogram)
    mean = sum(i * histogram[i] for i in range(256))
    return histogram, mean.item()


# Обчислення моментів
def compute_moments(hist, mean):
    second_moment = sum(((i - mean) ** 2) * hist[i] for i in range(256))
    third_moment = sum(((i - mean) ** 3) * hist[i] for i in range(256))
    fourth_moment = sum(((i - mean) ** 4) * hist[i] for i in range(256))
    fifth_moment = sum(((i - mean) ** 5) * hist[i] for i in range(256))
    sixth_moment = sum(((i - mean) ** 6) * hist[i] for i in range(256))  # Додаємо шостий момент
    return (second_moment.item(), third_moment.item(), fourth_moment.item(), fifth_moment.item(), sixth_moment.item())  # Додаємо шостий момент до виходу


# Відображення зображень і гістограм
def show_images_and_histograms(original, noisy, restored_wiener, restored_median):
    fig, axs = plt.subplots(2, 4, figsize=(22, 10))

    # Відображення зображень
    titles = ["Оригінальне", "З шумом Релея", "Відновлене (Вінер)", "Відновлене (медіанне)"]
    images = [original, noisy, restored_wiener, restored_median]
    for i, (img, title) in enumerate(zip(images, titles)):
        axs[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0, i].set_title(title)
        axs[0, i].axis('off')
        axs[1, i].hist(img.ravel(), 256, [0, 256])
        axs[1, i].set_title(f"Гістограма {title}")

    plt.tight_layout()
    plt.show()


# Основна функція запуску експерименту
def run_experiments(image, K=0.01):
    noisy_image = add_rayleigh_noise(image, scale=0.1)
    restored_image_wiener = wiener_filter(noisy_image, kernel_size=5, K=K)
    restored_image_median = median_filter_color(noisy_image, kernel_size=3)

    # Обчислення гістограм і моментів
    hist_original, mean_original = compute_histogram_and_mean(image)
    hist_noisy, mean_noisy = compute_histogram_and_mean(noisy_image)
    hist_restored_wiener, mean_restored_wiener = compute_histogram_and_mean(restored_image_wiener)
    hist_restored_median, mean_restored_median = compute_histogram_and_mean(restored_image_median)

    moments_original = compute_moments(hist_original, mean_original)
    moments_noisy = compute_moments(hist_noisy, mean_noisy)
    moments_restored_wiener = compute_moments(hist_restored_wiener, mean_restored_wiener)
    moments_restored_median = compute_moments(hist_restored_median, mean_restored_median)

    # Відображення результатів
    show_images_and_histograms(image, noisy_image, restored_image_wiener, restored_image_median)

    # Виведення моментів
    print(f"Другий момент (оригінал): {moments_original[0]}")
    print(f"Третій момент (оригінал): {moments_original[1]}")
    print(f"Четвертий момент (оригінал): {moments_original[2]}")
    print(f"П'ятий момент (оригінал): {moments_original[3]}")
    print(f"Шостий момент (оригінал): {moments_original[4]}")  # Виведення шостого моменту

    print(f"Другий момент (шум): {moments_noisy[0]}")
    print(f"Третій момент (шум): {moments_noisy[1]}")
    print(f"Четвертий момент (шум): {moments_noisy[2]}")
    print(f"П'ятий момент (шум): {moments_noisy[3]}")
    print(f"Шостий момент (шум): {moments_noisy[4]}")  # Виведення шостого моменту

    print(f"Другий момент (Вінер): {moments_restored_wiener[0]}")
    print(f"Третій момент (Вінер): {moments_restored_wiener[1]}")
    print(f"Четвертий момент (Вінер): {moments_restored_wiener[2]}")
    print(f"П'ятий момент (Вінер): {moments_restored_wiener[3]}")
    print(f"Шостий момент (Вінер): {moments_restored_wiener[4]}")  # Виведення шостого моменту

    print(f"Другий момент (медіанний): {moments_restored_median[0]}")
    print(f"Третій момент (медіанний): {moments_restored_median[1]}")
    print(f"Четвертий момент (медіанний): {moments_restored_median[2]}")
    print(f"П'ятий момент (медіанний): {moments_restored_median[3]}")
    print(f"Шостий момент (медіанний): {moments_restored_median[4]}")  # Виведення шостого моменту
    print()


# Зчитування зображення
image = cv2.imread('image4.jpg')
run_experiments(image, K=7)
