import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import wiener

# Функція для додавання Гауссового шуму
def add_gaussian_noise(image, mean=0, var=0.01):
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss * 255  # Масштабування шуму
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Функція для додавання шуму "сіль та перець"
def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = np.copy(image)
    total_pixels = image.size

    # Додавання "солі" (білих пікселів)
    num_salt = np.ceil(salt_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255

    # Додавання "перцю" (чорних пікселів)
    num_pepper = np.ceil(pepper_prob * total_pixels)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image

# Реалізація фільтра Вінера
def wiener_filter(image, kernel_size=5, K=0.01):
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)  # Вирівнююче ядро
    if len(image.shape) == 3:  # Якщо кольорове зображення
        restored_image = np.zeros_like(image, dtype=np.float32)  # Використовуємо float32 для точності
        for i in range(3):  # Фільтрація для кожного каналу
            restored_image[:, :, i] = wiener(image[:, :, i].astype(np.float32), kernel, balance=K, clip=False)
        return np.clip(restored_image, 0, 255).astype(np.uint8)  # Конвертація назад у uint8
    else:
        return np.clip(wiener(image.astype(np.float32), kernel, balance=K, clip=False), 0, 255).astype(np.uint8)

# Реалізація медіанного фільтра для кольорових зображень
def median_filter_color(image, kernel_size=3):
    if len(image.shape) == 3:  # Якщо кольорове зображення
        restored_image = np.zeros_like(image)  # Створення порожнього зображення для відновлення
        for i in range(3):  # Фільтрація для кожного каналу
            restored_image[:, :, i] = cv2.medianBlur(image[:, :, i], kernel_size)
        return restored_image
    else:
        return cv2.medianBlur(image, kernel_size)  # Для чорно-білого зображення

# Функція для обчислення гістограми
def compute_histogram(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    histogram = histogram / np.sum(histogram)  # Нормалізація гістограми (ймовірність появи рівня інтенсивності)
    return histogram

# Функція для обчислення середнього значення з гістограми
def compute_mean(hist):
    mean = 0
    for i in range(len(hist)):
        mean += i * hist[i]
    return mean.item()  # Повертаємо єдине значення

# Функція для обчислення другого моменту (дисперсії)
def compute_second_moment(hist, mean):
    second_moment = 0
    for i in range(len(hist)):
        second_moment += ((i - mean) ** 2) * hist[i]
    return second_moment.item()  # Повертаємо єдине значення

# Функція для обчислення третього моменту
def compute_third_moment(hist, mean):
    third_moment = 0
    for i in range(len(hist)):
        third_moment += ((i - mean) ** 3) * hist[i]
    return third_moment.item()  # Повертаємо єдине значення

# Функція для обчислення четвертого моменту
def compute_fourth_moment(hist, mean):
    fourth_moment = 0
    for i in range(len(hist)):
        fourth_moment += ((i - mean) ** 4) * hist[i]
    return fourth_moment.item()  # Повертаємо єдине значення

# Функція для обчислення п'ятого моменту
def compute_fifth_moment(hist, mean):
    fifth_moment = 0
    for i in range(len(hist)):
        fifth_moment += ((i - mean) ** 5) * hist[i]
    return fifth_moment.item()  # Повертаємо єдине значення

# Функція для обчислення шостого моменту
def compute_sixth_moment(hist, mean):
    sixth_moment = 0
    for i in range(len(hist)):
        sixth_moment += ((i - mean) ** 6) * hist[i]
    return sixth_moment.item()  # Повертаємо єдине значення

# Функція для виведення зображень та гістограм
def show_images_and_histograms(original, noisy, restored_wiener, restored_median, title1="Original", title2="Noisy",
                               title3="Wiener Restored", title4="Median Restored"):
    fig, axs = plt.subplots(2, 4, figsize=(22, 10))

    # Виведення зображень
    axs[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(title1)
    axs[0, 0].axis('off')

    axs[0, 1].imshow(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title(title2)
    axs[0, 1].axis('off')

    axs[0, 2].imshow(cv2.cvtColor(restored_wiener, cv2.COLOR_BGR2RGB))
    axs[0, 2].set_title(title3)
    axs[0, 2].axis('off')

    axs[0, 3].imshow(cv2.cvtColor(restored_median, cv2.COLOR_BGR2RGB))
    axs[0, 3].set_title(title4)
    axs[0, 3].axis('off')

    # Виведення гістограм
    axs[1, 0].hist(original.ravel(), 256, [0, 256])
    axs[1, 0].set_title(f"Гістограма {title1}")
    axs[1, 0].set_xlabel('Інтенсивність')
    axs[1, 0].set_ylabel('Кількість пікселів')

    axs[1, 1].hist(noisy.ravel(), 256, [0, 256])
    axs[1, 1].set_title(f"Гістограма {title2}")
    axs[1, 1].set_xlabel('Інтенсивність')
    axs[1, 1].set_ylabel('Кількість пікселів')

    axs[1, 2].hist(restored_wiener.ravel(), 256, [0, 256])
    axs[1, 2].set_title(f"Гістограма {title3}")
    axs[1, 2].set_xlabel('Інтенсивність')
    axs[1, 2].set_ylabel('Кількість пікселів')

    axs[1, 3].hist(restored_median.ravel(), 256, [0, 256])
    axs[1, 3].set_title(f"Гістограма {title4}")
    axs[1, 3].set_xlabel('Інтенсивність')
    axs[1, 3].set_ylabel('Кількість пікселів')

    plt.tight_layout()
    plt.show()

# Функція для проведення експериментів з обчисленням моментів
def run_experiments(image, noise_type='gaussian', K=0.01):
    # Додавання шуму
    if noise_type == 'gaussian':
        var = 0.01  # Використання K як дисперсії
        noisy_image = add_gaussian_noise(image, var=var)
        noise_desc = "Гауссовий шум"
    elif noise_type == 'salt_and_pepper':
        salt_prob = 0.01
        pepper_prob = 0
        pepper_prob = 0.01
        noisy_image = add_salt_pepper_noise(image, salt_prob=salt_prob, pepper_prob=pepper_prob)
        noise_desc = "Сіль та перець"

    # Відновлення зображення
    restored_wiener = wiener_filter(noisy_image, K=K)
    restored_median = median_filter_color(noisy_image)

    # Обчислення гістограм
    hist_original = compute_histogram(image)
    hist_noisy = compute_histogram(noisy_image)
    hist_wiener = compute_histogram(restored_wiener)
    hist_median = compute_histogram(restored_median)

    # Обчислення моментів
    mean_original = compute_mean(hist_original)
    mean_noisy = compute_mean(hist_noisy)
    mean_wiener = compute_mean(hist_wiener)
    mean_median = compute_mean(hist_median)

    second_moment_original = compute_second_moment(hist_original, mean_original)
    second_moment_noisy = compute_second_moment(hist_noisy, mean_noisy)
    second_moment_wiener = compute_second_moment(hist_wiener, mean_wiener)
    second_moment_median = compute_second_moment(hist_median, mean_median)

    third_moment_original = compute_third_moment(hist_original, mean_original)
    third_moment_noisy = compute_third_moment(hist_noisy, mean_noisy)
    third_moment_wiener = compute_third_moment(hist_wiener, mean_wiener)
    third_moment_median = compute_third_moment(hist_median, mean_median)

    fourth_moment_original = compute_fourth_moment(hist_original, mean_original)
    fourth_moment_noisy = compute_fourth_moment(hist_noisy, mean_noisy)
    fourth_moment_wiener = compute_fourth_moment(hist_wiener, mean_wiener)
    fourth_moment_median = compute_fourth_moment(hist_median, mean_median)

    fifth_moment_original = compute_fifth_moment(hist_original, mean_original)
    fifth_moment_noisy = compute_fifth_moment(hist_noisy, mean_noisy)
    fifth_moment_wiener = compute_fifth_moment(hist_wiener, mean_wiener)
    fifth_moment_median = compute_fifth_moment(hist_median, mean_median)

    sixth_moment_original = compute_sixth_moment(hist_original, mean_original)
    sixth_moment_noisy = compute_sixth_moment(hist_noisy, mean_noisy)
    sixth_moment_wiener = compute_sixth_moment(hist_wiener, mean_wiener)
    sixth_moment_median = compute_sixth_moment(hist_median, mean_median)

    # Виведення зображень та гістограм
    show_images_and_histograms(image, noisy_image, restored_wiener, restored_median,
                               title1="Оригінал", title2=noise_desc,
                               title3="Відновлене Вінера", title4="Медіанне відновлення")

    # Виведення моментів
    print(f"Моменти для {noise_desc}:")
    print(f"  Оригінал: 2-й: {second_moment_original:.2f}, 3-й: {third_moment_original:.2f}, 4-й: {fourth_moment_original:.2f}, 5-й: {fifth_moment_original:.2f}, 6-й: {sixth_moment_original:.2f}")
    print(f"  Шум:      2-й: {second_moment_noisy:.2f}, 3-й: {third_moment_noisy:.2f}, 4-й: {fourth_moment_noisy:.2f}, 5-й: {fifth_moment_noisy:.2f}, 6-й: {sixth_moment_noisy:.2f}")
    print(f"  Вінера:   2-й: {second_moment_wiener:.2f}, 3-й: {third_moment_wiener:.2f}, 4-й: {fourth_moment_wiener:.2f}, 5-й: {fifth_moment_wiener:.2f}, 6-й: {sixth_moment_wiener:.2f}")
    print(f"  Медіанне: 2-й: {second_moment_median:.2f}, 3-й: {third_moment_median:.2f}, 4-й: {fourth_moment_median:.2f}, 5-й: {fifth_moment_median:.2f}, 6-й: {sixth_moment_median:.2f}")

# Приклад використання
if __name__ == "__main__":
    image = cv2.imread("image4.jpg")  #  Шлях до зображення

    # Запуск експериментів з Гауссовим шумом
    run_experiments(image, noise_type='gaussian', K=0.2)

    # Запуск експериментів з шумом "сіль та перець"
    run_experiments(image, noise_type='salt_and_pepper', K=2)
