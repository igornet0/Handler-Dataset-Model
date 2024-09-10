import cv2
import pytesseract
import numpy as np
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Функция для коррекции гаммы
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

# Загружаем изображение
image_path = "1.jpg"
image = cv2.imread(image_path)

# Преобразуем изображение в градации серого
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Применяем коррекцию гаммы
gamma_corrected = adjust_gamma(gray, gamma=1.2)

# Применяем размытие для уменьшения шума
blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)

# Применяем адаптивное пороговое преобразование
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)

# Применяем морфологические операции для улучшения текста
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# Сохраняем промежуточное изображение, если нужно
cv2.imwrite('processed_image.jpg', morph)

# Используем pytesseract для распознавания текста
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(morph, config=custom_config, lang='rus+eng')

print("Распознанный текст:")
print(text)