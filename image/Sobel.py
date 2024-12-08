import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
image = cv2.imread('road.jpg', cv2.IMREAD_GRAYSCALE)  # 請將 'your_image.jpg' 替換為你的圖片路徑

# 使用 Sobel 進行邊緣檢測
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # 水平方向的梯度
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # 垂直方向的梯度

# 計算梯度的大小（可以理解為邊緣強度）
magnitude = cv2.magnitude(sobel_x, sobel_y)

# 將梯度大小轉換為8位元圖像（0到255）
magnitude = cv2.convertScaleAbs(magnitude)

# 使用 matplotlib 顯示圖片
plt.figure(figsize=(12, 6))

# 顯示 Magnitude

plt.imshow(magnitude, cmap='gray')
plt.title('Sobel')
plt.axis('off')

plt.tight_layout()
plt.show()
