
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取影像，並將其轉換為灰階
image = cv2.imread('road.jpg', cv2.IMREAD_GRAYSCALE)

# 檢查影像是否成功讀取
if image is None:
    print("Error: Could not read image.")
else:
    # 計算影像的直方圖
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))

    # 繪製直方圖
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])  # 0-255像素值範圍
    plt.plot(bin_edges[0:-1], histogram)  # 不包含最後一個邊緣
    plt.show()
