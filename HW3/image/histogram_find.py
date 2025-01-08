import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    """
    計算影像的灰度直方圖。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist

def find_top_values(hist, top_n=3):
    """
    找出直方圖中頻率最高的前 N 個數值及其像素值。
    """
    top_values = np.argsort(hist[:, 0])[::-1][:top_n]  # 找到最高頻率對應的索引
    top_frequencies = hist[top_values, 0]             # 對應的頻率值
    return list(zip(top_values, top_frequencies))

def plot_histogram(hist):
    """
    繪製灰度直方圖。
    """
    plt.figure(figsize=(8, 6))
    plt.plot(hist, color='k', label='Grayscale Histogram')
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()
    plt.show()

# 載入影像
image_path = "Road.jpg"  # 修改為你的影像路徑
image = cv2.imread(image_path)

if image is None:
    print("無法載入影像，請檢查路徑！")
    exit()

# 計算灰度直方圖
hist = calculate_histogram(image)

# 找出前三大數值
top_values = find_top_values(hist, top_n=3)

# 顯示結果
print("灰度直方圖頻率前三高的值與對應像素值：")
for rank, (pixel_value, frequency) in enumerate(top_values, 1):
    print(f"第 {rank} 名：像素值 = {pixel_value}，頻率 = {frequency}")

# 繪製直方圖
plot_histogram(hist)
