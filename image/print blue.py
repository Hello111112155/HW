import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖片
image = cv2.imread('road.jpg')  # 確保圖片的路徑正確

# 轉換到灰度圖像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用邊緣檢測（Canny）來識別馬路
edges = cv2.Canny(gray, 100, 255)

# 找到馬路的輪廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 創建一個與原圖相同大小的全白圖片
output = image.copy()

# 假設馬路區域是較大的連續區域，我們可以根據這個假設選擇塗色
for contour in contours:
    if cv2.contourArea(contour) > 1000:  # 根據輪廓的大小過濾小區域
        cv2.drawContours(output, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)  # 用綠色塗色

# 使用matplotlib顯示圖片
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 轉換為RGB格式顯示
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))  # 轉換為RGB格式顯示
plt.title('Colored Road Area')
plt.axis('off')

plt.show()
