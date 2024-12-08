import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取圖像
image = cv2.imread('road.jpg')  # 請確保圖像的路徑是正確的

# 定義灰色的範圍
lower_gray = np.array([0, 0, 0])  # 灰色的下邊界 (R, G, B)
upper_gray = np.array([70, 70, 70])  # 灰色的上邊界 (R, G, B)

# 創建一個掩模來檢測灰色區域
mask = cv2.inRange(image, lower_gray, upper_gray)

# 把灰色區域替換為藍色 (BGR格式)
image[mask > 0] = [255, 0, 0]  # 替換為藍色，BGR格式是 [藍色, 綠色, 紅色]

# 使用matplotlib顯示圖片
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # OpenCV默認為BGR，需要轉換為RGB
plt.title('Gray Areas Replaced with Blue')
plt.axis('off')  # 不顯示坐標軸
plt.show()

# 或者，你可以保存修改後的圖像
cv2.imwrite('image_with_blue.jpg', image)  # 保存圖片
print("Image saved as image_with_blue.jpg")
