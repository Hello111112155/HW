import cv2
import numpy as np

# 讀取圖片
image = cv2.imread("road.jpg")
# 替換為您的圖片路徑
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉為灰階圖

# 使用高斯模糊來減少雜訊
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用 Canny 邊緣檢測來檢測圖像中的邊緣
edges = cv2.Canny(blurred, 50, 150)

# 定義馬路區域的範圍（顏色篩選、輪廓篩選等，根據實際情況調整）
# 這裡假設馬路的顏色較暗，可以使用閾值方法來篩選
_, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

# 輪廓檢測
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原圖上畫出找到的輪廓
for contour in contours:
    # 如果輪廓面積較大，則認為是馬路區域
    if cv2.contourArea(contour) > 500:  # 這個數字可以根據需要調整
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)

# 顯示結果
cv2.imshow("Detected Road", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
