import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# 參數設定
LBP_RADIUS = 3  # 半徑
LBP_POINTS = 8 * LBP_RADIUS  # 鄰居點數
THRESHOLD_LOW = 70  # 馬路部分的灰度下限
THRESHOLD_HIGH = 100  # 馬路部分的灰度上限

def apply_lbp(image):
    """
    對影像進行 LBP 處理並提取特徵。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    return lbp

def extract_road_region(image, threshold_low, threshold_high):
    """
    提取影像中馬路部分。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold_low, threshold_high, cv2.THRESH_BINARY)
    return binary

# 載入影像
image_path = "Road.jpg"  # 請根據你的需求更改路徑
image = cv2.imread(image_path)

if image is None:
    print("無法載入影像，請檢查路徑！")
    exit()

# LBP 處理
lbp_result = apply_lbp(image)

# 提取馬路部分
road_mask = extract_road_region(image, THRESHOLD_LOW, THRESHOLD_HIGH)

# 使用馬路遮罩過濾 LBP 結果
road_lbp = cv2.bitwise_and(lbp_result.astype(np.uint8), lbp_result.astype(np.uint8), mask=road_mask)

# 顯示結果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1, 3, 3)
plt.title("Road LBP Region")
plt.imshow(road_lbp, cmap="gray")
cv2.imwrite("LBP.jpg", road_lbp)

plt.tight_layout()
plt.show()
