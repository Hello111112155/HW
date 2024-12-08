import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

def extract_road_lbp(image_path, radius=3, n_points=24, threshold=0.5):
    """
    使用 LBP 提取馬路區域，馬路以白色顯示，其他部分以黑色顯示。

    :param image_path: 圖片路徑
    :param radius: LBP 半徑
    :param n_points: 圓周上的點數
    :param threshold: 二值化門檻 (0~1)
    :return: 馬路區域的二值化影像
    """
    # 讀取圖片
    image_path = "road.jpg"
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 計算 LBP
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')

    # 正規化 LBP 值
    lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min())

    # 二值化 LBP 影像
    _, road_mask = cv2.threshold((lbp_normalized * 255).astype(np.uint8), int(threshold * 255), 255, cv2.THRESH_BINARY)

    # 形態學操作：清理雜訊並平滑結果
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

    # 將馬路顯示為白色，背景為黑色
    road_binary = np.zeros_like(gray)
    road_binary[road_mask > 0] = 255

    # 顯示結果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("LBP Image")
    plt.imshow(lbp, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Road Extraction (Binary)")
    plt.imshow(road_binary, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return road_binary

# 主程式
if __name__ == "__main__":
    # 替換成你的圖片路徑
    image_path = "road_image.jpg"

    # 提取馬路區域
    road_binary = extract_road_lbp(image_path)
    # 將結果保存到檔案
    cv2.imwrite("road_binary_output.jpg", road_binary)
    print("馬路提取結果已保存為 road_binary_output.jpg")
