import cv2
import numpy as np
import matplotlib.pyplot as plt

def extract_hsv_range(image_path, lower_hsv, upper_hsv):
    """
    使用 HSV 色彩空間提取指定範圍內的區域。

    :param image_path: 圖片路徑
    :param lower_hsv: HSV 下限值 (tuple，例如 (0, 50, 50))
    :param upper_hsv: HSV 上限值 (tuple，例如 (10, 255, 255))
    :return: 篩選出的二值化影像
    """
    # 讀取圖片
    image_path = "road.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"找不到圖片：{image_path}")

    # 將圖片轉換為 HSV 色彩空間
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 篩選指定 HSV 範圍內的區域
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    # 形態學操作：清理小雜訊
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 遮罩應用到原圖，提取指定區域
    result = cv2.bitwise_and(image, image, mask=mask)

    # 顯示原圖與結果
    plt.figure(figsize=(12, 6))


 
    plt.title("HSV")
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return mask

# 主程式
if __name__ == "__main__":
    # 替換為你的圖片路徑
    image_path = "road.jpg"

    # HSV 範圍 (例如：提取白色區域)
    lower_hsv = (0, 0, 0)      # 下限值 (H, S, V)
    upper_hsv = (255, 255, 255)   # 上限值 (H, S, V)

    try:
        # 提取指定 HSV 範圍內的區域
        mask = extract_hsv_range(image_path, lower_hsv, upper_hsv)
        # 保存二值化結果
        cv2.imwrite("hsv_mask_output.jpg", mask)
        print("HSV 範圍提取結果已保存為 hsv_mask_output.jpg")
    except FileNotFoundError as e:
        print(e)
