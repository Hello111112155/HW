import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from skimage.feature import local_binary_pattern

image_path = "road.jpg"

# 1. HSV範圍提取
def extract_hsv_range(image_path, lower_hsv, upper_hsv):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"找不到圖片：{image_path}")
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


# 2. Sobel邊緣檢測
def sobel_edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        return None
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = cv2.convertScaleAbs(magnitude)
    return magnitude


# 3. BFS填充
def bfs_fill(image_path, start_x, start_y, target_color=255, fill_color=128):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    queue = deque([(start_x, start_y)])
    visited = np.zeros_like(image, dtype=bool)
    visited[start_x, start_y] = True
    while queue:
        x, y = queue.popleft()
        image[x, y] = fill_color
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                if not visited[nx, ny] and image[nx, ny] == target_color:
                    queue.append((nx, ny))
                    visited[nx, ny] = True
    return image


# 4. LBP馬路區域提取
def extract_road_lbp(image_path, radius=3, n_points=24, threshold=0.5):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min())
    _, road_mask = cv2.threshold((lbp_normalized * 255).astype(np.uint8), int(threshold * 255), 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    
    road_binary = np.zeros_like(gray)
    road_binary[road_mask > 0] = 255
    
    return road_binary


# 5. 顯示直方圖
def show_histogram(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Could not read image.")
        return
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.plot(bin_edges[0:-1], histogram)
    plt.show()


# 6. 替換灰色區域為藍色
def replace_gray_with_blue(image_path):
    image = cv2.imread(image_path)
    lower_gray = np.array([0, 0, 0])  # 灰色下邊界
    upper_gray = np.array([75, 75, 75])  # 灰色上邊界
    mask = cv2.inRange(image, lower_gray, upper_gray)
    image[mask > 0] = [255, 0, 0]  # 替換為藍色
    return image


# 主程式：依序執行各個步驟
if __name__ == "__main__":
    image_path = 'road.jpg'  # 替換為你的圖片路徑

    # 1. 使用 HSV 提取
    hsv_result = extract_hsv_range(image_path, (0, 0, 0), (255, 255, 255))

    # 2. Sobel邊緣檢測
    sobel_result = sobel_edge_detection(image_path)

    # 3. BFS 填充
    bfs_result = bfs_fill(image_path, 100, 100)

    # 4. LBP 提取馬路
    road_binary = extract_road_lbp(image_path)

    # 5. 顯示直方圖
    show_histogram(image_path)

    # 6. 替換灰色為藍色
    final_result = replace_gray_with_blue(image_path)

    # 顯示結果
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(hsv_result, cv2.COLOR_BGR2RGB))
    plt.title("HSV")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(sobel_result, cmap='gray')
    plt.title("Sobel")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(bfs_result, cmap='gray')
    plt.title("search")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(road_binary, cmap='gray')
    plt.title("LBP")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    plt.title("Final Image with Blue")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # 保存最終圖片
    cv2.imwrite('final_result_with_blue.jpg', final_result)
    print("Final image with blue areas saved as final_result_with_blue.jpg")
