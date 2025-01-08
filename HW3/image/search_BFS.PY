import cv2
import numpy as np
from collections import deque

def bfs_segmentation(image, threshold=(0, 50)):
    """
    使用廣度優先搜尋 (BFS) 將影像中符合條件的區域分割。
    
    參數:
    - image: 輸入影像 (灰階影像)
    - threshold: 像素值範圍 (min_val, max_val)
    
    回傳:
    - mask: 分割區域的二值化遮罩
    """
    rows, cols = image.shape
    visited = np.zeros((rows, cols), dtype=bool)  # 訪問過的像素
    mask = np.zeros((rows, cols), dtype=np.uint8)  # 分割結果

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右

    for r in range(rows):
        for c in range(cols):
            if visited[r, c] or not (threshold[0] <= image[r, c] <= threshold[1]):
                continue

            # BFS 初始化
            queue = deque([(r, c)])
            visited[r, c] = True
            region_pixels = []

            while queue:
                x, y = queue.popleft()
                region_pixels.append((x, y))
                mask[x, y] = 255

                for dx, dy in directions:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < rows and 0 <= ny < cols and not visited[nx, ny]:
                        if threshold[0] <= image[nx, ny] <= threshold[1]:
                            queue.append((nx, ny))
                            visited[nx, ny] = True

    return mask

# 載入影像
image_path = "road.jpg"  # 修改為你的影像路徑
original_image = cv2.imread(image_path)

if original_image is None:
    print("無法載入影像，請檢查路徑！")
    exit()

# 轉為灰階
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# 設定馬路部分的像素值範圍（自行調整）
threshold = (70, 120)  # 假設馬路是較暗的灰階值

# 使用 BFS 進行分割
segmented_mask = bfs_segmentation(gray_image, threshold)

# 顯示結果
cv2.imshow("Original Image", original_image)
cv2.imshow("Segmented Mask", segmented_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 儲存結果
cv2.imwrite("segmented_mask.jpg", segmented_mask)
print("分割結果已儲存為 segmented_mask.jpg")
