import cv2
import numpy as np
from collections import deque

# 讀取圖片
image = cv2.imread('road.jpg', cv2.IMREAD_GRAYSCALE)

# 設定起始點（假設是道路的某個點）
start_x, start_y = 100, 100  # 請自行設定初始點

# BFS 需要的佇列
queue = deque([(start_x, start_y)])
visited = np.zeros_like(image, dtype=bool)  # 記錄哪些像素已訪問
visited[start_x, start_y] = True

# 定義顏色填充的條件，例如目標區域為白色（道路區域）
target_color = 255
fill_color = 128  # 要填充的顏色，例如灰色

# BFS
while queue:
    x, y = queue.popleft()
    
    # 填充顏色
    image[x, y] = fill_color
    
    # 取得四個方向的相鄰像素
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        
        # 檢查是否在邊界內並符合條件
        if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
            if not visited[nx, ny] and image[nx, ny] == target_color:
                queue.append((nx, ny))
                visited[nx, ny] = True

# 顯示結果
cv2.imshow('search', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
