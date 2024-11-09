import cv2
import numpy as np

def extract_edge(img: np.ndarray) -> np.ndarray:
    # ラプラシアンフィルタによるエッジ検出
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    
    edge_img = cv2.filter2D(img_gray, cv2.CV_64F, kernel)
    ret = np.uint8(np.abs(edge_img))
    
    return ret

path = '../img/lenna.bmp'
img = cv2.imread(path)
edge_img = extract_edge(img)
edge_color_img = cv2.cvtColor(edge_img, cv2.COLOR_GRAY2BGR)
cv2.imshow('Edge Image', edge_color_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('./laplacian.jpg', edge_color_img)


