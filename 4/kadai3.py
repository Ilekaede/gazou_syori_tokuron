import cv2
import numpy as np

def emboss_filter(img: np.ndarray) -> np.ndarray:

    kernel = np.array([[-2, -1, 0],
                       [-1,  1, 1],
                       [ 0,  1, 2]], dtype=np.float32)
    
    # フィルタリング処理
    embossed_img = cv2.filter2D(img, -1, kernel)
    
    return embossed_img

path = '../img/lenna.bmp'
img = cv2.imread(path)

embossed_img = emboss_filter(img)

cv2.imshow('Emboss Effect', embossed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('emboss.jpg', embossed_img)
