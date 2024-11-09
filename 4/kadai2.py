import cv2
import numpy as np

def move_img(img: np.ndarray, shift_x: int = 1, shift_y: int = 0, iterations: int = 100) -> np.ndarray:

    kernel = np.zeros((3, 3), dtype = np.float32)
    kernel[1 - shift_y, 1 - shift_x] = 1

    cv2.imwrite('./result.bmp', img)

    for _ in range(iterations):

        current_img = cv2.imread('./result.bmp')

        padded_img = np.pad(current_img, ((1, 1), (1, 1), (0, 0)), mode = 'constant', constant_values = 0)

        shifted_img = cv2.filter2D(padded_img, -1, kernel)

        shifted_img = shifted_img[1 : -1, 1 : -1]

        cv2.imwrite('./result.bmp', shifted_img)

    return shifted_img

path = '../img/lenna.bmp'
img = cv2.imread(path)
# cv2.imwrite('./lenna.jpg', img)

result = move_img(img, shift_x = -1, shift_y = -1, iterations = 100)
cv2.imshow('Shifted Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('./moving.jpg', result)

