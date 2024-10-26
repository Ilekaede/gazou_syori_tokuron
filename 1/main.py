import cv2

# 画像の読み込み
img = cv2.imread('./img/lenna.bmp', cv2.IMREAD_ANYCOLOR)

# 画像の書き込み
cv2.imwrite('./img/output_img.bmp', img)
