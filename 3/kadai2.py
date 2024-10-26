import cv2
import numpy as np
import matplotlib.pyplot as plt

def correct_brightness(img: np.ndarray, hist: np.ndarray) -> np.ndarray:
    # 累積分布関数(CDF)を計算
    cdfs = np.zeros(hist.size)
    cdfs[0] = hist[0]
    for i in range(1, cdfs.size):
        cdfs[i] = cdfs[i-1] + hist[i]

    # 出力画像を準備
    output_img = np.zeros(img.shape, dtype=np.uint8)

    # 明るさ補正処理
    img_flat = img.flatten()
    for i in range(img_flat.size):
        v = img_flat[i]
        output_img.flat[i] = np.round((cdfs[v] - cdfs[0]) * 255 / (img.size - cdfs[0]))

    return output_img

# ヒストグラムをプロット
def plot_histogram(image:np.ndarray, title:str):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

# 画像を読み込む
img = cv2.imread('./img/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# ヒストグラムを計算
hist, bins = np.histogram(img.flatten(), bins=256, range=[0,256])

# 明るさ補正を行う
output_img = correct_brightness(img, hist)

# 結果を表示
cv2.imshow("Original Image", img)
cv2.imshow("Brightness Corrected Image", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ヒストグラムを表示
plot_histogram(output_img, "Hist corrected brightness")
plt.savefig("hist_brightness_corrected.jpg", format='jpg')
plt.show()