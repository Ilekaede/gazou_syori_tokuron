import cv2
import numpy as np
import matplotlib.pyplot as plt

# 画像を読み込んでグレースケールに変換
input_img = cv2.imread('./img/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# ヒストグラムをプロット
def plot_histogram(img:np.ndarray, title:str):
    plt.hist(img.ravel(), bins=256, range=(0, 256), color='black')
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

plot_histogram(input_img, "Histgram of value frequency")
plt.savefig("./img/hist.jpg", format='jpg')
plt.show()