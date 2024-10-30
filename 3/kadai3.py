import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_histogram(image:np.ndarray, title:str):
    plt.hist(image.ravel(), bins=256, range=(0, 256), color='black')
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

def flatten_histogram(img:np.ndarray)->np.ndarray:
    # 画素値とその位置を取得し、リストに格納
    flat_image = img.flatten()
    pixel_indices = list(range(len(flat_image)))
    pixel_value_pairs = list(zip(flat_image, pixel_indices))

    # 画素値で昇順ソート
    pixel_value_pairs.sort()

    # 全画素数/256個の画素を1単位とする256個のセクションを作成
    total_pixels = len(flat_image)
    pixels_per_section = total_pixels // 256
    sections = [pixel_value_pairs[i * pixels_per_section: (i + 1) * pixels_per_section] for i in range(256)]

    # 各セクションに入った画素の位置をセクション番号に設定
    flattened_img = np.zeros_like(flat_image, dtype=np.uint8)
    for section_number, section in enumerate(sections):
        for _, idx in section:
            flattened_img[idx] = section_number

    # 元の形状に戻す
    return flattened_img.reshape(img.shape)

# 画像を読み込んでグレースケールに変換
input_image = cv2.imread('../img/lenna.bmp', cv2.IMREAD_GRAYSCALE)

# ヒストグラムをフラットにする
output_image = flatten_histogram(input_image)

# 画素値の頻度を計算
hist, bins = np.histogram(output_image.flatten(), bins=256, range=[0,256])

# 結果を表示
for value, freq in enumerate(hist):
    print(f"Pixel Value {value}: Frequency {freq}")

# オリジナルとフラット化画像を表示
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title("Original Image")

plt.subplot(2, 2, 2)
plot_histogram(input_image, "Histgram of value frequency")

plt.subplot(2, 2, 3)
plt.imshow(output_image, cmap='gray')
plt.title("Flattened Histogram Image")
# 画像を保存したい場合
# cv2.imwrite('./lenna_flat_hist.jpg', output_image)

plt.subplot(2, 2, 4)
plot_histogram(output_image, "Histogram of Flattened")
# グラフを保存したい場合
# plt.savefig("output_histogram_flatten.jpg", format='jpg')
plt.show()