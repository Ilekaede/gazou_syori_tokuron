import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans_clustering(img: np.ndarray, k: int) -> np.ndarray:

    # 画像をBGR->RGBに変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_flat = img.reshape((-1, 3))
    # KMeansクラスタリング
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
    kmeans.fit(img_flat)
    clustered = kmeans.cluster_centers_[kmeans.labels_]  # 各クラスタの中心色を取得
    labels = kmeans.labels_

    # クラスタリング結果を元の形状に戻す
    clustered_img = clustered.reshape(img.shape).astype(np.uint8)
    
    # もしクラスタリング結果を距離の遠い色同士で分類したいなら，上のclustered_imgをコメントアウト，下の2つのコメントアウトを外す
    # color_map = (plt.get_cmap('tab20', k)(np.arange(k))[:, :3] * 255).astype(np.uint8)
    # clustered_img = color_map[labels].reshape(img.shape).astype(np.uint8)
    
    return clustered_img

path = '../img/lenna.bmp'
img = cv2.imread(path)
clustered_img = kmeans_clustering(img, k=5)
clustered_img = cv2.cvtColor(clustered_img, cv2.COLOR_RGB2BGR)

cv2.imshow("clustered_img", clustered_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# cv2.imwrite('../5/kmeans_5.jpg', clustered_img)