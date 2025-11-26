import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def pca_restore(image, k):
    if k < 1:
        pca = PCA(n_components=k)
        pca.fit(image)
        compressed = pca.transform(image)
        image_restored = pca.inverse_transform(compressed)
        variance_ratio = pca.explained_variance_ratio_
        print("Variance Ratio for each component:")
        print(variance_ratio)
        cumulative_variance = np.cumsum(variance_ratio)
        print("Cumulative Variance:")
        print(cumulative_variance)
        components = pca.n_components_
        return image_restored, components
    else:
        pca = PCA(n_components=int(k))
        pca.fit(image)
        compressed = pca.transform(image)
        image_restored = pca.inverse_transform(compressed)
        return image_restored, int(k)

def full_pca(image):
    pca_full = PCA()
    pca_full.fit(image)
    explained_variance = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained_variance)
    return cumulative

def graph(components, cumulative, achieved):
    plt.figure()
    plt.plot(components, cumulative)
    plt.xlabel("Principal components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Cumulative Explained Variance explained by the components")
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axvline(x=achieved, color='k', linestyle='--')

image_raw = imread("images/ala_photo.jpg")
print(image_raw.shape)
image_sum = image_raw.sum(axis=2)
print(image_sum.shape)
image_bw = image_sum/image_sum.max()
print(image_bw.max())
image_restored, components = pca_restore(image_bw, 0.95)
image_restored1, comp1 = pca_restore(image_bw, 329)
image_restored2, comp2 = pca_restore(image_bw, 81)
image_restored3, comp3 = pca_restore(image_bw, 17)
cumulative = full_pca(image_bw)
comp_num = np.arange(1, len(cumulative) + 1)
graph(comp_num, cumulative, components)

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes[0, 0].imshow(image_raw)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis('off')
axes[0, 1].imshow(image_bw, cmap='gray')
axes[0, 1].set_title("B&W")
axes[0, 1].axis('off')
axes[0, 2].imshow(image_restored, cmap='gray')
axes[0, 2].set_title("PCA\n95% coverage")
axes[0, 2].axis('off')
axes[1, 0].imshow(image_restored1, cmap='gray')
axes[1, 0].set_title("PCA\n329 components")
axes[1, 0].axis('off')
axes[1, 1].imshow(image_restored2, cmap='gray')
axes[1, 1].set_title("PCA\n81 components")
axes[1, 1].axis('off')
axes[1, 2].imshow(image_restored3, cmap='gray')
axes[1, 2].set_title("PCA\n17 components")
axes[1, 2].axis('off')
plt.tight_layout()
plt.show()