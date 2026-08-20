# --- Import --- #
from sklearn.cluster import KMeans

# --- Main --- # 
def kmeans_recon(original_data, pcs):
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(pcs[:, 0:1]) # All rows, only first PC

    h, w = original_data.shape[0], original_data.shape[1]
    binary_img = np.zeros((h, w))