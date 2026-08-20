# --- Package Import --- #
import umap 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- Reduction --- #
def get_PCA(original_data, flat_data, region_df, algorithm, num_comps=3):
    scaler = StandardScaler()
    f_data = scaler.fit_transform(flat_data)

    pca = PCA(n_components=num_comps)
    pcs = pca.fit_transform(f_data) # Components 
    var_ratios = pca.explained_variance_ratio_ * 100 # For axes 

    # Need to embed in napari
    # Check correct for PCA
    plt.scatter(
    pcs[:, 0],
    pcs[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('PCA projection of hyperspectral cube', fontsize=10)

    return var_ratios, pcs

def get_UMAP(original_data, flat_data, metric='cosine'):
    reducer = umap.UMAP(n_components=3, metric=metric, random_state=42)#, densmap=True)
    embedding = reducer.fit_transform(flat_data)
 # add scatter 
    h, w = original_data.shape[0], original_data.shape[1]
    umap_cube = embedding.reshape(h, w, 3) # need to check this with either 2 or 3 i think

    # 4. Global min-max normalization across the whole cube for RGB display
    umap_norm = (umap_cube - umap_cube.min()) / (umap_cube.max() - umap_cube.min())

    plt.scatter(
    embedding[:, 0],
    embedding[:, 1])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of hyperspectral cube', fontsize=10)

    return umap_norm

# --- Reconstruction --- # 
def kmeans_recon(original_data, region_df):
    # 2. KMeans
    kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(pcs[:, 0:1])

    # 3. Build Binary Image
    # Always base the canvas on the spatial data passed in
    h, w = original_data.shape[0], original_data.shape[1]
    binary_img = np.zeros((h, w))

    if region_df is not None:
        # SUPERPIXEL MAPPING
        for i, (idx, row) in enumerate(region_df.iterrows()):
            c = row['coords']
            # We use the coordinates to fill the 2D map
            binary_img[c[:, 0], c[:, 1]] = clusters[i]
    else:
        # PIXEL MAPPING
        # reshape only works if clusters.size == h * w
        try:
            binary_img = clusters.reshape(h, w)
        except ValueError as e:
            print(f"Reshape Error: Cluster size {clusters.size} does not match {h}x{w} grid.")
            raise e

    return binary_img 