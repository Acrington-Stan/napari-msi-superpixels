<p align="center">
  <img width="350" alt="git_preview_img" src="https://github.com/user-attachments/assets/55e2978a-8bd6-4562-8a01-064e1442a817"/>
</p>

# napari-msi-superpixels 
Dedicated importer for Mass Spec Imaging (MSI) data to correct order for napari viewer else data will appear as a thin line and not actual image (data by default is x,y,channel - however, napari requires channel, x,y). Alongside this, the Skimage implementations of SLIC (Simple Linear Iterative Clustering) and Felzenszwalb's graph based segmentation are included as these two allow for the full hyperspectral cube that the importer constructs to be used in superpixel (SP) formation. Currently, the importer unit bins data to the nearest Dalton (Da). This is to say decimal values are rounded to the nearest whole integer (±1Da), as raw MSI data can be 180,000+ mass-to-charge (m/z) values, creating a severe computational bottleneck.

# Future Improvements 
Additional SP algorithms (e.g. quickshift, watershed, Linear Spectral Clustering (LSC)) will be added from Skimage and OpenCV, however, these cannot work with full hyperspectral cubes, so require dimsionality reduction techniques (e.g. Principle Component Analysis (PCA)) added alongside them. 

# Install - editable version (bash/powershell)

git clone https://github.com/Acrington-Stan/napari-msi-superpixels.git

cd napari-msi-superpixels

pip install -e .

Check requirements, the main one is napari. To install napari use "pip install napari[all]".

# TO DO
- Add Imzml parser to plugin.
- 
# Credit
napari plugin architecture corrections made with help from Claude Sonnet 5
