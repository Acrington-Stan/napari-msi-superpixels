# --- Import --- #
from nexusformat.nexus import NXroot, NXentry, NXprocess, NXdata, NXfield 
import numpy as np

# --- Main --- # 
def superpixel_meta_save(hsc, props, segments, file_name, working_direct): # props = Regionprops 
    #if not file_name.endswith(".nxs"):
    #    file_name += ".nxs"

    # - Turn each region prop into array - #
    sp_ids = np.array([p.label for p in props], dtype=np.int32)
    areas = np.array([p.area for p in props], dtype=np.int32)
    centroids = np.array([p.centroid for p in props], dtype=np.float32)  
    mean_intensities = np.array([p.intensity_mean for p in props], dtype=np.float32)
    eccentricities = np.array([p.eccentricity for p in props], dtype=np.float32)
    perimeters = np.array([p.perimeter for p in props], dtype=np.float32)
    std_intensities = np.array([p.intensity_std for p in props], dtype=np.float32)
    #coords = np.array([p.coords for p in props], dytpe=np.int32)

    # Construct the NeXus hierarchical structure - allows multiple groups to be attached 
    entry = NXentry(name="entry") 
    # - DATA GROUP 1 - #
    y_dim, x_dim, mz_dim = hsc.shape # Might need to check order for all data
    msi_data = NXdata(
        signal = NXfield(hsc, name="data", units="counts", 
                         description="Mass spectra intensity counts"),
        axes = [
                NXfield(np.arange(y_dim), name="y", units="pixels"),
                NXfield(np.arange(x_dim), name="x", units="pixels"),
                NXfield(np.arange(mz_dim), name="m_z", units="Th", description="Mass-to-charge ratio") # NOTE: Th = Thomson units 
                ]
            )

    msi_data.title = "MSI HS Cube"

    entry.data = msi_data

    # - DATA GROUP 2 - #
    superpixel_group = NXdata(
        signal = NXfield(segments, name="Superpixels", 
                         description="2D array of SPs"),
        axes = [
            NXfield(np.arange(segments.shape[0]), name="y", units="pixels,"),
            NXfield(np.arange(segments.shape[1]), name="x", units="pixels")
        ]
    )

    superpixel_group.title = "SP Map"

    entry.superpixels = superpixel_group

    # - DATA GROUP C - #
    # Metadata from regionprops as NXprocess
    process = NXprocess(name="superpixel_analysis")
    process.program = "scikit-image"
    process.version = "slic/regionprops"

    # Store tabular region properties as NXfields inside process group
    process.sp_id = NXfield(sp_ids, units="1", description="Superpixel label index")
    process.area = NXfield(areas, units="pixels", description="Number of pixels in SP")
    process.centroid = NXfield(centroids, units="pixels", description="Y, X coordinates of centroid")
    process.mean_intensity = NXfield(mean_intensities, units="AU", description="Average pixel intensity") # NOTE: AU = Arbitrary Units
    process.eccentricity = NXfield(eccentricities, units="1", description="Eccentricty of each SP")
    process.perimeter = NXfield(perimeters, units="pixels", description="Perimeter of each SP")
    process.std_intensity = NXfield(std_intensities, units="AU", description="Average STD of each pixels intensity")

    entry.process = process

    # - Wrap root and save - #
    root = NXroot(entry)
    root.save(file_name, mode = "w")

    print(f"Saved MSI, superpixel and metadata as: {file_name} to {working_direct}")