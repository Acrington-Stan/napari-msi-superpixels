# --- Package Import --- #
import re
import time
import numpy as np

# --- Main Functions --- #
def construct_hypercube(file):
    if 'Spectra' not in file:
        print("Error: 'Spectra' group not found")
    else:
        spectra_group = file['Spectra']

    matches = [key for key in spectra_group.keys() if re.search(r'Layer\d+', key, re.I)]

    if len(matches) <= 1:
        print(f"\nOnly one layer found: {matches}")
    else:
        print(f"\nNumber of layers found: {len(matches)}")

    layer = matches[0]
    spec = spectra_group[layer]

    keys = sorted(spec.keys())
    cords = []
    for k in keys:
        x, y = k.replace("Pixel", "").split(",")
        cords.append((int(x), int(y)))

    max_x = max(c[0] for c in cords) + 1
    max_y = max(c[1] for c in cords) + 1

    mass = file['ExperimentDetails/MassArray'][:]

    mass_min = int(np.floor(mass[0]))
    mass_max = int(np.ceil(mass[-1]))

    unit_masses = np.arange(mass_min, mass_max + 1)

    print(f"\nNumber of unit masses: {len(unit_masses)}")

    n_spectra = spec[keys[0]][:].squeeze().shape[0]

    print(f"\nNumber of spectra found: {n_spectra}")

    n_bins = mass_max - mass_min + 1
    bin_indices = np.round(mass).astype(int) - mass_min

    cube = np.zeros((max_x, max_y, n_spectra))
    unit_cube = np.zeros((max_x, max_y, n_bins), dtype=np.float32)

    start_time = time.time()

    for k, (x, y) in zip(keys, cords):
        spectrum = spec[k][:].squeeze()
        cube[x, y, :] = spectrum
        unit_cube[x, y] = np.bincount(bin_indices, weights=spectrum, minlength=n_bins)

    print(f"\nCube shape: {cube.shape}")
    print(f"\nUnit cube shape: {unit_cube.shape}")

    runtime = (time.time() - start_time)

    if runtime >= 60.00:
        print(f"\nRuntime: {runtime / 60:.2f} minutes")
    else:
        print(f"\nCube construction time: {runtime:.2f} seconds")

    return cube, unit_cube

def construct_imzml_cube(parser):
    # - Binned Cube - #
    coords = np.array(parser.coordinates)
    
    coords[:, 0] -= coords[:, 0].min()
    coords[:, 1] -= coords[:, 1].min()

    max_x = int(coords[:, 0].max())
    max_y = int(coords[:, 1].max())

    # Scan all spectra to find global min and max m/z bounds
    min_mz_global = float("inf")
    max_mz_global = float("-inf")

    for idx in range(len(coords)):
        mz, _ = parser.getspectrum(idx)
        if len(mz) > 0:
            min_mz_global = min(min_mz_global, mz[0])
            max_mz_global = max(max_mz_global, mz[-1])

    # Create the unit-bin m/z axis (1 Da int steps)
    unit_mz_axis = np.arange(np.floor(min_mz_global), np.ceil(max_mz_global) + 1, 1.0)

    # Initialize datacube
    binned_datacube = np.zeros((max_y + 1, max_x + 1, len(unit_mz_axis)), dtype=np.float32)

    # Populate datacube
    for idx, (x, y, z) in enumerate(coords):
        mz, intensities = parser.getspectrum(idx)
        if len(mz) == 0:
            continue

        # Map raw m/z values to nearest unit bin index
        bin_indices = np.digitize(mz, unit_mz_axis) - 1

        for m_val, intensity in zip(bin_indices, intensities):
            if 0 <= m_val < len(unit_mz_axis):
                binned_datacube[int(y), int(x), m_val] += intensity

    print("Binned datacube shape:", binned_datacube.shape)
    print("Unit m/z axis shape:", unit_mz_axis.shape)

    # - Full Scale Cube - # 
    # Define a common m/z axis for the whole datacube
    # (Using the min/max and step size of the first spectrum, or define your own resolution)
    sample_mz, _ = parser.getspectrum(0)
    min_mz, max_mz = sample_mz[0], sample_mz[-1]

    # Create a unified m/z vector (adjust number of bins as needed)
    n_bins = len(sample_mz)
    common_mz_axis = np.linspace(min_mz, max_mz, n_bins)

    # Initialize the 3D data cube with the standardized dimensions
    datacube = np.zeros((max_y + 1, max_x + 1, n_bins), dtype=np.float32)

    # Populate and interpolate each pixel onto the common m/z grid
    for idx, (x, y, z) in enumerate(coords):
        mz, intensities = parser.getspectrum(idx)

    # Interpolate this pixel's intensities onto the shared common_mz_axis
    # (Setting values outside the pixel's original range to 0)
        interpolated_intensities = np.interp(
            common_mz_axis, mz, intensities, left=0.0, right=0.0
            )
        datacube[int(y), int(x), :] = interpolated_intensities

    print("Datacube shape:", datacube.shape)
    print("Common m/z axis shape:", common_mz_axis.shape)

    return datacube, binned_datacube