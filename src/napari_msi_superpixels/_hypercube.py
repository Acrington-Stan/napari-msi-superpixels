# --- Package Import --- #
import re
import time
import numpy as np

# --- Main Function --- #
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