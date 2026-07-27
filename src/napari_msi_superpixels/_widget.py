# --- Package Import --- #
import h5py
import numpy as np
from ._hypercube import construct_hypercube
from magicgui import magicgui
from skimage.segmentation import felzenszwalb, slic, find_boundaries
from napari.layers import Image
from pathlib import Path
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QApplication
from napari.viewer import current_viewer

def reset_button(widget, label):
    widget.call_button.enabled = True
    widget.call_button.text = label

# --- Importer --- #
@thread_worker(start_thread=False)
def run_import(import_path):
    with h5py.File(import_path, "r") as f:
        cube, unit_cube = construct_hypercube(f)
        data_transposed = np.moveaxis(unit_cube, -1, 0)
    return data_transposed

def make_import_widget():
    @magicgui(
        call_button="Import MSI Data",
        auto_call=False,
        import_path={'label': 'H5 File'}
    )
    def import_msi_widget(import_path: Path = Path()):
        viewer = current_viewer()
        if not import_path.exists():
            print("\nInvalid path")
            return

        import_msi_widget.call_button.enabled = False
        import_msi_widget.call_button.text = "Importing..."
        QApplication.processEvents()

        worker = run_import(import_path)
        worker.returned.connect(lambda result: viewer.add_image(
            result, name=f"MSI Cube ({result.shape})"))
        worker.errored.connect(lambda e: print(f"Import failed: {e}"))
        worker.finished.connect(lambda: reset_button(import_msi_widget, "Import MSI Data"))
        worker.start()

    import_msi_widget.import_path.tooltip = "Path to MSI data"   
    return import_msi_widget                                      

# --- Superpixel Generator --- #
@thread_worker(start_thread=False)
def run_superpixel(cube, n_segs, comp, sig, iters, algorithm, scale, min_size):
    if algorithm == 'SLIC':
        segments = slic(cube, n_segments=n_segs, compactness=comp, sigma=sig,
                         channel_axis=-1, start_label=0, max_num_iter=iters)
    elif algorithm == 'Felzenszwalb':
        segments = felzenszwalb(cube, scale=scale, sigma=sig, min_size=min_size,
                                 channel_axis=-1)

    bounds = find_boundaries(segments, mode='inner')
    bounds_labels = segments * bounds
    return segments, bounds_labels, algorithm

def make_superpixel_widget():
    @magicgui(
        call_button="Generate SPs",
        auto_call=False,
        algorithm={'label': 'SP Algorithm', 'choices': ['SLIC', 'Felzenszwalb']},
        iters={'label': 'Iterations', 'min': 0, 'max': 100, 'visible': True},
        sig={'label': 'Sigma', 'min': 0.0, 'max': 5.0, 'visible': True},
        n_segs={'label': 'Number of Segments', 'min': 1, 'max': 5000, 'visible': True},
        comp={'label': 'Compactness', 'min': 0.1, 'max': 1.0, 'visible': True},
        scale={'label': 'Scale', 'min': 1, 'max': 500, 'visible': False},
        min_size={'label': 'Minimum Size', 'min': 1, 'max': 500, 'visible': False},
    )
    def superpixel_widget(layer: Image, algorithm: str = "SLIC", iters: int = 10, sig: float = 0.5,
                           n_segs: int = 50, comp: float = 0.1, scale: int = 50, min_size: int = 25):
        viewer = current_viewer()
        if layer is None:
            return

        cube_data = layer.data
        cube = np.moveaxis(cube_data, 0, -1)

        tic_img = cube.sum(axis=-1)
        tic_normalised = (tic_img - tic_img.min()) / (tic_img.max() - tic_img.min() + 1e-8)
        psu_rgb = np.stack([tic_normalised] * 3, axis=-1)
        TIC_layer_name = f"TIC Image {psu_rgb.shape}"

        if TIC_layer_name in viewer.layers:
            print("TIC already added skipping")
        else:
            viewer.add_image(psu_rgb, name=TIC_layer_name)

        superpixel_widget.call_button.enabled = False
        superpixel_widget.call_button.text = "Running..."
        QApplication.processEvents()

        worker = run_superpixel(cube, n_segs, comp, sig, iters, algorithm, scale, min_size)
        worker.returned.connect(lambda result: on_gen_done(viewer, result))
        worker.errored.connect(lambda e: print(f"Segmentation failed: {e}"))
        worker.finished.connect(lambda: reset_button(superpixel_widget, "Generate SPs"))
        worker.start()

    def toggle_algorithm_fields(event_or_value):
        algo = event_or_value.value if hasattr(event_or_value, 'value') else event_or_value
        is_slic = (algo == 'SLIC')
        superpixel_widget.n_segs.visible = is_slic
        superpixel_widget.comp.visible = is_slic
        superpixel_widget.iters.visible = is_slic
        superpixel_widget.scale.visible = not is_slic
        superpixel_widget.min_size.visible = not is_slic

    superpixel_widget.algorithm.tooltip = "Choose which algorithm to use for SP generation"
    superpixel_widget.layer.tooltip = "Choose what MSI data to segment"
    superpixel_widget.iters.tooltip = "Number of iterations for SLIC"
    superpixel_widget.sig.tooltip = "Blur strength"
    superpixel_widget.n_segs.tooltip = "Number of target segments for SLIC"
    superpixel_widget.comp.tooltip = "Compactness of SLIC, balances colour and space proxies"
    superpixel_widget.scale.tooltip = "Scale of SPs, larger values create larger SPs"
    superpixel_widget.min_size.tooltip = "Minimum size of SPs"

    superpixel_widget.algorithm.changed.connect(toggle_algorithm_fields)
    toggle_algorithm_fields(superpixel_widget.algorithm.value)

    return superpixel_widget

def on_gen_done(viewer, result):
    segments, bounds_label, algorithm = result
    print(f"Number of segments found: {len(np.unique(segments))}")
    label_name = (f"SLIC ({len(np.unique(segments))} SPs)" if algorithm == 'SLIC'
                  else f"Felzenszwalb ({len(np.unique(segments))} SPs)")
    viewer.add_labels(bounds_label, name=label_name)