# --- Package Import --- #
import h5py
import uuid
import time
import superpixel_flight_recorder
import numpy as np
import pyqtgraph as pg
import pandas as pd
import matplotlib.pyplot as plt 
import napari.types as nt
import dask.array as da
from ._hypercube import construct_hypercube, construct_imzml_cube # Relative port 
from magicgui import magicgui
from skimage.segmentation import felzenszwalb, slic, find_boundaries
from napari.layers import Image
from pathlib import Path
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import QApplication
from napari.viewer import current_viewer
from pyimzml.ImzMLParser import ImzMLParser
from datetime import datetime, timezone
from skimage.measure import label, regionprops_table
from ._saving import superpixel_meta_save 

# - Reset button function - #
def reset_button(widget, label):
    widget.call_button.enabled = True
    widget.call_button.text = label

# --- Importer | H5 --- #
@thread_worker(start_thread=False)
def run_import(import_path):
    with h5py.File(import_path, "r") as f:
        cube, unit_cube = construct_hypercube(f) # Change cube to _

        data_transposed = np.moveaxis(unit_cube, -1, 0)
        data_transposed_full = np.moveaxis(cube, -1, 0)

    return data_transposed, data_transposed_full

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

        def _on_imported(result):
            unit_cube, full_cube = result

            viewer.add_image(unit_cube, name=f"MSI Unit Cube ({unit_cube.shape})")

            viewer.add_image(full_cube, name=f"MSI Full Cube ({full_cube.shape})", visible=False)

        worker = run_import(import_path)
        worker.returned.connect(_on_imported)#(lambda result: viewer.add_image(
            #result, name=f"MSI Cube ({result.shape})"))
        worker.errored.connect(lambda e: print(f"Import failed: {e}"))
        worker.finished.connect(lambda: reset_button(import_msi_widget, "Import MSI Data"))
        worker.start()

    import_msi_widget.import_path.tooltip = "Path to MSI data"   

    return import_msi_widget      

# --- Importer | ImzML --- #
@thread_worker(start_thread=False)
def run_import_imzml(import_path):
    with ImzMLParser(import_path) as parser:
        cube, unit_cube = construct_imzml_cube(parser)
        data_transposed = np.moveaxis(unit_cube, -1, 0)
        
    return data_transposed          

def make_imzml_import_widget():
    @magicgui(
        call_button="Import MSI Data",
        auto_call=False,
        import_path={'label': 'ImzML File'}
    )
    def import_imzml_widget(import_path: Path = Path()):
        viewer = current_viewer()
        if not import_path.exists():
            print("\nInvalid path")
            return 

        import_imzml_widget.call_button.enabled = False
        import_imzml_widget.call_button.text = "Importing..."
        QApplication.processEvents()

        worker = run_import_imzml(import_path)
        worker.returned.connect(lambda result: viewer.add_image(
                    result, name=f"MSI Cube ({result.shape})"))
        worker.errored.connect(lambda e: print(f"Import failed: {e}"))
        worker.finished.connect(lambda: reset_button(import_imzml_widget, "Import MSI Data"))
        worker.start()
        
    import_imzml_widget.import_path.tooltip = "Path to MSI data"   
        
    return import_imzml_widget   
                         
# --- Superpixel Generator --- #
@thread_worker(start_thread=False)
def run_superpixel(cube, n_segs, comp, sig, iters, algorithm, scale, min_size, layer_name):

    start_time = time.perf_counter() # Start time counter 

    if algorithm == 'SLIC':
        segments = slic(cube, n_segments=n_segs, compactness=comp, sigma=sig,
                         channel_axis=-1, start_label=1, max_num_iter=iters) #start label = 0?
    elif algorithm == 'Felzenszwalb':
        segments = felzenszwalb(cube, scale=scale, sigma=sig, min_size=min_size,
                                 channel_axis=-1)
        
    # Calculate execution time 
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    num_sp = int(len(np.unique(segments)))

    # Metadata generation
    run_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()
    image_shape = list(cube.shape)

    # Call Rust function
    if algorithm == 'SLIC':
        log_json = superpixel_flight_recorder.create_slic_record(
            run_id=run_id,
            timestamp_utc=timestamp,
            layer_name=layer_name,
            image_shape=image_shape,
            execution_time_ms=elapsed_ms,
            num_superpixels=num_sp,
            n_segments=n_segs,
            compactness=float(comp),
            sigma=float(sig),
            enforce_connectivity=True,
        )
    else:
        log_json = superpixel_flight_recorder.create_felzenszwalb_record(
            run_id=run_id,
            timestamp_utc=timestamp,
            layer_name=layer_name,
            image_shape=image_shape,
            execution_time_ms=elapsed_ms,
            num_superpixels=num_sp,
            scale=float(scale),
            sigma=float(sig),
            min_size=int(min_size),
        )

    bounds = find_boundaries(segments, mode='inner')
    bounds_labels = segments * bounds

    return segments, bounds_labels, algorithm, log_json

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

        layer_name = layer.name
        
        worker = run_superpixel(cube, n_segs, comp, sig, iters, algorithm, scale, min_size, layer_name)
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

PROPERTY_SCHEMA = {
    'sp_id': 'label',
    'area': 'area',
    'centroid': 'centroid',
    'coords': 'coords'
}

class SPController:
    def __init__(self):
        self.props = None
        self.avg_spec = None
        self.segments = None
        self.cube = None
        self.region_df = None
        self.avg_img = None
        self.mz_axis = None
        self.sp_id = None

    def has_results(self) -> bool:
        return self.region_df is not None and self.avg_spec is not None

_controller = SPController()

def make_attributes_widget():
    @magicgui(
        call_button="Get SP Attributes",
        auto_call=False,
        #intense_img={'label': 'Intensity Image'},
        segments={'label': 'Segments'},
        cube={'label': 'Hyperspectral Cube'},
        tic_img={'label': 'TIC Image'},
        colormap={'label': 'Colormap', 'choices': ['gray', 'viridis', 'magma', 'inferno']}
    )
    def get_sp_attributes(segments: nt.LabelsData, cube: nt.ImageData, #LabelsData
                          tic_img: nt.ImageData, colormap: str = 'gray'):

        viewer = current_viewer()
        
        label_img = label(segments)

        region_table = regionprops_table(
                label_img,
                properties=tuple(PROPERTY_SCHEMA.values()),
        )
                #intensity_image=intense_img,
                #properties=(
                    #"area",
                    #"centroid",
                    #"label",
                    #"intensity_mean",
                    #"eccentricity",
                    #"perimeter",
                    #"intensity_std",
                    #"coords"
                    #))

        region_df = pd.DataFrame(region_table)

        cube = np.moveaxis(cube, 0, -1) #might be redundant 
        
        avg_spec = []

        for _, row in region_df.iterrows():
            coords = np.array(row['coords'])
            region_spectra = cube[coords[:, 0], coords[:, 1], :]
            avg_spec.append(region_spectra.mean(axis=0))

        avg_spec = np.array(avg_spec)
                #cords_list = row['coords']
                #region_spectra = [cube[row, col, :] for (row, col) in cords_list] #x,y
                #avg_spec.append(np.mean(region_spectra, axis=0))

        region_intens = avg_spec.sum(axis=1)#[np.sum(avg) for avg in avg_spec]

        img = np.zeros(tic_img.shape, dtype=float)#np.zeros_like(tic_img, dtype=float) #segments?

        for i, (_, row) in enumerate(region_df.iterrows()):
            coords = np.asarray(row['coords'])
            img[coords[:, 0], coords[:, 1]] = region_intens[i]
            #intensity = region_intens[r]
            #for (x, y) in row['coords']:
            #    img[x, y] = intensity

        #img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
        #cmap = plt.get_cmap(colormap) #Give options? 
        #img_coloured = cmap(img_norm)[..., :3]

        _controller.cube = cube
        _controller.segments = segments
        _controller.region_df = region_df
        _controller.avg_spec = avg_spec
        #_controller.avg_img = img_coloured
        _controller.sp_id = _controller.region_df['label'].values
        _controller.mz_axis = np.arange(cube.shape[-1]) #might need to be more careful with how

        viewer.add_image(img, name="Region Averaged Image", colormap=colormap)#, rgb=True)
    
    return get_sp_attributes

def make_reconstruction_widget():
    def reconstruction_widget():
        ...

# NOTE: how access non-layers? parse data from selected layers? 
def make_saving_widget():
    @magicgui(
        call_button="Save Data",
        auto_call=False,
        #cube={'label': 'MSI Data to Save'},
        #props={'label': 'Metadata to Save'},
        #segments={'label': 'Superpixels to Save'},
        file_name={'label': 'Name of Save File', 'widget_type': 'LineEdit'},
        directory={'label': 'Directory to Save Data', 'mode': 'd'},
        #avg_spec={'label': 'Average Spectra'},
        #avg_img={'label': 'Spectra Averaged Image'}
    )
    def saving_widget(file_name: str = 'results', directory: Path = Path(".")):#cube, props, segments, avg_spec, avg_img, file_name, directory):
        if not _controller.has_results():
            print("No results to save yet")
            return 

        if not file_name.endswith('.nxs'):
            file_name += '.nxs'

        save_path = Path(directory) / f'{file_name}'

        superpixel_meta_save(
            hsc=_controller.cube,
            region_df=_controller.region_df,
            segments=_controller.segments,
            avg_spec=_controller.avg_spec,
            avg_img=_controller.avg_img,
            save_path=save_path
            )

    return saving_widget

def make_plotting_widget():
    @magicgui(
        call_button="Open Plot",
        auto_call=False,
        log={'label': 'Toggle Logarithmic Scale'}
    )
    def plotting_widget(log: bool = False):
        #if mz_axis is None or avg_spec is None:
        #    return 

        if not _controller.has_results():
            return 

        #if _controller hasattr widget viewer.window.remove_dock_widget

        mz_axis = _controller.mz_axis
        
        viewer = current_viewer()

        layer = viewer.layers.selection.active

        #if viewer is None:
        #    raise RuntimeError

        layer.mouse_drag_callbacks.clear()
        
        plot_w = pg.PlotWidget()
        plot_w.setTitle("Click Image to Get Pixel Spectrum")
        plot_w.setLabel('bottom', "Mass-to-charge (m/z)")
        plot_w.setLabel('left', "Intensity (counts)")

        if log == True:
            plot_w.setLogMode(y=True)

        id_spectrum_dict = dict(zip(_controller.sp_id, _controller.avg_spec))

        intial_spec = _controller.avg_spec[0].copy()
        #intial_spec[intial_spec <= 0] = 0.1 #log safety?
        curve = plot_w.plot(mz_axis, intial_spec, pen=pg.mkPen('w', width=1)) 

        viewer.window.add_dock_widget(plot_w, area='right', name="Spectrum")

        @layer.mouse_drag_callbacks.append
        def get_spec_click(layer, event):
            coords = layer.world_to_data(event.position)
            y, x = int(coords[0]), int(coords[1])
            #check image bounds and click valid within
            max_y, max_x = _controller.cube.shape[0], _controller.cube.shape[1]
            if 0 <= y < max_y and 0 <= x < max_x:
                sp_id = _controller.segments[y,x] #slicing my clicked x y
                spec = id_spectrum_dict[sp_id] #sp_id is the numeric id
                #log processing if needed
                #spec[spec <= 0] = 0.1
                #log_spec = np.log10(spec)
                #update curve
                curve.setData(mz_axis, spec) #log_spec
                plot_w.setTitle(f"Mass spec at pixel (Y: {y}, X: {x})")

        v_line = pg.InfiniteLine(
            pos=0, 
            angle=90,          # 90 makes it vertical, 0 would be horizontal
            pen=pg.mkPen('r', width=2),  # Red line with width 2
            movable=True       # Optional: allows the user to drag the line around
        )
        plot_w.addItem(v_line)

        def update_line_from_viewer(event):
            # Get the current index of the m/z axis from Napari's slider
            # (Assuming your m/z dimension is the last axis or index 0 depending on your shape)
            current_index = viewer.dims.current_step[0] # Adjust index based on your cube's axes order
            
            # Map that index back to the actual m/z value
            current_mz = mz_axis[current_index]
            
            # Move the line without triggering an infinite loop of signals
            v_line.blockSignals(True)
            v_line.setValue(current_mz)
            v_line.blockSignals(False)

            # Connect Napari's dimension change event to your function
        viewer.dims.events.current_step.connect(update_line_from_viewer)

        def update_slice_from_line():
            current_mz = v_line.value()
            
            # Find the index in your mz_axis that is closest to the line's current position
            closest_index = np.argmin(np.abs(mz_axis - current_mz))
            
            # Update Napari's current slider step (block signals if needed to prevent loops)
            current_steps = list(viewer.dims.current_step)
            current_steps[0] = closest_index  # Update the m/z axis dimension index
            viewer.dims.current_step = current_steps

        # Connect the line's position change event to your function
        v_line.sigPositionChanged.connect(update_slice_from_line)

    return plotting_widget

def on_gen_done(viewer, result):
    segments, bounds_label, algorithm, log_json = result

    print("# --- Flight Recorder Log --- #")
    print(log_json)

    #img_coloured = get_sp_attributes()
    #Need to fix
    #viewer.add_image(img_coloured, name="SP Intensities")
    
    #print(f"Number of segments found: {len(np.unique(segments))}")
    label_name = (f"SLIC ({len(np.unique(segments))} SPs)" if algorithm == 'SLIC'
                  else f"Felzenszwalb ({len(np.unique(segments))} SPs)")
    
    viewer.add_labels(bounds_label, name=label_name)

    viewer.add_labels(segments, name="Raw Segments") #For correct image?