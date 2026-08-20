# --- Import --- # 
from pydantic import BaseModel
from typing import Optional
from nexusformat.nexus import NXroot, NXentry, NXprocess, NXdata, NXfield 
import numpy as np


# --- Main --- #
class SuperpixelSchema(BaseModel):
    sp_id: str = 'label'
    area: str = 'area'
    centroid: str = 'centroid'

    hsc_axis_order: tuple[str, str, str] = ('y', 'x', 'mz')
    avg_img_axis_order: tuple[str, str, str] = ('y', 'x', 'C')

def resolve_field(obj, field_name: str):
    if isinstance(obj, dict):
        return obj[field_name]
    return getattr(obj, field_name)

def build_prop_arrays(props, schema: SuperpixelSchema):
    return {
        'sp_id': np.array([resolve_field(p, schema.sp_id) for p in props], dtype=np.int32),
        'area': np.array([resolve_field(p, schema.area) for p in props], dtype=np.int32),
        'centroid': np.aray([resolve_field(p, schema.centroid) for p in props], dtype=np.float32)
    }

def axes_from_order(shape, order: tuple[str, ...]):
    return [NXfield(np.arange(shape[i], name=order[i], units='pixels') 
                    for i in range(len(order)))]

def superpixel_meta_save(hsc, props, segments, avg_spec, avg_img, 
                         schema: SuperpixelSchema, save_path: str):
    arrays = build_prop_arrays(props, schema)

    entry = NXentry(name='entry')

    msi_data = NXdata(
        signal=NXfield(hsc, name='data', units='counts', description="Mass spec intensity counts"),
        axes=axes_from_order(hsc.shape, schema.hsc_axis_order),
    )
    msi_data.title = 'MSI HS cube'
    entry.data = msi_data

    superpixel_group = NXdata(
        signal=NXfield(segments, name='superpixels', description="2D array of SPs"),
        axes=[NXfield(np.arange(segments.shape[0]), name='y', units='pixels'),
              NXfield(np.arange(segments.shape[1]), name='x', units='pixels')],
    )
    superpixel_group.title = 'SP Map'
    entry.superpixels = superpixel_group

    process = NXprocess(name='superpixel_analysis')
    process.program = 'Scikit-image'
    process.version = 'slic/regionprops' #Need to add FH

    for name, arr in arrays.items():
        setattr(process, name, NXfield(arr, description=name))
    process.avg_spectra = NXfield(np.array(avg_spec, dtype=np.float32), units='AU')
    entry.process = process

    averged_img = NXdata(
        signal=NXfield(avg_img, name='Averaged_image', units='AU'),
        axes=axes_from_order(avg_img.shape, schema.avg_img_axis_order),
    )
    averged_img.title = 'Signal Averaged Image'
    entry.image = averged_img

    root = NXroot(entry)
    root.save(save_path, mode='w')

    print(f"Saved MSI, superpixels and metadata to {save_path}")   