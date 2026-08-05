// Packages 
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

// Hyperparameters for each algorithm 
#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "algorithm", rename_all = "lowercase")]
pub enum SuperpixelParams {
    Slic {
        n_segments: usize,
        compactness: f64,
        sigma: f64,
        enforce_connectivity: bool,
    },
    Felzenszwalb {
        scale: f64,
        sigma: f64,
        min_size: usize,
    },
}

// Single superpixel run log entry 
#[derive(Serialize, Deserialize, Debug)]
pub struct SuperpixelRunRecord {
    pub run_id: String, 
    pub timestamp_utc: String,
    pub layer_name: String,
    pub image_shape: Vec<usize>,
    pub execution_time_ms: u64,
    pub num_superpixels_generated: usize, 
    pub params: SuperpixelParams, 
}

#[pyfunction]
fn create_slic_record(
    run_id: String,
    timestamp_utc: String,
    layer_name: String,
    image_shape: Vec<usize>,
    execution_time_ms: u64,
    num_superpixels: usize, 
    n_segments: usize,
    compactness: f64, 
    sigma: f64, 
    enforce_connectivity: bool,
) -> PyResult<String> {
    let params = SuperpixelParams::Slic {
        n_segments,
        compactness, 
        sigma,
        enforce_connectivity,
    };

    let record = SuperpixelRunRecord {
        run_id,
        timestamp_utc, 
        layer_name, 
        image_shape, 
        execution_time_ms, 
        num_superpixels_generated: num_superpixels,
        params,
    };

    serde_json::to_string_pretty(&record)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

#[pyfunction]
fn create_felzenszwalb_record(
    run_id: String,
    timestamp_utc: String,
    layer_name: String,
    image_shape: Vec<usize>,
    execution_time_ms: u64,
    num_superpixels: usize,
    scale: f64,
    sigma: f64,
    min_size: usize,
) -> PyResult<String> {
    let params = SuperpixelParams::Felzenszwalb {
        scale,
        sigma,
        min_size,
    };

    let record = SuperpixelRunRecord {
        run_id,
        timestamp_utc,
        layer_name,
        image_shape,
        execution_time_ms,
        num_superpixels_generated: num_superpixels,
        params,
    };

    serde_json::to_string_pretty(&record)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))

}

#[pymodule]
fn superpixel_flight_recorder(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_slic_record, m)?)?;
    m.add_function(wrap_pyfunction!(create_felzenszwalb_record, m)?)?;
    Ok(())
}