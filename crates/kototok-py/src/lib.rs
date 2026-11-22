//! `kototok-py`: PyO3 bindings for the AsymScript-BPE (`kototok`) tokenizer.
//!
//! Exposes `kototok.Tokenizer` (load/encode/decode) and `kototok.train(...)`.

use std::str::FromStr;

use kototok_core::{Model, Tokenizer as CoreTokenizer};
use kototok_train::{TrainPreset, train as train_model};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn to_value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// AsymScript-BPE tokenizer loaded from a `.ktok` model.
#[pyclass(name = "Tokenizer", frozen)]
struct PyTokenizer {
    inner: CoreTokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load a `.ktok` (binary) or `.json` model file.
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let model = Model::load_auto(path).map_err(to_value_err)?;
        let inner = CoreTokenizer::from_model(model).map_err(to_value_err)?;
        Ok(Self { inner })
    }

    /// Encode `text` to final token ids.
    fn encode(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.allow_threads(|| self.inner.encode(text))
            .map_err(to_value_err)
    }

    /// Encode `text` to `(token_id, byte_start, byte_end)` tuples.
    fn encode_with_offsets(
        &self,
        py: Python<'_>,
        text: &str,
    ) -> PyResult<Vec<(u32, usize, usize)>> {
        py.allow_threads(|| self.inner.encode_with_offsets(text))
            .map_err(to_value_err)
    }

    /// Encode a batch of texts; releases the GIL.
    fn encode_batch(&self, py: Python<'_>, texts: Vec<String>) -> PyResult<Vec<Vec<u32>>> {
        py.allow_threads(|| {
            let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
            self.inner.encode_batch(&refs)
        })
        .map_err(to_value_err)
    }

    /// Encode `text` and return each token's bytes as lossy strings.
    fn pieces(&self, text: &str) -> PyResult<Vec<String>> {
        self.inner.tokenize_pieces(text).map_err(to_value_err)
    }

    /// Decode token ids to a `str` (errors on invalid UTF-8 or unknown id).
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        self.inner.decode(&ids).map_err(to_value_err)
    }

    /// Decode token ids to raw `bytes`.
    fn decode_bytes<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.decode_bytes(&ids).map_err(to_value_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Lossy decode; never fails, unknown ids are skipped.
    fn decode_lossy(&self, ids: Vec<u32>) -> String {
        self.inner.decode_lossy(&ids)
    }

    /// Alive vocabulary size.
    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.model().vocab_size()
    }

    /// Save the model as `.ktok` binary.
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.model().save_binary(path).map_err(to_value_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "kototok.Tokenizer(vocab_size={})",
            self.inner.model().vocab_size()
        )
    }
}

/// Train a tokenizer on a UTF-8 corpus file.
///
/// `input` is a corpus text path. When `output` is given, the `.ktok` model is
/// saved there. Returns a ready-to-use `Tokenizer`.
#[pyfunction]
#[pyo3(signature = (input, output=None, vocab_size=4096, preset="asym-full", threads=0))]
fn train(
    py: Python<'_>,
    input: &str,
    output: Option<&str>,
    vocab_size: usize,
    preset: &str,
    threads: usize,
) -> PyResult<PyTokenizer> {
    let preset = TrainPreset::from_str(preset)
        .map_err(|_| PyValueError::new_err(format!("unknown preset: {preset}")))?;
    let corpus = std::fs::read_to_string(input).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let cfg = preset.config(vocab_size);

    let run = || train_model(&corpus, &cfg).map_err(to_value_err);
    let model = py.allow_threads(|| match threads {
        0 => run(),
        n => rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            .install(run),
    })?;

    if let Some(path) = output {
        model.save_binary(path).map_err(to_value_err)?;
    }
    let inner = CoreTokenizer::from_model(model).map_err(to_value_err)?;
    Ok(PyTokenizer { inner })
}

/// List the six ablation preset names.
#[pyfunction]
fn presets() -> Vec<&'static str> {
    TrainPreset::ALL.iter().map(|p| p.name()).collect()
}

/// Python module entry point.
#[pymodule]
fn kototok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
