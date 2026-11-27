//! `kototok-py`: PyO3 bindings for the AsymScript-BPE (`kototok`) tokenizer.
//!
//! Exposes `kototok.Tokenizer` (load/encode/decode) and `kototok.train(...)`.

use std::str::FromStr;

use kototok_core::{Model, ModelFormat, Tokenizer as CoreTokenizer};
use kototok_train::{TrainPreset, train as train_model};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn to_value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// AsymScript-BPE tokenizer loaded from any supported model format.
#[pyclass(name = "Tokenizer", frozen)]
struct PyTokenizer {
    inner: CoreTokenizer,
}

#[pymethods]
impl PyTokenizer {
    /// Load a `.ktok`, HF `tokenizer.json`/directory, `.tiktoken`, `vocab.json`+`merges.txt` directory, or kototok `.json` model.
    #[staticmethod]
    #[pyo3(signature = (path, tags = None))]
    fn load(path: &str, tags: Option<Vec<String>>) -> PyResult<Self> {
        let mut model = Model::load_auto(path).map_err(to_value_err)?;
        if let Some(tags) = tags {
            let set: std::collections::HashSet<String> = tags.into_iter().collect();
            model = model.filtered_by_tags(&set);
        }
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

    /// Save the model, inferring the format from `path` unless `format` is given.
    ///
    /// Supported formats: `ktok`, `json`, `hf`, `hf-dir`, `tiktoken`, `bpe`.
    #[pyo3(signature = (path, format = None))]
    fn save(&self, path: &str, format: Option<&str>) -> PyResult<()> {
        if let Some(format) = format {
            let format = ModelFormat::from_str(format).map_err(to_value_err)?;
            self.inner
                .model()
                .save_with_format(path, format)
                .map_err(to_value_err)
        } else {
            self.inner.model().save_auto(path).map_err(to_value_err)
        }
    }

    /// Save as a Hugging Face `tokenizer.json` file.
    fn save_hf(&self, path: &str) -> PyResult<()> {
        self.inner.model().save_hf_json(path).map_err(to_value_err)
    }

    /// Save as a Hugging Face tokenizer directory.
    fn save_hf_dir(&self, path: &str) -> PyResult<()> {
        self.inner.model().save_hf_dir(path).map_err(to_value_err)
    }

    /// Save as a tiktoken mergeable-rank table.
    fn save_tiktoken(&self, path: &str) -> PyResult<()> {
        self.inner.model().save_tiktoken(path).map_err(to_value_err)
    }

    /// Save as a GPT-2-style `vocab.json`/`merges.txt` directory.
    fn save_bpe(&self, path: &str) -> PyResult<()> {
        self.inner
            .model()
            .save_bpe_files(path)
            .map_err(to_value_err)
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
/// `input` is a corpus text path. When `output` is given, the model is saved
/// there in the format inferred from the path unless `format` is given.
/// Returns a ready-to-use `Tokenizer`.
#[pyfunction]
#[pyo3(signature = (input, output=None, vocab_size=4096, preset="asym-full", threads=0, format=None))]
fn train(
    py: Python<'_>,
    input: &str,
    output: Option<&str>,
    vocab_size: usize,
    preset: &str,
    threads: usize,
    format: Option<&str>,
) -> PyResult<PyTokenizer> {
    let preset = TrainPreset::from_str(preset)
        .map_err(|_| PyValueError::new_err(format!("unknown preset: {preset}")))?;
    let format = format
        .map(ModelFormat::from_str)
        .transpose()
        .map_err(to_value_err)?;
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
        if let Some(format) = format {
            model.save_with_format(path, format).map_err(to_value_err)?;
        } else {
            model.save_auto(path).map_err(to_value_err)?;
        }
    }
    let inner = CoreTokenizer::from_model(model).map_err(to_value_err)?;
    Ok(PyTokenizer { inner })
}

/// List the six ablation preset names.
#[pyfunction]
fn presets() -> Vec<&'static str> {
    TrainPreset::ALL.iter().map(|p| p.name()).collect()
}

/// Detect the format of an existing model file or directory.
#[pyfunction]
fn detect_format(path: &str) -> PyResult<String> {
    Model::detect_format(path)
        .map(|format| format.to_string())
        .map_err(to_value_err)
}

/// List the supported model format names.
#[pyfunction]
fn formats() -> Vec<&'static str> {
    vec!["ktok", "json", "hf", "hf-dir", "tiktoken", "bpe"]
}

/// Python module entry point.
#[pymodule]
fn kototok(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTokenizer>()?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    m.add_function(wrap_pyfunction!(detect_format, m)?)?;
    m.add_function(wrap_pyfunction!(formats, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
