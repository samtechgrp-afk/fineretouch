//! Local storage locations.
//!
//! | OS      | data dir                                          |
//! |---------|---------------------------------------------------|
//! | Windows | `%APPDATA%\FineRetouch`                           |
//! | macOS   | `~/Library/Application Support/FineRetouch`       |
//! | Linux   | `$XDG_DATA_HOME/FineRetouch` (`~/.local/share/…`) |
//!
//! Inside it: `logs/` (engine.log from the engine, engine-console.log from this shell),
//! `cache/` (engine scratch space) and `exports/` (the default export folder). The webview's own storage (localStorage for presets and
//! settings) lives in Tauri's per-identifier webview data directory.

use std::path::PathBuf;

use tauri::{AppHandle, Manager};

pub const APP_DIR_NAME: &str = "FineRetouch";

#[derive(Debug, Clone)]
pub struct AppPaths {
    pub data: PathBuf,
    pub logs: PathBuf,
    pub cache: PathBuf,
    /// Default export folder (`%APPDATA%\FineRetouch\exports` on Windows).
    pub exports: PathBuf,
    /// Bundled models (the `models/` resource from tauri.conf.json).
    pub models: PathBuf,
}

impl AppPaths {
    pub fn resolve(app: &AppHandle) -> Result<Self, Box<dyn std::error::Error>> {
        let data = app.path().data_dir()?.join(APP_DIR_NAME);
        let logs = data.join("logs");
        let cache = data.join("cache");
        let exports = data.join("exports");
        for dir in [&data, &logs, &cache, &exports] {
            std::fs::create_dir_all(dir)?;
        }
        let models = app.path().resource_dir()?.join("models");
        Ok(Self { data, logs, cache, exports, models })
    }
}
