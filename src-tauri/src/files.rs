//! Export destinations.
//!
//! Saving is done by the webview through the official plugins (apps/web/src/lib/exportManager.ts):
//! `tauri-plugin-dialog` for the native Save As and folder pickers, `tauri-plugin-fs` for the
//! write. The webview still has no general file-system access. It may write only
//!
//! * inside the app's exports folder (`<data dir>/FineRetouch/exports`), granted statically in
//!   `capabilities/default.json` and added to the runtime scope at startup, and
//! * to files and folders the user picked in a native dialog. The dialog plugin adds every pick
//!   to the fs scope, and `tauri-plugin-persisted-scope` remembers them, so a default export
//!   folder chosen once stays writable after a restart.
//!
//! This module only adds what the plugins lack: "Show in folder" for exported files.

use std::path::{Path, PathBuf};

use tauri::AppHandle;
use tauri_plugin_fs::FsExt;

/// The webview may only reveal what it may write: anything else would let a page probe the disk.
fn may_reveal(app: &AppHandle, path: &Path) -> bool {
    app.try_fs_scope().is_some_and(|scope| scope.is_allowed(path))
}

/// Show an exported file (selected) or folder in the platform file manager.
#[tauri::command]
pub fn reveal_path(app: AppHandle, path: String) -> Result<(), String> {
    let path = PathBuf::from(path);
    if !may_reveal(&app, &path) {
        return Err("only exported files and folders can be shown".into());
    }
    let mut cmd;
    #[cfg(target_os = "windows")]
    {
        cmd = std::process::Command::new("explorer");
        if path.is_dir() {
            cmd.arg(&path);
        } else {
            cmd.arg(format!("/select,{}", path.display()));
        }
    }
    #[cfg(target_os = "macos")]
    {
        cmd = std::process::Command::new("open");
        if path.is_dir() {
            cmd.arg(&path);
        } else {
            cmd.arg("-R").arg(&path);
        }
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        cmd = std::process::Command::new("xdg-open");
        cmd.arg(if path.is_dir() { path.as_path() } else { path.parent().unwrap_or(&path) });
    }
    cmd.spawn().map(|_| ()).map_err(|e| e.to_string())
}

/// Make the default exports folder writable for this session (the capability file grants the
/// same path statically; the runtime grant also covers `reveal_path` and symlinked data dirs).
pub fn allow_exports_dir(app: &AppHandle, dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(scope) = app.try_fs_scope() {
        scope.allow_directory(dir, true)?;
    }
    Ok(())
}
