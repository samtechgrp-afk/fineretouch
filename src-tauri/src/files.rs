//! Native export dialogs and file writes.
//!
//! The webview never gets general file-system access. It can only write to a path the
//! user picked in one of these dialogs during this session: a Save As target, or a plain
//! file name inside a folder chosen with the folder picker. Imports keep using the
//! webview's own `<input type="file">` and drag and drop, which already open the native
//! picker and hand the editor zero-copy `File` objects.

use std::collections::HashSet;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;

use percent_encoding::percent_decode_str;
use tauri::ipc::{InvokeBody, Request};
use tauri::{AppHandle, Manager, State};
use tauri_plugin_dialog::DialogExt;

#[derive(Default)]
pub struct ExportScope {
    files: Mutex<HashSet<PathBuf>>,
    dirs: Mutex<HashSet<PathBuf>>,
}

impl ExportScope {
    fn allows_reveal(&self, path: &Path) -> bool {
        self.files.lock().unwrap().contains(path)
            || self.dirs.lock().unwrap().iter().any(|d| path.starts_with(d))
    }
}

fn default_dir(app: &AppHandle) -> Option<PathBuf> {
    app.path().picture_dir().ok().or_else(|| app.path().home_dir().ok())
}

/// Save As dialog for a single export. Returns the chosen path, or None if cancelled.
#[tauri::command]
pub async fn choose_save_path(
    app: AppHandle,
    scope: State<'_, ExportScope>,
    default_name: String,
) -> Result<Option<String>, String> {
    let name = sanitize_name(&default_name).unwrap_or_else(|| "photo.jpg".into());
    let ext = Path::new(&name).extension().and_then(|e| e.to_str()).unwrap_or("jpg").to_ascii_lowercase();
    let mut dialog = app.dialog().file().set_title("Export photo").set_file_name(&name);
    if let Some(dir) = default_dir(&app) {
        dialog = dialog.set_directory(dir);
    }
    let label = match ext.as_str() {
        "jpg" | "jpeg" => "JPEG image",
        "png" => "PNG image",
        "webp" => "WebP image",
        "tif" | "tiff" => "TIFF image",
        _ => "Image",
    };
    let Some(picked) = dialog.add_filter(label, &[ext.as_str()]).blocking_save_file() else {
        return Ok(None);
    };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    scope.files.lock().unwrap().insert(path.clone());
    Ok(Some(path.to_string_lossy().into_owned()))
}

/// Folder picker for batch exports. Returns the folder, or None if cancelled.
#[tauri::command]
pub async fn choose_export_folder(app: AppHandle, scope: State<'_, ExportScope>) -> Result<Option<String>, String> {
    let mut dialog = app.dialog().file().set_title("Export photos to folder");
    if let Some(dir) = default_dir(&app) {
        dialog = dialog.set_directory(dir);
    }
    let Some(picked) = dialog.blocking_pick_folder() else {
        return Ok(None);
    };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    scope.dirs.lock().unwrap().insert(path.clone());
    Ok(Some(path.to_string_lossy().into_owned()))
}

/// A bare file name: no directories, no `..`, nothing hidden or empty.
fn sanitize_name(name: &str) -> Option<String> {
    let mut comps = Path::new(name).components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(n)), None) => {
            let n = n.to_str()?.trim();
            (!n.is_empty() && !n.starts_with('.') && !n.contains(['/', '\\', ':'])).then(|| n.to_string())
        }
        _ => None,
    }
}

fn header(request: &Request<'_>, name: &str) -> Result<String, String> {
    let raw = request
        .headers()
        .get(name)
        .ok_or_else(|| format!("missing {name} header"))?
        .to_str()
        .map_err(|e| e.to_string())?;
    percent_decode_str(raw).decode_utf8().map(|s| s.into_owned()).map_err(|e| e.to_string())
}

/// Write an export. Body: the encoded image bytes (raw IPC body, not JSON).
/// Headers: `x-fineretouch-target` (a Save As path, or a chosen folder) and
/// `x-fineretouch-name` (empty for Save As, else the file name inside the folder).
#[tauri::command]
pub async fn write_export(request: Request<'_>, scope: State<'_, ExportScope>) -> Result<(), String> {
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err("expected the image bytes as the raw request body".into());
    };
    let target = PathBuf::from(header(&request, "x-fineretouch-target")?);
    let name = header(&request, "x-fineretouch-name")?;

    let dest = if name.is_empty() {
        if !scope.files.lock().unwrap().contains(&target) {
            return Err("that path was not chosen in the Save dialog".into());
        }
        target
    } else {
        if !scope.dirs.lock().unwrap().contains(&target) {
            return Err("that folder was not chosen in the export dialog".into());
        }
        let name = sanitize_name(&name).ok_or("invalid file name")?;
        let dest = target.join(name);
        scope.files.lock().unwrap().insert(dest.clone());
        dest
    };

    // Write next to the destination, then rename: a crash never leaves half a JPEG behind.
    let tmp = dest.with_extension(format!(
        "{}.part",
        dest.extension().and_then(|e| e.to_str()).unwrap_or("tmp")
    ));
    std::fs::write(&tmp, bytes).map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, &dest).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("could not save {}: {e}", dest.display())
    })
}

/// Show an exported file (selected) or folder in the platform file manager.
#[tauri::command]
pub fn reveal_path(scope: State<'_, ExportScope>, path: String) -> Result<(), String> {
    let path = PathBuf::from(path);
    if !scope.allows_reveal(&path) {
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

#[cfg(test)]
mod tests {
    use super::sanitize_name;

    #[test]
    fn file_names_cannot_escape_the_folder() {
        assert_eq!(sanitize_name("IMG_0001_fineretouch.jpg").as_deref(), Some("IMG_0001_fineretouch.jpg"));
        assert_eq!(sanitize_name("../evil.jpg"), None);
        assert_eq!(sanitize_name("a/b.jpg"), None);
        assert_eq!(sanitize_name("/etc/passwd"), None);
        assert_eq!(sanitize_name(".hidden"), None);
        assert_eq!(sanitize_name(""), None);
        assert_eq!(sanitize_name(".."), None);
    }
}
