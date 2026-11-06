//! FineRetouch desktop shell.
//!
//! * Serves the static Next.js export (`apps/web/out`) in a native webview.
//! * Starts the local AI engine (`binaries/fineretouch-engine`, built by
//!   `scripts/package_desktop.py`) as a Tauri sidecar on a free loopback port, hands it the
//!   bundled models and a per-launch token, and stops it when the app exits.
//! * Gives the editor native Save As / folder dialogs for exports.
//!
//! Nothing here needs the network: models, engine and UI all ship inside the installer.

mod engine;
mod files;
mod paths;

use serde_json::json;
use tauri::{RunEvent, WebviewWindowBuilder};

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(engine::EngineState::default())
        .manage(files::ExportScope::default())
        .invoke_handler(tauri::generate_handler![
            engine::engine_status,
            engine::restart_engine,
            files::choose_save_path,
            files::choose_export_folder,
            files::write_export,
            files::reveal_path,
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            let dirs = paths::AppPaths::resolve(&handle)?;
            let launch = engine::start(&handle, &dirs)?;

            // The editor reads this synchronously at module load (apps/web/src/lib/desktop.ts),
            // so it is injected before any page script runs instead of fetched over IPC.
            let info = json!({
                "engineUrl": launch.url,
                "engineToken": launch.token,
                "dataDir": dirs.data.to_string_lossy(),
                "version": app.package_info().version.to_string(),
                "platform": std::env::consts::OS,
            });
            let script = format!(
                "Object.defineProperty(window, '__FINERETOUCH_DESKTOP__', {{ value: Object.freeze({info}), configurable: false }});"
            );

            // The window is declared in tauri.conf.json with "create": false and built here,
            // so it gets the init script above.
            let config = app
                .config()
                .app
                .windows
                .iter()
                .find(|w| w.label == "main")
                .cloned()
                .ok_or("tauri.conf.json has no window labelled \"main\"")?;
            WebviewWindowBuilder::from_config(&handle, &config)?
                .initialization_script(&script)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build the FineRetouch desktop app")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                engine::stop(app);
            }
        });
}
