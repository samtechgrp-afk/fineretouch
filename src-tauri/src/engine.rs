//! Lifecycle of the local AI engine sidecar (`fineretouch-engine`).
//!
//! Start-up: pick a free 127.0.0.1 port, generate a per-launch token, spawn the sidecar with
//! the bundled models, then poll `GET /health` in the background and emit `engine://ready`.
//! The editor does not wait for that: it polls `/health` itself and switches its AI features
//! on once the engine answers (a one-file engine needs a few seconds to unpack on first run).
//!
//! The engine also watches this process (`--parent-pid`) and exits by itself if the shell
//! dies without running [`stop`], so no orphaned engine keeps a port open.
//!
//! Development: set `FINERETOUCH_ENGINE_URL=http://localhost:8000` to use an engine started
//! by hand (`uvicorn app.main:app` in services/ai) instead of spawning the sidecar.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::Path;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

use crate::paths::AppPaths;

/// Name of the sidecar in tauri.conf.json `bundle.externalBin` (without the target triple).
const SIDECAR: &str = "fineretouch-engine";
/// Webview origins: macOS/Linux use tauri://localhost, Windows http(s)://tauri.localhost,
/// and `tauri dev` loads the Next dev server.
const WEBVIEW_ORIGINS: &str =
    "tauri://localhost,http://tauri.localhost,https://tauri.localhost,http://localhost:3000";
/// First launch of a one-file engine unpacks ~200 MB; give slow disks time.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum RunState {
    Starting,
    Ready,
    Stopped,
    External,
}

#[derive(Debug, Clone, Serialize)]
pub struct EngineStatus {
    pub state: RunState,
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

pub struct Launch {
    pub url: String,
    pub token: String,
}

#[derive(Default)]
pub struct EngineState(Mutex<Inner>);

#[derive(Default)]
struct Inner {
    child: Option<CommandChild>,
    port: u16,
    token: String,
    url: String,
    state: Option<RunState>,
    message: Option<String>,
    /// Bumped on every spawn so events from a replaced child are ignored.
    generation: u64,
    paths: Option<AppPaths>,
}

impl Inner {
    fn status(&self) -> EngineStatus {
        EngineStatus {
            state: self.state.unwrap_or(RunState::Stopped),
            url: self.url.clone(),
            message: self.message.clone(),
        }
    }
}

fn random_token() -> Result<String, String> {
    let mut bytes = [0u8; 24];
    getrandom::fill(&mut bytes).map_err(|e| format!("no OS randomness: {e}"))?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

fn free_port() -> std::io::Result<u16> {
    // Bound and released immediately; the engine binds it a moment later. The window for
    // another program to grab it in between is tiny, and a restart reuses the same port.
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    Ok(listener.local_addr()?.port())
}

/// Start the engine (or adopt `FINERETOUCH_ENGINE_URL`) and return what the webview needs.
pub fn start(app: &AppHandle, paths: &AppPaths) -> Result<Launch, Box<dyn std::error::Error>> {
    let state = app.state::<EngineState>();
    let mut inner = state.0.lock().unwrap();
    inner.paths = Some(paths.clone());

    if let Ok(url) = std::env::var("FINERETOUCH_ENGINE_URL") {
        inner.url = url.trim_end_matches('/').to_string();
        inner.token = std::env::var("FINERETOUCH_ENGINE_TOKEN").unwrap_or_default();
        inner.state = Some(RunState::External);
        return Ok(Launch { url: inner.url.clone(), token: inner.token.clone() });
    }

    inner.port = free_port()?;
    inner.token = random_token()?;
    inner.url = format!("http://127.0.0.1:{}", inner.port);
    spawn_locked(app, &mut inner);
    Ok(Launch { url: inner.url.clone(), token: inner.token.clone() })
}

/// Spawn the sidecar. Failures are recorded in the state (and the log) rather than
/// aborting start-up: the editor still works without AI, with the GPU preview engine.
fn spawn_locked(app: &AppHandle, inner: &mut Inner) {
    inner.generation += 1;
    let generation = inner.generation;
    let paths = inner.paths.clone().expect("paths are set before spawning");
    let console_log = paths.logs.join("engine-console.log");

    let command = match app.shell().sidecar(SIDECAR) {
        Ok(cmd) => cmd,
        Err(err) => {
            fail_locked(inner, &console_log, format!("engine binary missing: {err}"));
            return;
        }
    };
    let command = command
        .args([
            "--host".to_string(),
            "127.0.0.1".to_string(),
            "--port".to_string(),
            inner.port.to_string(),
            "--parent-pid".to_string(),
            std::process::id().to_string(),
            "--models-dir".to_string(),
            paths.models.to_string_lossy().into_owned(),
            "--data-dir".to_string(),
            paths.data.to_string_lossy().into_owned(),
        ])
        .envs([
            ("FINERETOUCH_ENGINE_TOKEN".to_string(), inner.token.clone()),
            ("CORS_ORIGINS".to_string(), WEBVIEW_ORIGINS.to_string()),
            ("PYTHONUNBUFFERED".to_string(), "1".to_string()),
            ("TMPDIR".to_string(), paths.cache.to_string_lossy().into_owned()),
        ]);

    match command.spawn() {
        Ok((mut events, child)) => {
            append_log(&console_log, &format!("--- engine pid {} on {}", child.pid(), inner.url));
            inner.child = Some(child);
            inner.state = Some(RunState::Starting);
            inner.message = None;

            let app_events = app.clone();
            tauri::async_runtime::spawn(async move {
                while let Some(event) = events.recv().await {
                    match event {
                        CommandEvent::Stdout(line) | CommandEvent::Stderr(line) => {
                            append_log(&console_log, String::from_utf8_lossy(&line).trim_end());
                        }
                        CommandEvent::Error(err) => append_log(&console_log, &format!("error: {err}")),
                        CommandEvent::Terminated(payload) => {
                            let msg = format!("engine exited (code {:?}, signal {:?})", payload.code, payload.signal);
                            append_log(&console_log, &msg);
                            let state = app_events.state::<EngineState>();
                            let mut inner = state.0.lock().unwrap();
                            if inner.generation == generation {
                                inner.child = None;
                                inner.state = Some(RunState::Stopped);
                                inner.message = Some(msg);
                                let _ = app_events.emit("engine://stopped", inner.status());
                            }
                            break;
                        }
                        _ => {}
                    }
                }
            });

            let app_health = app.clone();
            let (port, token) = (inner.port, inner.token.clone());
            std::thread::spawn(move || wait_until_healthy(app_health, generation, port, token));
        }
        Err(err) => fail_locked(inner, &console_log, format!("could not start engine: {err}")),
    }
}

fn fail_locked(inner: &mut Inner, log: &Path, message: String) {
    append_log(log, &message);
    inner.child = None;
    inner.state = Some(RunState::Stopped);
    inner.message = Some(message);
}

fn wait_until_healthy(app: AppHandle, generation: u64, port: u16, token: String) {
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    while Instant::now() < deadline {
        {
            let state = app.state::<EngineState>();
            let inner = state.0.lock().unwrap();
            if inner.generation != generation || inner.state != Some(RunState::Starting) {
                return; // replaced, stopped or crashed meanwhile
            }
        }
        if health_ok(port, &token) {
            let state = app.state::<EngineState>();
            let mut inner = state.0.lock().unwrap();
            if inner.generation == generation {
                inner.state = Some(RunState::Ready);
                let _ = app.emit("engine://ready", inner.status());
            }
            return;
        }
        std::thread::sleep(Duration::from_millis(400));
    }
    let state = app.state::<EngineState>();
    let mut inner = state.0.lock().unwrap();
    if inner.generation == generation && inner.state == Some(RunState::Starting) {
        inner.message = Some(format!("engine did not answer within {}s", STARTUP_TIMEOUT.as_secs()));
    }
}

/// Minimal HTTP/1.0 probe so the shell needs no HTTP client dependency.
fn health_ok(port: u16, token: &str) -> bool {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    let Ok(mut stream) = TcpStream::connect_timeout(&addr, Duration::from_millis(500)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    let request = format!(
        "GET /health HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\nX-FineRetouch-Token: {token}\r\nConnection: close\r\n\r\n"
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut head = [0u8; 16];
    match stream.read(&mut head) {
        Ok(n) => {
            let line = String::from_utf8_lossy(&head[..n]);
            line.starts_with("HTTP/1.1 200") || line.starts_with("HTTP/1.0 200")
        }
        Err(_) => false,
    }
}

fn append_log(path: &Path, line: &str) {
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Kill the sidecar (app exit). The engine's parent watchdog covers the one-file bootloader
/// case where killing the bootloader leaves its Python child running for a moment.
pub fn stop(app: &AppHandle) {
    let state = app.state::<EngineState>();
    let mut inner = state.0.lock().unwrap();
    inner.generation += 1;
    if let Some(child) = inner.child.take() {
        let _ = child.kill();
    }
    if inner.state != Some(RunState::External) {
        inner.state = Some(RunState::Stopped);
    }
}

#[tauri::command]
pub fn engine_status(state: State<'_, EngineState>) -> EngineStatus {
    state.0.lock().unwrap().status()
}

/// Restart on the same port and token, so the URL the webview already has stays valid.
#[tauri::command]
pub async fn restart_engine(app: AppHandle) -> Result<EngineStatus, String> {
    {
        let state = app.state::<EngineState>();
        let mut inner = state.0.lock().unwrap();
        if inner.state == Some(RunState::External) {
            return Ok(inner.status());
        }
        inner.generation += 1;
        if let Some(child) = inner.child.take() {
            child.kill().map_err(|e| e.to_string())?;
        }
        inner.state = Some(RunState::Stopped);
    }
    // Let the old process release the port (its watchdog or the kill ends it).
    let port = app.state::<EngineState>().0.lock().unwrap().port;
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline && TcpListener::bind(("127.0.0.1", port)).is_err() {
        std::thread::sleep(Duration::from_millis(200));
    }
    let state = app.state::<EngineState>();
    let mut inner = state.0.lock().unwrap();
    spawn_locked(&app, &mut inner);
    Ok(inner.status())
}
