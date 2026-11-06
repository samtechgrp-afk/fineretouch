// Release builds on Windows are GUI apps: no console window next to the editor.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    fineretouch_desktop_lib::run()
}
