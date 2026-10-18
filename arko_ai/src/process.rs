//! Child processes that open no console window on Windows.
//!
//! On Windows a program without a console of its own, such as a windowed app,
//! gives every console program it starts a new console window. The window
//! flashes up for a short command and stays open for a long one.
//! [`NoConsoleWindow::no_console_window`] starts the program with
//! [`CREATE_NO_WINDOW`] instead, so it gets a console with no window. On other
//! systems it does nothing.
//!
//! It suits programs that run in the background with their input and output
//! redirected. Under a console app the program then gets a console of its
//! own, so it doesn't share the app's console window or its Ctrl-C.
//!
//! ```
//! use arko_ai::process::NoConsoleWindow;
//!
//! let mut git = std::process::Command::new("git");
//! git.args(["status", "--porcelain"]).no_console_window();
//! ```

/// Windows' process creation flag that starts a console program without a
/// console window.
///
/// `creation_flags` replaces the flags set before, so a command that needs
/// other flags as well passes them in the same call, as in
/// `command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW)`.
pub const CREATE_NO_WINDOW: u32 = 0x0800_0000;

/// Starts a [`std::process::Command`] or a [`tokio::process::Command`]
/// without a console window on Windows.
pub trait NoConsoleWindow {
    /// Starts the program without a console window on Windows, and does
    /// nothing on other systems.
    ///
    /// On Windows this sets the command's creation flags to
    /// [`CREATE_NO_WINDOW`], in place of any set before. Its documentation
    /// shows how to combine it with other flags.
    fn no_console_window(&mut self) -> &mut Self;
}

impl NoConsoleWindow for std::process::Command {
    fn no_console_window(&mut self) -> &mut Self {
        #[cfg(windows)]
        std::os::windows::process::CommandExt::creation_flags(self, CREATE_NO_WINDOW);
        self
    }
}

impl NoConsoleWindow for tokio::process::Command {
    fn no_console_window(&mut self) -> &mut Self {
        #[cfg(windows)]
        self.creation_flags(CREATE_NO_WINDOW);
        self
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    /// A Python program that prints whether its console is shared with the
    /// process whose ID it gets (`shared` or `own`) and whether the console
    /// has a visible window (`visible` or `hidden`).
    const CONSOLE_PROBE: &str = "import ctypes, sys
k = ctypes.windll.kernel32
u = ctypes.windll.user32
k.GetConsoleWindow.restype = ctypes.c_void_p
u.IsWindowVisible.argtypes = [ctypes.c_void_p]
ids = (ctypes.c_uint32 * 4096)()
count = k.GetConsoleProcessList(ids, 4096)
window = k.GetConsoleWindow()
shared = int(sys.argv[1]) in ids[:count]
visible = bool(window) and bool(u.IsWindowVisible(window))
print('shared' if shared else 'own', 'visible' if visible else 'hidden')";

    fn python() -> String {
        std::env::var("ARKO_TEST_PYTHON").unwrap_or_else(|_| "python".into())
    }

    #[test]
    fn the_flag_is_windows_create_no_window() {
        assert_eq!(
            CREATE_NO_WINDOW,
            windows_sys::Win32::System::Threading::CREATE_NO_WINDOW
        );
    }

    /// A program started without a console window gets a hidden console of
    /// its own instead of this process's console, and its output is read
    /// through the pipes as usual.
    #[test]
    fn a_std_command_gets_a_hidden_console_of_its_own() {
        let output = std::process::Command::new(python())
            .args(["-c", CONSOLE_PROBE, &std::process::id().to_string()])
            .stdin(std::process::Stdio::null())
            .no_console_window()
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "own hidden");
    }

    #[tokio::test]
    async fn a_tokio_command_gets_a_hidden_console_of_its_own() {
        let output = tokio::process::Command::new(python())
            .args(["-c", CONSOLE_PROBE, &std::process::id().to_string()])
            .stdin(std::process::Stdio::null())
            .no_console_window()
            .output()
            .await
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(String::from_utf8_lossy(&output.stdout).trim(), "own hidden");
    }
}
