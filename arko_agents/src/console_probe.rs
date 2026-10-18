//! A Python program that reports on its console, for tests of commands
//! started without a console window on Windows.

/// Python code that prints `CONSOLE=<shared|own> <visible|hidden>`, saying
/// whether its console is this test process's console and whether that
/// console has a visible window.
pub(crate) fn code() -> String {
    format!(
        "import ctypes
kernel32 = ctypes.windll.kernel32
user32 = ctypes.windll.user32
kernel32.GetConsoleWindow.restype = ctypes.c_void_p
user32.IsWindowVisible.argtypes = [ctypes.c_void_p]
ids = (ctypes.c_uint32 * 4096)()
count = kernel32.GetConsoleProcessList(ids, 4096)
window = kernel32.GetConsoleWindow()
shared = {host} in ids[:count]
visible = bool(window) and bool(user32.IsWindowVisible(window))
print('CONSOLE=' + ('shared' if shared else 'own') + ' ' + ('visible' if visible else 'hidden'), flush=True)
",
        host = std::process::id()
    )
}

/// The Python interpreter, `python` unless `ARKO_TEST_PYTHON` names one.
pub(crate) fn python() -> String {
    std::env::var("ARKO_TEST_PYTHON").unwrap_or_else(|_| "python".into())
}
