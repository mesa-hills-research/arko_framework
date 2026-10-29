//! A logger for the tests, which keeps every record at info level and
//! above. A process has one logger, so every test of the crate shares it.

use std::sync::{Mutex, Once};

static LOG: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        LOG.lock()
            .unwrap()
            .push((record.level(), record.args().to_string()));
    }
    fn flush(&self) {}
}

/// Starts keeping records, if no test has yet.
pub(crate) fn install() {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Info);
    });
}

/// The records at `level` or above that mention `needle`, kept since the
/// logger was installed.
pub(crate) fn logged(level: log::Level, needle: &str) -> Vec<String> {
    install();
    LOG.lock()
        .unwrap()
        .iter()
        .filter(|(at, line)| *at <= level && line.contains(needle))
        .map(|(_, line)| line.clone())
        .collect()
}
