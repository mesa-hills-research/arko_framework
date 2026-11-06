//! Positional reads and writes and atomic replacement, shared by the index files and
//! the embedding cache.
use crate::bench_v1::Error;
use std::{fs::File, io::Write, path::Path};

/// Reads exactly `buf.len()` bytes at `offset` without moving a shared cursor.
#[cfg(unix)]
pub(crate) fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::read_exact_at(file, buf, offset)
}
/// Reads exactly `buf.len()` bytes at `offset`.
#[cfg(windows)]
pub(crate) fn read_exact_at(
    file: &File,
    mut buf: &mut [u8],
    mut offset: u64,
) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt as _;
    while !buf.is_empty() {
        match file.seek_read(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Writes all of `buf` at `offset` without moving a shared cursor.
#[cfg(unix)]
pub(crate) fn write_all_at(file: &File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    std::os::unix::fs::FileExt::write_all_at(file, buf, offset)
}
/// Writes all of `buf` at `offset`.
#[cfg(windows)]
pub(crate) fn write_all_at(file: &File, mut buf: &[u8], mut offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt as _;
    while !buf.is_empty() {
        match file.seek_write(buf, offset) {
            Ok(0) => return Err(std::io::ErrorKind::WriteZero.into()),
            Ok(n) => {
                buf = &buf[n..];
                offset += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Replaces `dir/name` with `data` through a temporary file, fsynced, renamed over it, and
/// the directory fsynced (on Unix).
pub(crate) fn atomic_write(dir: &Path, name: &str, data: &[u8]) -> Result<(), Error> {
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(data)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, dir.join(name))?;
    crate::files::sync_dir(dir)?;
    Ok(())
}

/// The error when another process, or another handle in this one, holds the lock of an
/// index or watcher state directory. The errors of [`index2::build`](crate::index2::build),
/// [`search::index`](crate::search::index), [`bench_v1::open_watch`](crate::bench_v1::open_watch)
/// and `watch2::Watcher2::open` downcast to it then, and only then.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateInUse {
    what: &'static str,
}
impl std::fmt::Display for StateInUse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} already in use", self.what)
    }
}
impl std::error::Error for StateInUse {}

/// Takes `file`'s exclusive lock without waiting. `what` names the state it guards.
pub(crate) fn lock_or_in_use(file: &File, what: &'static str) -> Result<(), Error> {
    in_use_or(fs2::FileExt::try_lock_exclusive(file), what)
}

/// [`StateInUse`] when another holder has the lock, else the error as it is, so a
/// file system without working locks isn't taken for a busy state.
fn in_use_or(locked: std::io::Result<()>, what: &'static str) -> Result<(), Error> {
    match locked {
        Ok(()) => Ok(()),
        Err(e) if is_contended(&e) => Err(Box::new(StateInUse { what })),
        Err(e) => Err(format!("{what} can't be locked: {e}").into()),
    }
}

/// Whether a lock call failed because another holder has the lock.
pub(crate) fn is_contended(e: &std::io::Error) -> bool {
    e.raw_os_error().is_some() && e.raw_os_error() == fs2::lock_contended_error().raw_os_error()
}

pub(crate) fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
pub(crate) fn read_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

/// A little-endian reader over a byte slice that fails instead of panicking on
/// truncated input.
pub(crate) struct Cursor<'a> {
    pub buf: &'a [u8],
    pub at: usize,
}
impl<'a> Cursor<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, at: 0 }
    }
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).ok_or("truncated file")?;
        let b = self.buf.get(self.at..end).ok_or("truncated file")?;
        self.at = end;
        Ok(b)
    }
    pub fn u32(&mut self) -> Result<u32, Error> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into()?))
    }
    pub fn string(&mut self) -> Result<String, Error> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8(self.bytes(n)?.to_vec())?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_held_lock_is_in_use() {
        let held = in_use_or(Err(fs2::lock_contended_error()), "index state").unwrap_err();
        assert!(held.downcast_ref::<StateInUse>().is_some(), "{held}");
        assert_eq!(held.to_string(), "index state already in use");
        let failed = in_use_or(Err(std::io::ErrorKind::Unsupported.into()), "index state");
        let failed = failed.unwrap_err();
        assert!(failed.downcast_ref::<StateInUse>().is_none(), "{failed}");
        assert!(!failed.to_string().contains("already in use"), "{failed}");
    }
}
