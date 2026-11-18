//! Replacing a file a person or a tool edits, in one step, without changing
//! who can read or write it.
//!
//! The new content goes to a new file beside the old one, which then takes
//! the old file's place, with a rename on Unix and `ReplaceFileW` on Windows.
//! An error or a crash leaves the old file as it was. On Unix the new file is
//! created readable by its owner alone, and before any content is written it
//! is checked to be the old file's owner's and given its group, extended
//! attributes and permissions. On Windows `ReplaceFileW` gives it the old
//! file's security descriptor and attributes as it takes its place.
//!
//! A file whose identity a new file can't have is refused and left as it was
//! ([`RewriteError::Kept`]): one with more than one hard link, one owned by
//! another user, one whose group, extended attribute or ACL can't be given to
//! the new file, one with an ACL on macOS, one in a directory the new file
//! can't be made in, and any file on a Unix system other than Linux, Android
//! and macOS, whose ACLs and attributes this doesn't read.

use std::fs::File;
use std::path::Path;

/// Why [`rewrite_file`] left a file as it was.
#[derive(Debug)]
pub(super) enum RewriteError {
    /// A new file in its place couldn't have its identity, as the sentence
    /// says.
    Kept(String),
    Io(std::io::Error),
}

impl From<std::io::Error> for RewriteError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Replaces the existing file at `path` with `bytes` (see the module). A
/// symbolic link is followed, so the file it names is replaced and the link
/// stays, and a file the process can't write is refused, as a write in place
/// would be.
pub(super) fn rewrite_file(path: &Path, bytes: &[u8]) -> Result<(), RewriteError> {
    use std::io::Write;
    rewrite_with(path, |file| file.write_all(bytes))
}

/// [`rewrite_file`], with `write` putting the content in the new file.
fn rewrite_with(
    path: &Path,
    write: impl FnOnce(&mut File) -> std::io::Result<()>,
) -> Result<(), RewriteError> {
    let target = std::fs::canonicalize(path)?;
    // (Opening it to write checks the access a write in place would need.)
    let original = std::fs::OpenOptions::new().write(true).open(&target)?;
    let links = system::hard_links(&original)?;
    if links > 1 {
        return Err(RewriteError::Kept(format!(
            "It has {links} hard links, and a new file in its place would leave the others with the old content."
        )));
    }
    let (_, temporary) = crate::utils::temporary_beside(&target)?;
    let mut new = system::create(&temporary).map_err(|error| {
        RewriteError::Kept(format!("A new file can't be made beside it ({error})."))
    })?;
    let written = system::give_identity(&original, &new).and_then(|()| {
        write(&mut new)?;
        new.sync_all()?;
        Ok(())
    });
    drop(new);
    drop(original);
    let written =
        written.and_then(|()| system::take_place(&temporary, &target).map_err(Into::into));
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written
}

#[cfg(unix)]
mod system {
    use super::RewriteError;
    use std::fs::File;
    use std::path::Path;

    pub(super) fn hard_links(file: &File) -> std::io::Result<u64> {
        use std::os::unix::fs::MetadataExt;
        Ok(file.metadata()?.nlink())
    }

    /// A new, empty file at `path`, readable by its owner alone.
    pub(super) fn create(path: &Path) -> std::io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
    }

    /// Gives `new`, still empty, the owner, group, extended attributes and
    /// permissions of `original`, or says why it can't have them.
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    pub(super) fn give_identity(original: &File, new: &File) -> Result<(), RewriteError> {
        use std::os::unix::fs::MetadataExt;
        let (was, made) = (original.metadata()?, new.metadata()?);
        if made.uid() != was.uid() {
            return Err(RewriteError::Kept(
                "It belongs to another user, and a new file in its place would belong to this one."
                    .to_string(),
            ));
        }
        if made.gid() != was.gid() && std::os::unix::fs::fchown(new, None, Some(was.gid())).is_err()
        {
            return Err(RewriteError::Kept(
                "Its group can't be given to a new file in its place.".to_string(),
            ));
        }
        #[cfg(target_vendor = "apple")]
        if attributes::has_acl(original)? || attributes::has_acl(new)? {
            return Err(RewriteError::Kept(
                "It has an ACL, which a new file in its place can't be given.".to_string(),
            ));
        }
        attributes::copy(original, new)?;
        new.set_permissions(was.permissions())?;
        Ok(())
    }

    /// On other Unix systems a file's ACL and extended attributes aren't
    /// read, so a new file can't be given them.
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    pub(super) fn give_identity(_original: &File, _new: &File) -> Result<(), RewriteError> {
        Err(RewriteError::Kept(
            "On this system its ACL and extended attributes can't be given to a new file in its place."
                .to_string(),
        ))
    }

    pub(super) fn take_place(temporary: &Path, target: &Path) -> std::io::Result<()> {
        std::fs::rename(temporary, target)
    }

    /// Extended attributes, through each system's calls for them.
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    mod attributes {
        use super::RewriteError;
        use nix::libc::{self, c_char, c_int, c_void, size_t, ssize_t};
        use std::ffi::{CStr, CString};
        use std::fs::File;
        use std::os::fd::AsRawFd;

        /// The error an attribute that isn't there gives.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        const MISSING: c_int = libc::ENODATA;
        #[cfg(target_vendor = "apple")]
        const MISSING: c_int = libc::ENOATTR;

        // SAFETY (for the four calls): each takes the open file's
        // descriptor, a NUL-terminated name, and a buffer with its length, or
        // none with a length of 0, in which case it writes nothing.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        unsafe fn list(fd: c_int, names: *mut c_char, size: size_t) -> ssize_t {
            unsafe { libc::flistxattr(fd, names, size) }
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        unsafe fn get(fd: c_int, name: *const c_char, value: *mut c_void, size: size_t) -> ssize_t {
            unsafe { libc::fgetxattr(fd, name, value, size) }
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        unsafe fn set(fd: c_int, name: *const c_char, value: *const c_void, size: size_t) -> c_int {
            unsafe { libc::fsetxattr(fd, name, value, size, 0) }
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        unsafe fn remove(fd: c_int, name: *const c_char) -> c_int {
            unsafe { libc::fremovexattr(fd, name) }
        }
        #[cfg(target_vendor = "apple")]
        unsafe fn list(fd: c_int, names: *mut c_char, size: size_t) -> ssize_t {
            unsafe { libc::flistxattr(fd, names, size, 0) }
        }
        #[cfg(target_vendor = "apple")]
        unsafe fn get(fd: c_int, name: *const c_char, value: *mut c_void, size: size_t) -> ssize_t {
            unsafe { libc::fgetxattr(fd, name, value, size, 0, 0) }
        }
        #[cfg(target_vendor = "apple")]
        unsafe fn set(fd: c_int, name: *const c_char, value: *const c_void, size: size_t) -> c_int {
            unsafe { libc::fsetxattr(fd, name, value, size, 0, 0) }
        }
        #[cfg(target_vendor = "apple")]
        unsafe fn remove(fd: c_int, name: *const c_char) -> c_int {
            unsafe { libc::fremovexattr(fd, name, 0) }
        }

        /// The names of `file`'s extended attributes.
        fn names(file: &File) -> std::io::Result<Vec<CString>> {
            loop {
                let size = unsafe { list(file.as_raw_fd(), std::ptr::null_mut(), 0) };
                if size < 0 {
                    let error = std::io::Error::last_os_error();
                    return match error.raw_os_error() {
                        Some(libc::ENOTSUP) => Ok(Vec::new()),
                        _ => Err(error),
                    };
                }
                let mut names = vec![0u8; size as usize];
                let got = unsafe { list(file.as_raw_fd(), names.as_mut_ptr().cast(), names.len()) };
                if got < 0 {
                    let error = std::io::Error::last_os_error();
                    // (The list grew since its size was asked.)
                    if error.raw_os_error() == Some(libc::ERANGE) {
                        continue;
                    }
                    return Err(error);
                }
                names.truncate(got as usize);
                return Ok(names
                    .split(|byte| *byte == 0)
                    .filter(|name| !name.is_empty())
                    .map(|name| CString::new(name).expect("a name has no NUL"))
                    .collect());
            }
        }

        /// The value of `file`'s attribute `name`, if it has it.
        fn value(file: &File, name: &CStr) -> std::io::Result<Option<Vec<u8>>> {
            loop {
                let size = unsafe { get(file.as_raw_fd(), name.as_ptr(), std::ptr::null_mut(), 0) };
                let missing = |error: std::io::Error| match error.raw_os_error() {
                    Some(MISSING) => Ok(None),
                    _ => Err(error),
                };
                if size < 0 {
                    return missing(std::io::Error::last_os_error());
                }
                let mut value = vec![0u8; size as usize];
                let got = unsafe {
                    get(
                        file.as_raw_fd(),
                        name.as_ptr(),
                        value.as_mut_ptr().cast(),
                        value.len(),
                    )
                };
                if got < 0 {
                    let error = std::io::Error::last_os_error();
                    if error.raw_os_error() == Some(libc::ERANGE) {
                        continue;
                    }
                    return missing(error);
                }
                value.truncate(got as usize);
                return Ok(Some(value));
            }
        }

        /// What an error calls the attribute `name`.
        fn named(name: &CStr) -> String {
            let name = name.to_string_lossy();
            if name.starts_with("system.posix_acl") {
                "Its ACL".to_string()
            } else {
                format!("Its extended attribute {name}")
            }
        }

        /// Gives `new` every extended attribute of `original`, and takes
        /// away the ones `original` lacks, such as a default ACL of the
        /// directory.
        pub(super) fn copy(original: &File, new: &File) -> Result<(), RewriteError> {
            let names = names(original)?;
            for name in &names {
                let Some(wanted) = value(original, name)? else {
                    continue;
                };
                if value(new, name)?.as_ref() == Some(&wanted) {
                    continue;
                }
                let set = unsafe {
                    set(
                        new.as_raw_fd(),
                        name.as_ptr(),
                        wanted.as_ptr().cast(),
                        wanted.len(),
                    )
                };
                if set != 0 && !system_managed(name) {
                    return Err(RewriteError::Kept(format!(
                        "{} can't be given to a new file in its place.",
                        named(name)
                    )));
                }
            }
            for name in self::names(new)? {
                if names.contains(&name) {
                    continue;
                }
                let removed = unsafe { remove(new.as_raw_fd(), name.as_ptr()) };
                if removed != 0 && !system_managed(&name) {
                    return Err(RewriteError::Kept(format!(
                        "A new file in its place gets an attribute it lacks ({}), which can't be taken away.",
                        name.to_string_lossy()
                    )));
                }
            }
            Ok(())
        }

        /// Whether the system sets the attribute `name` itself, and it
        /// grants no access, as macOS's `com.apple.provenance` does.
        fn system_managed(name: &CStr) -> bool {
            cfg!(target_vendor = "apple") && name.to_bytes() == b"com.apple.provenance"
        }

        #[cfg(target_vendor = "apple")]
        unsafe extern "C" {
            fn acl_get_fd(fd: c_int) -> *mut c_void;
            fn acl_free(acl: *mut c_void) -> c_int;
        }

        /// Whether `file` has an ACL. On macOS an ACL isn't an extended
        /// attribute.
        #[cfg(target_vendor = "apple")]
        pub(super) fn has_acl(file: &File) -> std::io::Result<bool> {
            // SAFETY: the descriptor is the open file's, and an ACL it
            // returns is freed once.
            let acl = unsafe { acl_get_fd(file.as_raw_fd()) };
            if acl.is_null() {
                let error = std::io::Error::last_os_error();
                return match error.raw_os_error() {
                    Some(libc::ENOENT) | Some(libc::ENOTSUP) => Ok(false),
                    _ => Err(error),
                };
            }
            unsafe { acl_free(acl) };
            Ok(true)
        }
    }
}

#[cfg(windows)]
mod system {
    use super::RewriteError;
    use std::fs::File;
    use std::path::Path;

    pub(super) fn hard_links(file: &File) -> std::io::Result<u64> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Storage::FileSystem::{
            BY_HANDLE_FILE_INFORMATION, GetFileInformationByHandle,
        };
        // SAFETY: the handle is the open file's, and the call writes a
        // struct of the type it is given.
        let mut information: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut information) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(information.nNumberOfLinks.into())
    }

    pub(super) fn create(path: &Path) -> std::io::Result<File> {
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
    }

    /// `ReplaceFileW` gives the new file the old one's security descriptor
    /// and attributes as it takes its place.
    pub(super) fn give_identity(_original: &File, _new: &File) -> Result<(), RewriteError> {
        Ok(())
    }

    /// Puts `temporary` in `target`'s place with `ReplaceFileW`, which keeps
    /// `target` under a backup name until the swap is done, and puts it back
    /// when the swap fails part way.
    pub(super) fn take_place(temporary: &Path, target: &Path) -> std::io::Result<()> {
        use std::os::windows::ffi::OsStrExt;
        use windows_sys::Win32::Storage::FileSystem::ReplaceFileW;
        let wide =
            |path: &Path| -> Vec<u16> { path.as_os_str().encode_wide().chain([0]).collect() };
        let (_, backup) = crate::utils::temporary_beside(target)?;
        let (replaced, replacement, kept) = (wide(target), wide(temporary), wide(&backup));
        // SAFETY: the three names are NUL-terminated, and the last two
        // arguments are reserved and null.
        let swapped = unsafe {
            ReplaceFileW(
                replaced.as_ptr(),
                replacement.as_ptr(),
                kept.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if swapped != 0 {
            let _ = std::fs::remove_file(&backup);
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if !target.exists() && backup.exists() {
            let _ = std::fs::rename(&backup, target);
        }
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Folder(std::path::PathBuf);

    impl Folder {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!("arko-rewrite-{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        fn names(&self) -> Vec<String> {
            let mut names: Vec<String> = std::fs::read_dir(&self.0)
                .unwrap()
                .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            names.sort();
            names
        }
    }

    impl Drop for Folder {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The file is replaced whole, so a reader that opened it before keeps
    /// the old content, and nothing is left beside it.
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    ))]
    #[test]
    fn a_file_is_replaced_whole() {
        use std::io::Read;
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        let mut reader = File::open(&path).unwrap();
        rewrite_file(&path, b"new").unwrap();
        let mut read = String::new();
        reader.read_to_string(&mut read).unwrap();
        assert_eq!(read, "old");
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        drop(reader);
        assert_eq!(folder.names(), ["file.txt"]);
    }

    /// A file with a second hard link is refused and left as it was.
    #[test]
    fn a_file_with_another_link_is_refused() {
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        std::fs::hard_link(&path, folder.0.join("link.txt")).unwrap();
        let refused = rewrite_file(&path, b"new");
        assert!(matches!(refused, Err(RewriteError::Kept(_))), "{refused:?}");
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
        assert_eq!(folder.names(), ["file.txt", "link.txt"]);
    }

    /// The group, permissions and extended attributes stay, and a
    /// private file's new content is never readable by others, since the
    /// new file has them before any of it is written.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn the_group_permissions_and_attributes_stay() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        let groups = std::process::Command::new("id").arg("-G").output().unwrap();
        let groups: Vec<u32> = String::from_utf8(groups.stdout)
            .unwrap()
            .split_whitespace()
            .map(|group| group.parse().unwrap())
            .collect();
        for mode in [0o600, 0o640, 0o755] {
            for group in groups.iter().take(2) {
                std::os::unix::fs::chown(&path, None, Some(*group)).unwrap();
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
                // The new file, as it is before its content is written.
                let original = File::open(&path).unwrap();
                let new = system::create(&folder.0.join("new")).unwrap();
                system::give_identity(&original, &new).unwrap();
                let made = new.metadata().unwrap();
                assert_eq!(
                    (made.mode() & 0o7777, made.gid(), made.len()),
                    (mode, *group, 0)
                );
                std::fs::remove_file(folder.0.join("new")).unwrap();
                rewrite_file(&path, format!("{mode:o} {group}").as_bytes()).unwrap();
                let now = std::fs::metadata(&path).unwrap();
                assert_eq!((now.mode() & 0o7777, now.gid()), (mode, *group));
            }
        }
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        // (Python, which the tests use elsewhere, has the calls for them.)
        let python = std::env::var("ARKO_TEST_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let attribute = |code: &str| {
            std::process::Command::new(&python)
                .args(["-I", "-c", code])
                .arg(&path)
                .output()
        };
        let tagged = attribute("import os, sys; os.setxattr(sys.argv[1], 'user.origin', b'kept')");
        match tagged {
            Ok(output) if output.status.success() => {
                rewrite_file(&path, b"tagged").unwrap();
                let read = attribute(
                    "import os, sys; sys.stdout.write(os.getxattr(sys.argv[1], 'user.origin').decode())",
                )
                .unwrap();
                assert_eq!(read.stdout, b"kept");
            }
            _ => eprintln!(
                "skipped user attributes, since Python isn't here or the file system has none"
            ),
        }
        let acl = std::process::Command::new("setfacl")
            .args(["-m", "u:nobody:r"])
            .arg(&path)
            .status();
        match acl {
            Ok(status) if status.success() => {
                rewrite_file(&path, b"with an acl").unwrap();
                let listed = std::process::Command::new("getfacl")
                    .arg(&path)
                    .output()
                    .unwrap();
                let listed = String::from_utf8(listed.stdout).unwrap();
                assert!(listed.contains("user:nobody:r--"), "{listed}");
                assert_eq!(std::fs::read(&path).unwrap(), b"with an acl");
            }
            _ => eprintln!(
                "skipped the ACL, since setfacl isn't here or the file system has no ACLs"
            ),
        }
        assert_eq!(folder.names(), ["file.txt"]);
    }

    /// A directory's default ACL, which a new file in it gets, is taken
    /// away from the new file when the old one has none.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn a_default_acl_of_the_directory_is_not_added() {
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        let set = std::process::Command::new("setfacl")
            .args(["-d", "-m", "u:nobody:r"])
            .arg(&folder.0)
            .status();
        if !matches!(set, Ok(status) if status.success()) {
            eprintln!("skipped, since setfacl isn't here or the file system has no ACLs");
            return;
        }
        rewrite_file(&path, b"new").unwrap();
        let listed = std::process::Command::new("getfacl")
            .arg(&path)
            .output()
            .unwrap();
        let listed = String::from_utf8(listed.stdout).unwrap();
        assert!(!listed.contains("nobody"), "{listed}");
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    /// On macOS, extended attributes stay, and a file with an ACL is
    /// refused and left as it was.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn macos_attributes_stay_and_an_acl_is_refused() {
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        let tagged = std::process::Command::new("xattr")
            .args(["-w", "org.example.origin", "kept"])
            .arg(&path)
            .status()
            .unwrap();
        assert!(tagged.success());
        rewrite_file(&path, b"new").unwrap();
        let read = std::process::Command::new("xattr")
            .args(["-p", "org.example.origin"])
            .arg(&path)
            .output()
            .unwrap();
        assert_eq!(String::from_utf8(read.stdout).unwrap().trim(), "kept");
        let denied = std::process::Command::new("chmod")
            .args(["+a", "everyone deny read"])
            .arg(&path)
            .status()
            .unwrap();
        assert!(denied.success());
        let refused = rewrite_file(&path, b"newer");
        assert!(matches!(refused, Err(RewriteError::Kept(_))), "{refused:?}");
        let _ = std::process::Command::new("chmod")
            .args(["-N"])
            .arg(&path)
            .status();
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert_eq!(folder.names(), ["file.txt"]);
    }

    /// On Windows, the file keeps its own access rules.
    #[cfg(windows)]
    #[test]
    fn windows_access_rules_stay() {
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        // The built-in Guests group, by its SID.
        let granted = std::process::Command::new("icacls")
            .arg(&path)
            .args(["/grant", "*S-1-5-32-546:(R)"])
            .output()
            .unwrap();
        assert!(granted.status.success(), "{granted:?}");
        let rules = || {
            let listed = std::process::Command::new("icacls")
                .arg(&path)
                .output()
                .unwrap();
            String::from_utf8_lossy(&listed.stdout)
                .lines()
                .map(|line| {
                    line.replace(&path.display().to_string(), "")
                        .trim()
                        .to_string()
                })
                .filter(|line| !line.is_empty() && !line.starts_with("Successfully"))
                .collect::<Vec<_>>()
        };
        let before = rules();
        assert!(
            before
                .iter()
                .any(|rule| rule.contains("S-1-5-32-546") || rule.contains("Guests")),
            "{before:?}"
        );
        rewrite_file(&path, b"new").unwrap();
        assert_eq!(rules(), before);
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert_eq!(folder.names(), ["file.txt"]);
    }

    /// A directory the new file can't be made in is refused, and the file
    /// is left as it was.
    #[cfg(unix)]
    #[test]
    fn a_directory_without_room_for_a_new_file_is_refused() {
        use std::os::unix::fs::PermissionsExt;
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        std::fs::set_permissions(&folder.0, std::fs::Permissions::from_mode(0o555)).unwrap();
        let refused = rewrite_file(&path, b"new");
        std::fs::set_permissions(&folder.0, std::fs::Permissions::from_mode(0o755)).unwrap();
        // (A process that ignores permissions makes the file anyway.)
        if std::fs::read(&path).unwrap() == b"old" {
            assert!(matches!(refused, Err(RewriteError::Kept(_))), "{refused:?}");
        }
        assert_eq!(folder.names(), ["file.txt"]);
    }

    /// A disk that fills while the new file is written leaves the old one as
    /// it was, and nothing beside it.
    #[cfg(any(
        target_os = "linux",
        target_os = "android",
        target_vendor = "apple",
        windows
    ))]
    #[test]
    fn a_full_disk_leaves_the_file_as_it_was() {
        use std::io::Write;
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old content").unwrap();
        let refused = rewrite_with(&path, |file| {
            file.write_all(b"part of the new")?;
            Err(std::io::Error::new(
                std::io::ErrorKind::StorageFull,
                "no space left",
            ))
        });
        assert!(matches!(refused, Err(RewriteError::Io(_))), "{refused:?}");
        assert_eq!(std::fs::read(&path).unwrap(), b"old content");
        assert_eq!(folder.names(), ["file.txt"]);
    }
}
