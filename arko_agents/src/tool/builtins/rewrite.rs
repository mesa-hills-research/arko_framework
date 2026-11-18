//! Replacing a file a person or a tool edits, in one step, without changing
//! who can read or write it.
//!
//! The new content goes to a new file beside the old one, which then takes
//! the old file's place, with a rename on Unix and `ReplaceFileW` on Windows.
//! An error or a crash leaves the old file as it was. The new file is given
//! the old one's properties before any content is written. A file is
//! replaced only when every property observed of it is one this keeps or
//! knowingly ignores, and anything else, a property that can't be read
//! included, refuses the edit and leaves the file as it was
//! ([`RewriteError::Kept`]).
//!
//! | System | Property | Kept, ignored or refused |
//! |---|---|---|
//! | All | Type | A regular file is kept. A symbolic link is followed, so the file it names is replaced and the link stays. |
//! | All | Write access | Refused unless the file opens for writing, as a write in place needs (a read-only, immutable or append-only file). |
//! | All | Hard links | One is kept. More are refused, since a new file would leave the others with the old content. |
//! | All | Times, file ID, other processes' open handles and locks | Ignored. An edit changes the times, and the handles keep the old file, which is the point of a swap. |
//! | Unix | Owner | Kept when the new file, which the process owns, has it. Otherwise refused. |
//! | Unix | Group | Kept with `fchown`. Refused when that fails. |
//! | Unix | Permissions and the sticky bit | Kept with `fchmod`, before any content (until then the new file is readable by its owner alone). |
//! | Unix | Setuid and setgid bits | Refused, since writing a file can clear them. |
//! | Linux | Extended attributes `user.*`, the ACL (`system.posix_acl_access`) and the SELinux label (`security.selinux`) | Kept by copying them, or by taking them off the new file when it has them and the old one doesn't (a directory's default ACL). Refused when that fails. |
//! | Linux | Any other extended attribute | Refused (`trusted.*`, `security.capability`, `security.ima`, NFS and rich ACLs, and any yet unknown). |
//! | Linux | Inode flags noatime, nodump, sync, nocow, compress and nocompress | Kept with `FS_IOC_SETFLAGS` on the empty new file, and checked. |
//! | Linux | Inode flags extents, inline data, huge file and EOF blocks | Ignored, since they are how the file system lays the file out. |
//! | Linux | The encrypt flag (fscrypt) | Kept when the new file has it from its directory. Otherwise refused. |
//! | Linux | Any other inode flag | Refused. |
//! | macOS | Extended attributes | Kept by copying each, and taking off those the old file lacks. `com.apple.provenance`, which the system manages and which grants no access, is ignored where it can't be set or taken off. Any other failure is refused. |
//! | macOS | An ACL | Refused, on the old file or the new one. |
//! | macOS | File flags hidden and nodump | Kept with `fchflags`. |
//! | macOS | File flags compressed and tracked | Ignored. One is how the file is stored, and the other tracks a document's identity and grants no access. |
//! | macOS | Any other file flag | Refused. |
//! | Windows | Owner, group and DACL | Kept. The new file is created with them, its rules made its own and protected from the directory's, and `ReplaceFileW` gives it the old file's whole security descriptor. An owner the process can't give is refused. |
//! | Windows | Mandatory labels, central access policies and resource attributes | Refused, as is a failure to read them. |
//! | Windows | Audit rules | Ignored, and an edit may drop them. Reading them needs a privilege the process doesn't have, and they grant no access, so a save can't widen who reads the file. |
//! | Windows | Attributes hidden, system, not content indexed, archive and normal | Kept by `ReplaceFileW`. |
//! | Windows | Any other attribute | Refused, among them encrypted, compressed, sparse, a reparse point, offline, temporary, integrity stream, no scrub data, extended attributes, and a cloud file's pinned, unpinned and recall attributes. |
//! | Windows | Named data streams | Refused. |
//! | Other Unix systems | Everything | Refused, since their ACLs and attributes aren't read. |

use std::fs::File;
use std::path::Path;

/// Why [`rewrite_file`] left a file as it was.
#[derive(Debug)]
pub(super) enum RewriteError {
    /// A new file in its place couldn't have its properties, as the sentence
    /// says.
    Kept(String),
    Io(std::io::Error),
}

impl From<std::io::Error> for RewriteError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// A refusal saying `why`.
fn kept(why: impl Into<String>) -> RewriteError {
    RewriteError::Kept(why.into())
}

/// Replaces the existing file at `path` with `bytes` (see the module).
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
    system::check(&original, &target)?;
    let (_, temporary) = crate::utils::temporary_beside(&target)?;
    let mut new = system::create(&temporary, &target)
        .map_err(|error| kept(format!("A new file can't be made beside it ({error}).")))?;
    let written = system::give_properties(&original, &new).and_then(|()| {
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
    use super::{RewriteError, kept};
    use std::fs::File;
    use std::path::Path;

    /// What refuses the file before a new one is made.
    pub(super) fn check(original: &File, _target: &Path) -> Result<(), RewriteError> {
        use std::os::unix::fs::MetadataExt;
        let metadata = original.metadata()?;
        if !cfg!(any(
            target_os = "linux",
            target_os = "android",
            target_vendor = "apple"
        )) {
            return Err(kept(
                "On this system its ACL and extended attributes can't be read, so a new file in its place might not have them.",
            ));
        }
        if metadata.nlink() > 1 {
            return Err(kept(format!(
                "It has {} hard links, and a new file in its place would leave the others with the old content.",
                metadata.nlink()
            )));
        }
        if metadata.mode() & 0o6000 != 0 {
            return Err(kept(
                "It has the setuid or setgid bit, which writing a new file in its place can clear.",
            ));
        }
        #[cfg(target_vendor = "apple")]
        macos::check(original)?;
        Ok(())
    }

    /// A new, empty file at `path`, readable by its owner alone until
    /// [`give_properties`] gives it those of the file at `_like`.
    pub(super) fn create(path: &Path, _like: &Path) -> std::io::Result<File> {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(path)
    }

    /// Gives `new`, still empty, the owner, group, extended attributes, flags
    /// and permissions of `original`, or says why it can't have them.
    pub(super) fn give_properties(original: &File, new: &File) -> Result<(), RewriteError> {
        use std::os::unix::fs::MetadataExt;
        let (was, made) = (original.metadata()?, new.metadata()?);
        if made.uid() != was.uid() {
            return Err(kept(
                "It belongs to another user, and a new file in its place would belong to this one.",
            ));
        }
        if made.gid() != was.gid() && std::os::unix::fs::fchown(new, None, Some(was.gid())).is_err()
        {
            return Err(kept("Its group can't be given to a new file in its place."));
        }
        #[cfg(any(target_os = "linux", target_os = "android"))]
        linux::give_flags(original, new)?;
        #[cfg(target_vendor = "apple")]
        macos::give_flags(original, new)?;
        #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
        attributes::copy(original, new)?;
        new.set_permissions(was.permissions())?;
        Ok(())
    }

    pub(super) fn take_place(temporary: &Path, target: &Path) -> std::io::Result<()> {
        std::fs::rename(temporary, target)
    }

    /// Linux's inode flags.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    mod linux {
        use super::{RewriteError, kept};
        use nix::libc::{self, c_int};
        use std::fs::File;
        use std::os::fd::AsRawFd;

        /// noatime, nodump, sync, nocow, compress and nocompress.
        const KEPT: c_int = 0x80 | 0x40 | 0x8 | 0x0080_0000 | 0x4 | 0x400;
        /// extents, inline data, huge file and EOF blocks.
        const LAYOUT: c_int = 0x0008_0000 | 0x1000_0000 | 0x0004_0000 | 0x0040_0000;
        /// fscrypt's, which only the directory gives.
        const ENCRYPTED: c_int = 0x800;

        /// The file's inode flags, or none where its file system has none.
        fn flags(file: &File) -> std::io::Result<c_int> {
            let mut flags: c_int = 0;
            // SAFETY: the descriptor is the open file's, and the call writes
            // an int.
            if unsafe { libc::ioctl(file.as_raw_fd(), libc::FS_IOC_GETFLAGS, &mut flags) } == 0 {
                return Ok(flags);
            }
            let error = std::io::Error::last_os_error();
            match error.raw_os_error() {
                Some(libc::ENOTTY | libc::ENOTSUP | libc::EINVAL) => Ok(0),
                _ => Err(error),
            }
        }

        /// Gives `new`, still empty, the flags of `original` it keeps.
        pub(super) fn give_flags(original: &File, new: &File) -> Result<(), RewriteError> {
            let (theirs, mine) = (flags(original)?, flags(new)?);
            let unknown = (theirs | mine) & !(KEPT | LAYOUT | ENCRYPTED);
            if unknown != 0 {
                return Err(kept(format!(
                    "It or a new file beside it has inode flags ({unknown:#x}) a new file in its place can't keep."
                )));
            }
            if theirs & ENCRYPTED != mine & ENCRYPTED {
                return Err(kept(
                    "It is encrypted and a new file beside it isn't, or the other way round.",
                ));
            }
            let wanted = (mine & !KEPT) | (theirs & KEPT);
            if wanted != mine {
                // SAFETY: the descriptor is the new file's, and the call
                // reads an int.
                let set = unsafe { libc::ioctl(new.as_raw_fd(), libc::FS_IOC_SETFLAGS, &wanted) };
                if set != 0 || flags(new)? & KEPT != theirs & KEPT {
                    return Err(kept(format!(
                        "Its inode flags ({:#x}) can't be given to a new file in its place.",
                        theirs & KEPT
                    )));
                }
            }
            Ok(())
        }
    }

    /// macOS's file flags and ACLs.
    #[cfg(target_vendor = "apple")]
    mod macos {
        use super::{RewriteError, kept};
        use nix::libc::{self, c_int, c_uint, c_void};
        use std::fs::File;
        use std::os::fd::AsRawFd;

        /// hidden and nodump.
        const KEPT: c_uint = libc::UF_HIDDEN | libc::UF_NODUMP;
        /// compressed and tracked.
        const IGNORED: c_uint = libc::UF_COMPRESSED | libc::UF_TRACKED;

        fn flags(file: &File) -> std::io::Result<c_uint> {
            // SAFETY: the descriptor is the open file's, and the call writes
            // a stat.
            let mut stat: libc::stat = unsafe { std::mem::zeroed() };
            if unsafe { libc::fstat(file.as_raw_fd(), &mut stat) } != 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(stat.st_flags)
        }

        unsafe extern "C" {
            fn acl_get_fd(fd: c_int) -> *mut c_void;
            fn acl_free(acl: *mut c_void) -> c_int;
        }

        /// Whether `file` has an ACL. On macOS an ACL isn't an extended
        /// attribute.
        fn has_acl(file: &File) -> std::io::Result<bool> {
            // SAFETY: the descriptor is the open file's, and an ACL it
            // returns is freed once.
            let acl = unsafe { acl_get_fd(file.as_raw_fd()) };
            if acl.is_null() {
                let error = std::io::Error::last_os_error();
                return match error.raw_os_error() {
                    Some(libc::ENOENT) => Ok(false),
                    _ => Err(error),
                };
            }
            unsafe { acl_free(acl) };
            Ok(true)
        }

        pub(super) fn check(original: &File) -> Result<(), RewriteError> {
            let unknown = flags(original)? & !(KEPT | IGNORED);
            if unknown != 0 {
                return Err(kept(format!(
                    "It has file flags ({unknown:#x}) a new file in its place can't keep."
                )));
            }
            if has_acl(original)? {
                return Err(kept(
                    "It has an ACL, which a new file in its place can't be given.",
                ));
            }
            Ok(())
        }

        /// Gives `new`, still empty, the flags of `original` it keeps.
        pub(super) fn give_flags(original: &File, new: &File) -> Result<(), RewriteError> {
            if has_acl(new)? {
                return Err(kept(
                    "A new file beside it gets an ACL from its folder, which it lacks.",
                ));
            }
            let (theirs, mine) = (flags(original)? & KEPT, flags(new)?);
            let wanted = (mine & !KEPT) | theirs;
            // SAFETY: the descriptor is the new file's.
            if wanted != mine && unsafe { libc::fchflags(new.as_raw_fd(), wanted) } != 0 {
                return Err(kept(format!(
                    "Its file flags ({theirs:#x}) can't be given to a new file in its place."
                )));
            }
            let unknown = flags(new)? & !(KEPT | IGNORED);
            if unknown != 0 || flags(new)? & KEPT != theirs {
                return Err(kept(format!(
                    "A new file beside it has file flags ({unknown:#x}) it can't be given."
                )));
            }
            Ok(())
        }
    }

    /// Extended attributes, through each system's calls for them.
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    mod attributes {
        use super::{RewriteError, kept};
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

        /// Whether this keeps the attribute `name` (see the module's table).
        fn keeps(name: &CStr) -> bool {
            let name = name.to_bytes();
            !cfg!(any(target_os = "linux", target_os = "android"))
                || name.starts_with(b"user.")
                || name == b"system.posix_acl_access"
                || name == b"security.selinux"
        }

        /// Whether the system sets the attribute `name` itself, and it
        /// grants no access, as macOS's `com.apple.provenance` does.
        fn system_managed(name: &CStr) -> bool {
            cfg!(target_vendor = "apple") && name.to_bytes() == b"com.apple.provenance"
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
                "its ACL".to_string()
            } else {
                format!("the extended attribute {name}")
            }
        }

        /// Gives `new` the extended attributes of `original`, and takes away
        /// the ones `original` lacks, such as a directory's default ACL.
        pub(super) fn copy(original: &File, new: &File) -> Result<(), RewriteError> {
            let names = names(original)?;
            for name in &names {
                if !keeps(name) {
                    return Err(kept(format!(
                        "It has {}, which a new file in its place can't be given.",
                        named(name)
                    )));
                }
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
                    return Err(kept(format!(
                        "It has {}, which a new file in its place can't be given.",
                        named(name)
                    )));
                }
            }
            for name in self::names(new)? {
                if names.contains(&name) {
                    continue;
                }
                let removed =
                    keeps(&name) && unsafe { remove(new.as_raw_fd(), name.as_ptr()) } == 0;
                if !removed && !system_managed(&name) {
                    return Err(kept(format!(
                        "A new file beside it gets {}, which it lacks and which can't be taken off.",
                        named(&name)
                    )));
                }
            }
            Ok(())
        }

        /// Only the attributes the table lists are kept on Linux, so one
        /// nobody thought of refuses. (Most others need a privilege to
        /// set, so a test can't give a file one.)
        #[cfg(all(test, any(target_os = "linux", target_os = "android")))]
        #[test]
        fn only_the_listed_attributes_are_kept() {
            for (name, listed) in [
                ("user.origin", true),
                ("system.posix_acl_access", true),
                ("security.selinux", true),
                ("system.posix_acl_default", false),
                ("trusted.overlay.opaque", false),
                ("security.capability", false),
                ("security.ima", false),
                ("system.nfs4_acl", false),
                ("users.origin", false),
            ] {
                assert_eq!(keeps(&CString::new(name).unwrap()), listed, "{name}");
            }
        }
    }
}

#[cfg(windows)]
mod system {
    use super::{RewriteError, kept};
    use std::fs::File;
    use std::path::Path;
    use windows_sys::Win32::Storage::FileSystem as fs;

    /// The attributes `ReplaceFileW` keeps, or that every file has.
    const KEPT: u32 = fs::FILE_ATTRIBUTE_HIDDEN
        | fs::FILE_ATTRIBUTE_SYSTEM
        | fs::FILE_ATTRIBUTE_NOT_CONTENT_INDEXED
        | fs::FILE_ATTRIBUTE_ARCHIVE
        | fs::FILE_ATTRIBUTE_NORMAL;

    /// `path` as a NUL-terminated wide string.
    fn wide(path: &Path) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        path.as_os_str().encode_wide().chain([0]).collect()
    }

    /// What refuses the file before a new one is made.
    pub(super) fn check(original: &File, target: &Path) -> Result<(), RewriteError> {
        use std::os::windows::io::AsRawHandle;
        use windows_sys::Win32::Security::{
            ATTRIBUTE_SECURITY_INFORMATION, LABEL_SECURITY_INFORMATION, SCOPE_SECURITY_INFORMATION,
        };
        // SAFETY: the handle is the open file's, and the call writes a
        // struct of the type it is given.
        let mut information: fs::BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        if unsafe { fs::GetFileInformationByHandle(original.as_raw_handle(), &mut information) }
            == 0
        {
            return Err(std::io::Error::last_os_error().into());
        }
        if information.nNumberOfLinks > 1 {
            return Err(kept(format!(
                "It has {} hard links, and a new file in its place would leave the others with the old content.",
                information.nNumberOfLinks
            )));
        }
        let attributes = information.dwFileAttributes;
        if attributes & fs::FILE_ATTRIBUTE_ENCRYPTED != 0 {
            return Err(kept(
                "It is encrypted, and a new file beside it wouldn't be while its content is written.",
            ));
        }
        if attributes & !KEPT != 0 {
            return Err(kept(format!(
                "It has file attributes ({:#x}) a new file in its place can't keep.",
                attributes & !KEPT
            )));
        }
        // Labels, central access policies and resource attributes, each
        // read through its own flag. (Audit rules need a privilege.)
        let mut descriptor = security_of(
            target,
            LABEL_SECURITY_INFORMATION
                | SCOPE_SECURITY_INFORMATION
                | ATTRIBUTE_SECURITY_INFORMATION,
        )
        .map_err(|error| {
            kept(format!(
                "Its labels and access policies can't be read ({error})."
            ))
        })?;
        if rules(&mut descriptor, false)? > 0 {
            return Err(kept(
                "It has a mandatory label, an access policy or resource attributes, which a new file in its place can't keep.",
            ));
        }
        if has_named_streams(target)? {
            return Err(kept(
                "It has named data streams, which a new file in its place wouldn't have.",
            ));
        }
        Ok(())
    }

    /// Whether the file at `path` has a data stream besides its content.
    fn has_named_streams(path: &Path) -> std::io::Result<bool> {
        use windows_sys::Win32::Foundation::{ERROR_HANDLE_EOF, INVALID_HANDLE_VALUE};
        let name = wide(path);
        // SAFETY: the name is NUL-terminated, each call writes a struct of
        // the type it is given, and the search is closed once.
        unsafe {
            let mut stream: fs::WIN32_FIND_STREAM_DATA = std::mem::zeroed();
            let search = fs::FindFirstStreamW(
                name.as_ptr(),
                fs::FindStreamInfoStandard,
                (&mut stream as *mut fs::WIN32_FIND_STREAM_DATA).cast(),
                0,
            );
            if search == INVALID_HANDLE_VALUE {
                let error = std::io::Error::last_os_error();
                return match error.raw_os_error() {
                    Some(code) if code as u32 == ERROR_HANDLE_EOF => Ok(false),
                    _ => Err(error),
                };
            }
            let found = loop {
                let length = stream
                    .cStreamName
                    .iter()
                    .position(|unit| *unit == 0)
                    .unwrap_or(stream.cStreamName.len());
                if String::from_utf16_lossy(&stream.cStreamName[..length]) != "::$DATA" {
                    break Ok(true);
                }
                if fs::FindNextStreamW(
                    search,
                    (&mut stream as *mut fs::WIN32_FIND_STREAM_DATA).cast(),
                ) == 0
                {
                    // (Only the end of the list ends the search well.)
                    let error = std::io::Error::last_os_error();
                    break match error.raw_os_error() {
                        Some(code) if code as u32 == ERROR_HANDLE_EOF => Ok(false),
                        _ => Err(error),
                    };
                }
            };
            fs::FindClose(search);
            found
        }
    }

    /// The security information `wanted` of the file at `path`, as a
    /// self-relative security descriptor in a buffer aligned for it.
    pub(super) fn security_of(path: &Path, wanted: u32) -> std::io::Result<Vec<u64>> {
        use windows_sys::Win32::Security::GetFileSecurityW;
        let name = wide(path);
        let mut needed = 0u32;
        // SAFETY: with no buffer and a length of 0, the call only says the
        // length it needs.
        unsafe { GetFileSecurityW(name.as_ptr(), wanted, std::ptr::null_mut(), 0, &mut needed) };
        if needed == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let mut descriptor = vec![0u64; (needed as usize).div_ceil(8)];
        // SAFETY: the buffer holds `needed` bytes.
        let read = unsafe {
            GetFileSecurityW(
                name.as_ptr(),
                wanted,
                descriptor.as_mut_ptr().cast(),
                needed,
                &mut needed,
            )
        };
        if read == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(descriptor)
    }

    /// How many rules the DACL (`dacl`) or the SACL of `descriptor` has, with
    /// each DACL rule made the file's own instead of inherited.
    fn rules(descriptor: &mut [u64], dacl: bool) -> std::io::Result<u32> {
        use windows_sys::Win32::Security::{
            ACE_HEADER, ACL, GetAce, GetSecurityDescriptorDacl, GetSecurityDescriptorSacl,
            INHERITED_ACE,
        };
        let pointer = descriptor.as_mut_ptr().cast();
        let (mut present, mut list, mut defaulted) = (0, std::ptr::null_mut::<ACL>(), 0);
        // SAFETY: the descriptor is one GetFileSecurityW wrote, and its lists
        // and their rules are inside it.
        unsafe {
            let found = if dacl {
                GetSecurityDescriptorDacl(pointer, &mut present, &mut list, &mut defaulted)
            } else {
                GetSecurityDescriptorSacl(pointer, &mut present, &mut list, &mut defaulted)
            };
            if found == 0 {
                return Err(std::io::Error::last_os_error());
            }
            if present == 0 || list.is_null() {
                return Ok(0);
            }
            let count = u32::from((*list).AceCount);
            if dacl {
                for index in 0..count {
                    let mut ace = std::ptr::null_mut();
                    if GetAce(list, index, &mut ace) == 0 {
                        return Err(std::io::Error::last_os_error());
                    }
                    (*ace.cast::<ACE_HEADER>()).AceFlags &= !(INHERITED_ACE as u8);
                }
            }
            Ok(count)
        }
    }

    /// A new, empty file at `path` with the owner, group and DACL of the file
    /// at `like` from the start, so it is never open to more than `like` is.
    /// The DACL's rules are made the file's own and protected from the
    /// directory's, which would otherwise be added to them.
    pub(super) fn create(path: &Path, like: &Path) -> std::io::Result<File> {
        use std::os::windows::io::FromRawHandle;
        use windows_sys::Win32::Foundation::{GENERIC_WRITE, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::Security::{
            DACL_SECURITY_INFORMATION, GROUP_SECURITY_INFORMATION, OWNER_SECURITY_INFORMATION,
            SE_DACL_PROTECTED, SECURITY_ATTRIBUTES, SetSecurityDescriptorControl,
        };
        let mut descriptor = security_of(
            like,
            OWNER_SECURITY_INFORMATION | GROUP_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
        )?;
        rules(&mut descriptor, true)?;
        let pointer = descriptor.as_mut_ptr().cast();
        // SAFETY: the descriptor is the one GetFileSecurityW wrote.
        if unsafe { SetSecurityDescriptorControl(pointer, SE_DACL_PROTECTED, SE_DACL_PROTECTED) }
            == 0
        {
            return Err(std::io::Error::last_os_error());
        }
        let attributes = SECURITY_ATTRIBUTES {
            nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
            lpSecurityDescriptor: pointer,
            bInheritHandle: 0,
        };
        let name = wide(path);
        // SAFETY: the name is NUL-terminated, and the attributes and the
        // descriptor they point to outlive the call.
        let handle = unsafe {
            fs::CreateFileW(
                name.as_ptr(),
                GENERIC_WRITE,
                fs::FILE_SHARE_READ | fs::FILE_SHARE_WRITE | fs::FILE_SHARE_DELETE,
                &attributes,
                fs::CREATE_NEW,
                fs::FILE_ATTRIBUTE_NORMAL,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error());
        }
        // SAFETY: the handle is new, and the file owns it from here.
        Ok(unsafe { File::from_raw_handle(handle) })
    }

    /// The new file has the old one's owner, group and DACL from
    /// [`create`], and `ReplaceFileW` gives it the old one's whole security
    /// descriptor and attributes as it takes its place.
    pub(super) fn give_properties(_original: &File, _new: &File) -> Result<(), RewriteError> {
        Ok(())
    }

    /// Puts `temporary` in `target`'s place with `ReplaceFileW`, which keeps
    /// `target` under a backup name until the swap is done.
    pub(super) fn take_place(temporary: &Path, target: &Path) -> std::io::Result<()> {
        let (_, backup) = crate::utils::temporary_beside(target)?;
        let (replaced, replacement, aside) = (wide(target), wide(temporary), wide(&backup));
        // SAFETY: the three names are NUL-terminated, and the last two
        // arguments are reserved and null.
        let swapped = unsafe {
            fs::ReplaceFileW(
                replaced.as_ptr(),
                replacement.as_ptr(),
                aside.as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            )
        };
        if swapped != 0 {
            let _ = std::fs::remove_file(&backup);
            return Ok(());
        }
        Err(super::put_back(
            target,
            &backup,
            std::io::Error::last_os_error(),
        ))
    }
}

/// The error of a swap that failed with `error`. A failed `ReplaceFileW` can
/// leave the replaced file under its backup's name, so it is moved back, and
/// when that fails too the error says where the file is.
#[cfg(any(windows, test))]
fn put_back(target: &Path, backup: &Path, error: std::io::Error) -> std::io::Error {
    if target.exists() || !backup.exists() {
        return error;
    }
    match std::fs::rename(backup, target) {
        Ok(()) => error,
        Err(moving) => std::io::Error::new(
            error.kind(),
            format!(
                "{error}. The file is now {}, since moving it back failed ({moving}).",
                backup.display()
            ),
        ),
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
                let new = system::create(&folder.0.join("new"), &path).unwrap();
                system::give_properties(&original, &new).unwrap();
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
        // The new file has the file's rules before any content is written.
        let new_path = folder.0.join("new");
        let new = system::create(&new_path, &path).unwrap();
        assert_eq!(std::fs::metadata(&new_path).unwrap().len(), 0);
        assert_eq!(access_rules(&new_path), access_rules(&path));
        drop(new);
        std::fs::remove_file(&new_path).unwrap();
        rewrite_file(&path, b"new").unwrap();
        assert_eq!(rules(), before);
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert_eq!(folder.names(), ["file.txt"]);
    }

    /// The rules of a file's DACL, each as its bytes, whether inherited or
    /// its own, as a set.
    #[cfg(windows)]
    fn access_rules(path: &Path) -> Vec<Vec<u8>> {
        use windows_sys::Win32::Security::{
            ACE_HEADER, ACL, DACL_SECURITY_INFORMATION, GetAce, GetSecurityDescriptorDacl,
            INHERITED_ACE,
        };
        let mut descriptor = system::security_of(path, DACL_SECURITY_INFORMATION).unwrap();
        let (mut present, mut dacl, mut defaulted) = (0, std::ptr::null_mut::<ACL>(), 0);
        let mut rules = Vec::new();
        // SAFETY: the DACL and its rules are inside the descriptor.
        unsafe {
            assert_ne!(
                GetSecurityDescriptorDacl(
                    descriptor.as_mut_ptr().cast(),
                    &mut present,
                    &mut dacl,
                    &mut defaulted
                ),
                0
            );
            assert!(present != 0 && !dacl.is_null());
            for index in 0..u32::from((*dacl).AceCount) {
                let mut ace = std::ptr::null_mut();
                assert_ne!(GetAce(dacl, index, &mut ace), 0);
                let size = (*ace.cast::<ACE_HEADER>()).AceSize;
                let mut rule = std::slice::from_raw_parts(ace.cast::<u8>(), size.into()).to_vec();
                rule[1] &= !(INHERITED_ACE as u8);
                rules.push(rule);
            }
        }
        // (A rule both inherited and the file's own grants once.)
        rules.sort();
        rules.dedup();
        rules
    }

    /// A file with the setuid or setgid bit is refused and left as it was,
    /// since writing can clear them.
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn the_setuid_and_setgid_bits_are_refused() {
        use std::os::unix::fs::PermissionsExt;
        let folder = Folder::new();
        let path = folder.0.join("run.sh");
        for mode in [0o4755, 0o2755] {
            std::fs::write(&path, b"old").unwrap();
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
            if std::fs::metadata(&path).unwrap().permissions().mode() & 0o6000 == 0 {
                eprintln!("skipped {mode:o}, since the file system took the bit off");
                continue;
            }
            let refused = rewrite_file(&path, b"new");
            assert!(matches!(refused, Err(RewriteError::Kept(_))), "{refused:?}");
            assert_eq!(std::fs::read(&path).unwrap(), b"old");
        }
        assert_eq!(folder.names(), ["run.sh"]);
    }

    /// Linux's inode flags noatime and nodump stay.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn linux_inode_flags_stay() {
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        let set = std::process::Command::new("chattr")
            .args(["+A", "+d"])
            .arg(&path)
            .status();
        if !matches!(set, Ok(status) if status.success()) {
            eprintln!("skipped, since chattr isn't here or the file system has no such flags");
            return;
        }
        let shown = || {
            let listed = std::process::Command::new("lsattr")
                .arg(&path)
                .output()
                .unwrap();
            String::from_utf8(listed.stdout)
                .unwrap()
                .split_whitespace()
                .next()
                .unwrap_or_default()
                .to_string()
        };
        let before = shown();
        rewrite_file(&path, b"new").unwrap();
        assert_eq!(shown(), before);
        assert!(before.contains('A') && before.contains('d'), "{before}");
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
    }

    /// On macOS the hidden flag stays, and an immutable file is refused and
    /// left as it was.
    #[cfg(target_vendor = "apple")]
    #[test]
    fn macos_flags_stay_or_refuse() {
        use std::os::macos::fs::MetadataExt;
        let folder = Folder::new();
        let path = folder.0.join("file.txt");
        std::fs::write(&path, b"old").unwrap();
        let flag = |flag: &str| {
            let set = std::process::Command::new("chflags")
                .arg(flag)
                .arg(&path)
                .status()
                .unwrap();
            assert!(set.success(), "{flag}");
        };
        flag("hidden");
        rewrite_file(&path, b"new").unwrap();
        let flags = std::fs::metadata(&path).unwrap().st_flags();
        assert_ne!(flags & 0x8000, 0, "{flags:#x}");
        flag("uchg");
        let refused = rewrite_file(&path, b"newer");
        flag("nouchg");
        assert!(refused.is_err(), "{refused:?}");
        assert_eq!(std::fs::read(&path).unwrap(), b"new");
        assert_eq!(folder.names(), ["file.txt"]);
    }

    /// On Windows the hidden attribute stays, and a file with a named data
    /// stream, a compressed one and an encrypted one are refused and left
    /// as they were.
    #[cfg(windows)]
    #[test]
    fn windows_attributes_stay_or_refuse() {
        use std::os::windows::fs::MetadataExt;
        let folder = Folder::new();
        let run = |program: &str, arguments: &[&str], path: &Path| {
            std::process::Command::new(program)
                .args(arguments)
                .arg(path)
                .output()
                .is_ok_and(|output| output.status.success())
        };
        let hidden = folder.0.join("hidden.txt");
        std::fs::write(&hidden, b"old").unwrap();
        assert!(run("attrib", &["+h"], &hidden));
        rewrite_file(&hidden, b"new").unwrap();
        let attributes = std::fs::metadata(&hidden).unwrap().file_attributes();
        assert_ne!(attributes & 0x2, 0, "{attributes:#x}");
        assert_eq!(std::fs::read(&hidden).unwrap(), b"new");

        let streams = folder.0.join("streams.txt");
        std::fs::write(&streams, b"old").unwrap();
        std::fs::write(folder.0.join("streams.txt:extra"), b"more").unwrap();
        let refused = rewrite_file(&streams, b"new");
        assert!(matches!(refused, Err(RewriteError::Kept(_))), "{refused:?}");
        assert_eq!(std::fs::read(&streams).unwrap(), b"old");

        let compressed = folder.0.join("compressed.txt");
        std::fs::write(&compressed, b"old").unwrap();
        if run("compact", &["/c"], &compressed) {
            let refused = rewrite_file(&compressed, b"new");
            assert!(matches!(refused, Err(RewriteError::Kept(_))), "{refused:?}");
            assert_eq!(std::fs::read(&compressed).unwrap(), b"old");
        } else {
            eprintln!("skipped compression, since compact failed");
        }

        let encrypted = folder.0.join("encrypted.txt");
        std::fs::write(&encrypted, b"old").unwrap();
        if run("cipher", &["/e"], &encrypted)
            && std::fs::metadata(&encrypted).unwrap().file_attributes() & 0x4000 != 0
        {
            let refused = rewrite_file(&encrypted, b"new");
            assert!(matches!(refused, Err(RewriteError::Kept(_))), "{refused:?}");
            assert_eq!(std::fs::read(&encrypted).unwrap(), b"old");
        } else {
            eprintln!("skipped encryption, since cipher failed");
        }
        assert_eq!(
            folder.names(),
            [
                "compressed.txt",
                "encrypted.txt",
                "hidden.txt",
                "streams.txt"
            ]
        );
    }

    /// A swap that failed with the file moved to its backup puts it back,
    /// and when it can't, the error says where the file is.
    #[test]
    fn a_failed_swap_puts_the_file_back_or_says_where_it_is() {
        let folder = Folder::new();
        let (target, backup) = (folder.0.join("file.txt"), folder.0.join("backup"));
        std::fs::write(&backup, b"old").unwrap();
        let failed = || std::io::Error::other("the swap failed");
        let error = put_back(&target, &backup, failed());
        assert_eq!(error.to_string(), "the swap failed");
        assert_eq!(std::fs::read(&target).unwrap(), b"old");
        assert_eq!(folder.names(), ["file.txt"]);
        // The backup can't be moved back, since the file's folder is gone.
        let target = folder.0.join("gone").join("file.txt");
        std::fs::rename(folder.0.join("file.txt"), &backup).unwrap();
        let error = put_back(&target, &backup, failed()).to_string();
        assert!(error.contains(&backup.display().to_string()), "{error}");
        assert_eq!(std::fs::read(&backup).unwrap(), b"old");
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
