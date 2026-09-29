use super::*;

use Verdict::{Allowed, Outside, Protected};

#[derive(Debug, Clone, Copy, PartialEq)]
enum Verdict {
    Allowed,
    Outside,
    Protected,
}

fn host(
    style: PathStyle,
    workspace: &str,
    home: &str,
    temp: &[&str],
    env: &[(&str, &str)],
) -> Host {
    let path = |text: &str| AbsPath::parse(style, text, None).unwrap();
    Host::new(
        style,
        Some(path(workspace)),
        Some(path(home)),
        temp.iter().map(|text| path(text)).collect(),
        Vec::new(),
        env.iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect(),
    )
}

fn linux() -> Host {
    host(
        PathStyle {
            windows: false,
            ignore_case: false,
        },
        "/home/luke/proj",
        "/home/luke",
        &["/tmp", "/var/tmp"],
        &[
            ("HOME", "/home/luke"),
            ("BUILD", "/home/luke/proj/build"),
            ("TMPDIR", "/tmp"),
        ],
    )
}

fn mac() -> Host {
    host(
        PathStyle {
            windows: false,
            ignore_case: true,
        },
        "/Users/luke/proj",
        "/Users/luke",
        &["/private/var/folders/ab/xyz/T", "/tmp", "/private/tmp"],
        &[
            ("HOME", "/Users/luke"),
            ("TMPDIR", "/private/var/folders/ab/xyz/T"),
        ],
    )
}

fn windows() -> Host {
    host(
        PathStyle {
            windows: true,
            ignore_case: true,
        },
        r"C:\Users\luke\proj",
        r"C:\Users\luke",
        &[r"C:\Users\luke\AppData\Local\Temp"],
        &[
            ("USERPROFILE", r"C:\Users\luke"),
            ("TEMP", r"C:\Users\luke\AppData\Local\Temp"),
            ("SystemRoot", r"C:\Windows"),
        ],
    )
}

fn verdict(guard: &CommandGuard, host: &Host, dialect: ShellDialect, command: &str) -> Verdict {
    match guard.check_on(host, dialect, command) {
        Ok(()) => Allowed,
        Err(refusal) if refusal.protected => Protected,
        Err(_) => Outside,
    }
}

#[track_caller]
fn expect(host: &Host, dialect: ShellDialect, cases: &[(Verdict, &str)]) {
    let guard = CommandGuard::default();
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|(expected, command)| {
            let actual = verdict(&guard, host, dialect, command);
            let detail = guard
                .check_on(host, dialect, command)
                .err()
                .map(|refusal| refusal.message)
                .unwrap_or_default();
            (actual != *expected)
                .then(|| format!("{command:?}: expected {expected:?}, got {actual:?} {detail}"))
        })
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

#[test]
fn posix_everyday_commands_run() {
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "rm -rf target"),
            (Allowed, "rm -rf ./build/* dist node_modules"),
            (Allowed, "cargo clean && rm -rf target"),
            (Allowed, "rm -f /tmp/build.log"),
            (Allowed, "rm -rf /tmp/arko-test-123"),
            (Allowed, "find . -name '*.o' -delete"),
            (Allowed, "find . -type f -name '*.pyc' -exec rm -f {} +"),
            (Allowed, "git clean -fdx"),
            (Allowed, "git status && git diff"),
            (Allowed, "echo done > /dev/null 2>&1"),
            (Allowed, "cargo test 2>/dev/null | tee test.log"),
            (Allowed, "ls -la / && cat /etc/hosts"),
            (Allowed, "grep -rn 'rm -rf /' ."),
            (Allowed, "echo 'rm -rf ~'"),
            (Allowed, "rm -rf \"$HOME/proj/build\""),
            (Allowed, "rm -rf ~/proj/tmp"),
            (Allowed, "X=build; rm -rf \"$X\""),
            (Allowed, "export OUT=dist && rm -rf $OUT/*"),
            (Allowed, "rm -rf $BUILD/out"),
            (Allowed, "rm -rf ${BUILD:?}/out"),
            (Allowed, "rm -rf \"${MISSING:?not set}\"/*"),
            (Allowed, "cd build && rm -rf *"),
            (Allowed, "mkdir -p out && cd out && rm -rf ./*"),
            (Allowed, "mkfs.ext4 disk.img"),
            (Allowed, "dd if=/dev/zero of=disk.img bs=1M count=10"),
            (Allowed, "chmod -R 755 scripts"),
            (Allowed, "chmod +x ~/bin/tool"),
            (Allowed, "echo 'export PATH=$PATH:~/bin' >> ~/.bashrc"),
            (Allowed, "mv old.txt new.txt"),
            (Allowed, "rsync -a --delete src/ backup/"),
            (
                Allowed,
                "cat <<'EOF' > notes.md\nrm -rf /\nrm -rf ~\nEOF\necho ok",
            ),
            (Allowed, "fdisk -l /dev/sda"),
            (Allowed, "ls build | xargs rm -rf"),
            (Allowed, "find . -name '*.tmp' | xargs rm -f"),
            (Allowed, "rm -rf \"\""),
            (Allowed, "rm -rf $UNSET_VARIABLE"),
            (Allowed, "sudo rm -rf build"),
            (Allowed, "bash -c 'rm -rf target'"),
            (Allowed, "rm -rf \"$(pwd)/build\""),
            (Allowed, "rm -rf `pwd`/build"),
            (
                Allowed,
                "TMP=$(mktemp -d); cp -r src \"$TMP\"; rm -rf \"$TMP\"",
            ),
            (Allowed, "cd \"$(mktemp -d)\" && rm -rf *"),
        ],
    );
}

#[test]
fn posix_drastic_commands_are_refused() {
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Protected, "rm -rf /"),
            (Protected, "rm -rf /*"),
            (Protected, "sudo rm -rf --no-preserve-root /"),
            (Protected, "rm -rf ~"),
            (Protected, "rm -rf ~/"),
            (Protected, "rm -rf ~/*"),
            (Protected, "rm -rf $HOME"),
            (Protected, "rm -rf \"$HOME\"/"),
            (Protected, "rm -rf /home/luke"),
            (Protected, "rm -rf /home"),
            (Protected, "rm -rf /home/*"),
            (Protected, "rm -rf ~bob"),
            (Protected, "rm -rf /home/bob"),
            // An unset variable empties the path down to the root.
            (Protected, "rm -rf \"$OUT\"/*"),
            (Protected, "rm -rf $OUT/"),
            (Protected, "rm -rf ${OUT}/"),
            (Protected, "rm -rf \"$STEAMROOT/\"*"),
            (Protected, "rm -rf $PREFIX/usr"),
            // A stray space.
            (Protected, "rm -rf / tmp/build"),
            // The wrong working directory.
            (Protected, "cd; rm -rf *"),
            (Protected, "cd $UNSET && rm -rf *"),
            (Protected, "cd ~ && rm -rf .* *"),
            (Protected, "cd / && rm -rf *"),
            (Protected, "cd .. && rm -rf *"),
            (Protected, "rm -rf .."),
            (Protected, "rm -rf ../../.."),
            (Protected, "git -C ~ clean -fdx"),
            (Protected, "cd ~ && git clean -ffdx"),
            // System and device paths.
            (Protected, "rm -rf /usr/lib"),
            (Protected, "rm -rf /etc"),
            (Protected, "rm -rf /opt"),
            (Protected, "find / -name '*.pyc' -delete"),
            (Protected, "find ~ -type f -exec rm {} \\;"),
            (Protected, "dd if=/dev/zero of=/dev/sda bs=1M"),
            (Protected, "echo x > /dev/sda"),
            (Protected, "mkfs.ext4 /dev/sdb1"),
            (Protected, "wipefs -a /dev/nvme0n1"),
            (Protected, "rm /dev/null"),
            (Protected, "chmod -R 777 /"),
            (Protected, "chmod 000 /etc/passwd"),
            (Protected, "chown -R luke /usr"),
            (Protected, "truncate -s 0 /etc/passwd"),
            (Protected, "mv ~ /tmp/old-home"),
            (Protected, "mv /usr /usr.bak"),
            (Protected, "rsync -a --delete empty/ ~/"),
            (Protected, "rm -rf /mnt/backup"),
            (Protected, "rm -rf /media/usb"),
            // Pipelines and nested shells.
            (Protected, "ls / | xargs rm -rf"),
            (Protected, "find ~ -type f | xargs rm -f"),
            (Protected, "bash -c \"rm -rf ~\""),
            (Protected, "sudo sh -c 'cd / && rm -rf *'"),
            (Protected, "eval \"rm -rf /\""),
            (Protected, "env FOO=1 nice -n 5 rm -rf /"),
            (Protected, "timeout 10 rm -rf ~"),
            (Protected, "if true; then rm -rf ~; fi"),
            (Protected, "rm -rf \"$(dirname \"$0\")\"/.."),
            (Protected, "pwsh -Command \"Remove-Item / -Recurse\""),
        ],
    );
}

#[test]
fn posix_paths_outside_the_working_directory_need_permission() {
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Outside, "rm -rf ../other-project"),
            (Outside, "rm -rf ~/Documents"),
            (Outside, "rm -rf ~/.cache/pip"),
            (Outside, "rm -rf /opt/app/cache"),
            (Outside, "mv ~/.ssh/config ."),
            (Outside, "echo x > ~/.bashrc"),
            (Outside, "rm -rf \"$(git rev-parse --show-toplevel)/build\""),
            (Outside, "source .env && rm -rf \"$OUT_DIR\""),
            (Outside, "source .env; rm -rf \"${OUT:?}/build\""),
            (
                Outside,
                "cd \"$(mktemp -d build.XXXX)\" && rm -rf ../../src",
            ),
            (
                Outside,
                "rm -rf \"$(git rev-parse --show-toplevel)/../other\"",
            ),
        ],
    );
}

#[test]
fn macos_paths_compare_without_case() {
    expect(
        &mac(),
        ShellDialect::Posix,
        &[
            (Protected, "rm -rf /USERS/LUKE"),
            (Protected, "rm -rf /Users/bob"),
            (Protected, "rm -rf /Volumes/Backup"),
            (Protected, "rm -rf /System/Library"),
            (Protected, "rm -rf /Applications"),
            (Protected, "diskutil eraseDisk APFS Blank disk2"),
            (Protected, "diskutil apfs deleteContainer disk3"),
            (Allowed, "diskutil list"),
            (Allowed, "rm -rf \"$TMPDIR\"/arko-*"),
            (Allowed, "rm -rf /private/var/folders/ab/xyz/T/scratch"),
            (Allowed, "rm -rf /Users/luke/PROJ/build"),
            (Outside, "rm -rf /Applications/Old.app"),
        ],
    );
}

#[test]
fn cmd_commands() {
    expect(
        &windows(),
        ShellDialect::Cmd,
        &[
            (Allowed, "rd /s /q build"),
            (Allowed, "del /q *.obj && rmdir /s /q out"),
            (Allowed, "del /f /q %TEMP%\\build.log"),
            (Allowed, "echo done > nul 2>&1"),
            (Allowed, "dir C:\\ /s"),
            (Allowed, "rd /s /q %UNDEFINED%"),
            (Allowed, "set OUT=dist& rd /s /q %OUT%"),
            (Allowed, "if exist build rd /s /q build"),
            (Protected, "rd /s /q C:\\"),
            (Protected, "rmdir /S /Q c:\\users\\luke"),
            (Protected, "del /f /s /q C:\\*"),
            (Protected, "del /f /s /q C:\\*.*"),
            (Protected, "rd /s /q %USERPROFILE%"),
            (Protected, "rd /s /q \"%USERPROFILE%\\\""),
            (Protected, "rd /s /q .."),
            (Protected, "cd \\ && del /s /q *"),
            (Protected, "cd /d C:\\ & rd /s /q ."),
            (Protected, "rd /s /q \"C:\\Program Files\\App\""),
            (Protected, "rd /s /q C:\\Windows\\System32"),
            (Protected, "rd /s /q D:\\"),
            (Protected, "rd /s /q \\\\server\\share"),
            (Protected, "format D: /q /y"),
            (Protected, "diskpart /s wipe.txt"),
            (Protected, "robocopy empty C:\\Users\\luke /MIR"),
            (Protected, "icacls C:\\ /grant Everyone:F /t"),
            (Protected, "takeown /f C:\\Windows /r"),
            (Protected, "@echo off & rd /s /q C:\\Users"),
            (Protected, "cmd /c rd /s /q C:\\"),
            (
                Protected,
                "powershell -Command \"Remove-Item C:\\ -Recurse -Force\"",
            ),
            (Outside, "rd /s /q C:\\Users\\luke\\Documents"),
            (Outside, "del D:\\data\\old\\file.txt"),
            (Outside, "for /d %d in (C:\\Users\\luke\\*) do rd /s /q %d"),
        ],
    );
}

#[test]
fn powershell_commands() {
    let encoded = {
        let units: Vec<u8> = "Remove-Item C:\\ -Recurse"
            .encode_utf16()
            .flat_map(u16::to_le_bytes)
            .collect();
        base64(&units)
    };
    let encoded_command = format!("powershell -EncodedCommand {encoded}");
    expect(
        &windows(),
        ShellDialect::PowerShell,
        &[
            (Allowed, "Remove-Item -Recurse -Force .\\build"),
            (Allowed, "Remove-Item build,dist -Recurse"),
            (Allowed, "rm -r -fo node_modules"),
            (
                Allowed,
                "Get-ChildItem -Recurse -Filter *.tmp | Remove-Item",
            ),
            (
                Allowed,
                "Get-ChildItem build | ForEach-Object { Remove-Item $_ -Recurse }",
            ),
            (Allowed, "Remove-Item \"$env:TEMP\\x.log\""),
            (Allowed, "'hi' > $null"),
            (Allowed, "Get-ChildItem C:\\ -Recurse | Measure-Object"),
            (Allowed, "$d = 'build'; Remove-Item $d -Recurse"),
            (Allowed, "Remove-Item $undefined -Recurse"),
            (
                Allowed,
                "if (Test-Path build) { Remove-Item build -Recurse -Force }",
            ),
            (Allowed, "Set-Content -Path notes.txt -Value 'x'"),
            (Allowed, "# Remove-Item C:\\ -Recurse\nWrite-Output ok"),
            (Protected, "Remove-Item -Recurse -Force C:\\"),
            (Protected, "Remove-Item -Path 'C:\\' -Recurse"),
            (Protected, "Remove-Item \"$x\\*\" -Recurse -Force"),
            (Protected, "ri $HOME -r -fo"),
            (Protected, "Remove-Item ~ -Recurse"),
            (Protected, "Remove-Item $env:USERPROFILE -Recurse"),
            (
                Protected,
                "Remove-Item \"$env:USERPROFILE\\*\" -Recurse -Force",
            ),
            (
                Protected,
                "Get-ChildItem C:\\ | Remove-Item -Recurse -Force",
            ),
            (Protected, "gci ~ | % { Remove-Item $_ -Recurse -Force }"),
            (
                Protected,
                "[System.IO.Directory]::Delete('C:\\Users\\luke', $true)",
            ),
            (Protected, "Format-Volume -DriveLetter D"),
            (Protected, "Clear-Disk -Number 1 -RemoveData"),
            (Protected, "Set-Location \\; Remove-Item * -Recurse"),
            (Protected, "cd; Remove-Item * -Recurse -Force"),
            (Protected, "Set-Location ..; Remove-Item * -Recurse"),
            (Protected, "Remove-Item 'C:\\Program Files\\App' -Recurse"),
            (Protected, "Move-Item C:\\Users\\luke D:\\old"),
            (Protected, "cmd /c rd /s /q C:\\"),
            (Protected, encoded_command.as_str()),
            (Outside, "Remove-Item ~\\Documents -Recurse"),
            (Outside, "Move-Item ~\\notes.txt ."),
            (
                Outside,
                "foreach ($dir in $dirs) { Remove-Item $dir -Recurse }",
            ),
        ],
    );
}

#[test]
fn settings_allow_outside_paths_but_never_protected_ones() {
    let host = linux();
    let outside = CommandGuard {
        allow_outside_workspace: true,
        ..CommandGuard::default()
    };
    assert_eq!(
        verdict(&outside, &host, ShellDialect::Posix, "rm -rf ../other"),
        Allowed
    );
    assert_eq!(
        verdict(&outside, &host, ShellDialect::Posix, "rm -rf ~"),
        Protected
    );

    let mut host = linux();
    host.allowed = vec![AbsPath::parse(host.style, "/home/luke/.cache/app", None).unwrap()];
    let guard = CommandGuard::default();
    assert_eq!(
        verdict(&guard, &host, ShellDialect::Posix, "rm -rf ~/.cache/app/*"),
        Allowed
    );
    assert_eq!(
        verdict(&guard, &host, ShellDialect::Posix, "rm -rf ~/.cache"),
        Outside
    );

    let everything = CommandGuard {
        dangerously_allow_all_commands: true,
        ..CommandGuard::default()
    };
    assert_eq!(
        verdict(&everything, &host, ShellDialect::Posix, "rm -rf /"),
        Allowed
    );
}

#[test]
fn a_home_working_directory_allows_nothing_by_being_the_working_directory() {
    let style = PathStyle {
        windows: false,
        ignore_case: false,
    };
    let home = host(
        style,
        "/home/luke",
        "/home/luke",
        &["/tmp"],
        &[("HOME", "/home/luke")],
    );
    expect(
        &home,
        ShellDialect::Posix,
        &[
            (Protected, "rm -rf *"),
            (Outside, "rm -rf Documents"),
            (Allowed, "ls -la"),
        ],
    );
}

#[test]
fn refusals_explain_the_path_and_the_variable() {
    let refusal = CommandGuard::default()
        .check_on(&linux(), ShellDialect::Posix, "rm -rf \"$OUT\"/*")
        .unwrap_err();
    assert!(refusal.protected);
    assert_eq!(refusal.target, "/*");
    assert!(
        refusal.message.contains("`rm` would delete `/*`"),
        "{}",
        refusal.message
    );
    assert!(
        refusal.message.contains("the filesystem root"),
        "{}",
        refusal.message
    );
    assert!(
        refusal.message.contains("`$OUT` is empty or not set here"),
        "{}",
        refusal.message
    );

    let refusal = CommandGuard::default()
        .check_on(&linux(), ShellDialect::Posix, "rm -rf ~/Documents")
        .unwrap_err();
    assert!(!refusal.protected);
    assert!(
        refusal
            .message
            .contains("outside the working directory `/home/luke/proj`"),
        "{}",
        refusal.message
    );
}

#[test]
fn dialects_come_from_the_shell_program() {
    assert_eq!(
        ShellDialect::from_shell(Path::new("/bin/sh")),
        ShellDialect::Posix
    );
    assert_eq!(
        ShellDialect::from_shell(Path::new("bash")),
        ShellDialect::Posix
    );
    assert_eq!(
        ShellDialect::from_shell(Path::new("cmd.exe")),
        ShellDialect::Cmd
    );
    assert_eq!(
        ShellDialect::from_shell(Path::new(
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe"
        )),
        ShellDialect::PowerShell
    );
    assert_eq!(
        ShellDialect::from_shell(Path::new("/usr/bin/pwsh")),
        ShellDialect::PowerShell
    );
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut text = String::new();
    for chunk in bytes.chunks(3) {
        let value = chunk.iter().enumerate().fold(0u32, |value, (index, byte)| {
            value | (u32::from(*byte) << (16 - 8 * index))
        });
        for index in 0..4 {
            if index <= chunk.len() {
                text.push(ALPHABET[(value >> (18 - 6 * index)) as usize & 63] as char);
            } else {
                text.push('=');
            }
        }
    }
    text
}
