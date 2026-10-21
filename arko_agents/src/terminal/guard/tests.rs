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

// ------------------------------------------------------------------ rules

/// Checks each command under `rules` on `host`: `true` when it may run.
#[track_caller]
fn expect_ruled(rules: &[CommandRule], host: &Host, dialect: ShellDialect, cases: &[(bool, &str)]) {
    let guard = CommandGuard {
        rules: rules.to_vec(),
        ..CommandGuard::default()
    };
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|(runs, command)| {
            let result = guard.check_on(host, dialect, command);
            (result.is_ok() != *runs).then(|| {
                format!(
                    "{command:?}: expected {}, got {}",
                    if *runs { "to run" } else { "a refusal" },
                    result
                        .err()
                        .map_or("none".into(), |refusal| refusal.message)
                )
            })
        })
        .collect();
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

fn read_only_git() -> Vec<CommandRule> {
    vec![
        CommandRule::allow("git status *"),
        CommandRule::allow("git log *"),
        CommandRule::allow("git diff *"),
        CommandRule::deny("git * --output* *"),
    ]
}

#[test]
fn an_allow_list_allows_only_its_commands_and_a_deny_wins() {
    expect_ruled(
        &read_only_git(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "git status"),
            (true, "git status --short"),
            (true, "git log --oneline -5 -- src"),
            (true, "/usr/bin/git diff HEAD~1"),
            (true, "\"git\" log"),
            (true, "git -C repo log"),
            (true, "git --no-pager log -p"),
            (true, "git -P diff"),
            (true, "git --no-optional-locks status"),
            (true, "git log $BUILD"),
            (true, "cargo build && ls -la"),
            (false, "git"),
            (false, "git push origin main"),
            (false, "git commit -am wip"),
            (false, "git log --output=log.txt"),
            (false, "git diff --output log.txt"),
            (false, "git -c core.pager=less log"),
            (false, "git --git-dir=other/.git log"),
            (false, "git --work-tree other status"),
            // An alias isn't one of the allowed commands.
            (false, "git lg"),
            (false, "git status && git push"),
        ],
    );
}

#[test]
fn rules_match_commands_not_mentions_of_a_program() {
    expect_ruled(
        &read_only_git(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "ls /usr/share/doc/git"),
            (true, "grep -rn git ."),
            (true, "echo git push"),
            (true, "cat <<EOF\ngit push\nEOF"),
            (
                true,
                "cat <<'EOF' > notes.md\ngit commit -am x\nEOF\necho ok",
            ),
            (true, "echo \"git push\" > notes.txt"),
            (true, "# git push\nls"),
            (true, "rg 'git push' src"),
            (true, "which git && command -v git"),
            (true, "cat .gitignore"),
            (true, "for f in $(git diff --name-only); do echo $f; done"),
        ],
    );
}

#[test]
fn rules_follow_commands_through_the_shell() {
    expect_ruled(
        &read_only_git(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "bash -c 'git status'"),
            (true, "ls | xargs git log"),
            (true, "sudo git status"),
            (false, "bash -c 'git push'"),
            (false, "sh -c \"cd sub && git commit -m x\""),
            (false, "eval 'git push'"),
            (false, "env X=1 git push"),
            (false, "FOO=1 git push"),
            (false, "sudo -u luke git push"),
            (false, "timeout 10 nice -n 5 git push"),
            (false, "ls | xargs git add"),
            (false, "find . -name '*.rs' | xargs -n 1 git add"),
            (false, "find . -name '*.rs' -exec git add {} +"),
            (
                false,
                "find . -name '*.rs' -execdir sh -c 'git add \"$1\"' _ {} \\;",
            ),
            (false, "cd sub && git commit -m x"),
            (false, "true && git push || echo failed"),
            (false, "if git diff --quiet; then git push; fi"),
            (false, "(git push)"),
            (false, "echo \"pushed: $(git push)\""),
            (false, "echo `git push`"),
            (false, "OUT=$(git push 2>&1)"),
            (false, "G=git; $G push"),
            (false, "ARGS='push origin'; git $ARGS"),
            (false, "nohup git push &"),
        ],
    );
}

#[test]
fn a_deny_pattern_spans_words() {
    expect_ruled(
        &[CommandRule::deny("cargo * publish *")],
        &linux(),
        ShellDialect::Posix,
        &[
            (false, "cargo publish"),
            (false, "cargo +nightly publish --dry-run"),
            (false, "cargo publish -p arko_ai"),
            (true, "cargo test publish_tests"),
            (true, "cargo build --release"),
            (true, "git push"),
        ],
    );
}

#[test]
fn an_allow_list_catches_arguments_a_deny_rule_cant_read() {
    // As the docs say: a deny rule matches the words the guard reads; an
    // argument only known when the command runs matches only `*`.
    let hidden = ["cargo $(printf publish)", "sh -c 'cargo \"$1\"' _ publish"];
    for command in hidden {
        expect_ruled(
            &[CommandRule::deny("cargo publish *")],
            &linux(),
            ShellDialect::Posix,
            &[(true, command)],
        );
        expect_ruled(
            &[
                CommandRule::allow("cargo build *"),
                CommandRule::allow("cargo test *"),
            ],
            &linux(),
            ShellDialect::Posix,
            &[(false, command), (true, "cargo test -p arko_ai")],
        );
    }
}

#[test]
fn a_rule_within_a_directory_needs_the_command_there() {
    let rules = [
        CommandRule::allow("git status *"),
        CommandRule::allow("git *").within("scratch"),
    ];
    expect_ruled(
        &rules,
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "git status"),
            (true, "cd scratch && git init && git commit -m x"),
            (true, "cd scratch/spike-1 && git reset --hard HEAD~1"),
            (true, "git -C scratch rebase -i HEAD~2"),
            (true, "git -C /home/luke/proj/scratch/spike-2 commit -m x"),
            (true, "cd /tmp && git -C /home/luke/proj/scratch log"),
            (false, "git commit -m x"),
            (false, "cd scratch && cd .. && git commit -m x"),
            (false, "cd scratch && git -C .. commit -m x"),
            (false, "git -C other commit -m x"),
            (false, "cd scratch-other && git commit -m x"),
            // The directory isn't known: never within.
            (
                false,
                "cd \"$(git rev-parse --show-toplevel)\" && git commit -m x",
            ),
            (false, "git -C \"$(cat repo.txt)\" commit -m x"),
            (false, "cd scratch && git --git-dir=../.git commit -m x"),
            (false, "cd scratch && GIT_DIR=../.git git commit -m x"),
            (false, "cd scratch && env -C .. git commit -m x"),
            (false, "cd scratch && find . -execdir git add {} +"),
        ],
    );
}

#[test]
fn another_repository_stays_elsewhere_whatever_directory_follows() {
    // git works on the repository GIT_DIR or --git-dir names, wherever -C
    // points.
    let rules = [
        CommandRule::allow("git status *"),
        CommandRule::allow("git *").within("scratch"),
    ];
    expect_ruled(
        &rules,
        &linux(),
        ShellDialect::Posix,
        &[
            (
                true,
                "cd \"$(cat dir.txt)\" && git -C /home/luke/proj/scratch log",
            ),
            (
                false,
                "GIT_DIR=/home/luke/proj/.git git -C /home/luke/proj/scratch push",
            ),
            (
                false,
                "git --git-dir=/home/luke/proj/.git -C /home/luke/proj/scratch commit -m x",
            ),
            (
                false,
                "git --work-tree /home/luke/proj -C /home/luke/proj/scratch checkout .",
            ),
            (
                false,
                "export GIT_DIR=/home/luke/proj/.git; git -C /home/luke/proj/scratch commit -m x",
            ),
            (
                false,
                "env GIT_WORK_TREE=/home/luke/proj git -C /home/luke/proj/scratch checkout .",
            ),
        ],
    );
}

#[test]
fn a_subshell_keeps_its_directory_and_variables() {
    // `( … )` runs in a subshell, and so does each member of a pipeline but
    // the last: a `cd` or an assignment there ends with it.
    let rules = [
        CommandRule::allow("git status *"),
        CommandRule::allow("git *").within("scratch"),
    ];
    expect_ruled(
        &rules,
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "(cd scratch && git commit -m x)"),
            (true, "cd scratch && (cd .. && ls) && git commit -m x"),
            (true, "cd scratch && ((cd ..); git commit -m x)"),
            (true, "cd scratch; f() { ls; }; git commit -m x"),
            (
                false,
                "(cd /home/luke/proj/scratch && cargo test); git commit -m x",
            ),
            (false, "(cd scratch) && git commit -m x"),
            (false, "(cd scratch; (cd ..)); git commit -m x"),
            (false, "cd scratch | cat; git commit -m x"),
            (false, "ls | (cd scratch) | cat && git commit -m x"),
            // The last member runs in the current shell in zsh (and in bash
            // with `lastpipe`): the directory isn't known after it.
            (false, "ls | cd scratch; git commit -m x"),
        ],
    );
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "(cd / && ls) && rm -rf *"),
            (Allowed, "cd / | cat; rm -rf *"),
            (Allowed, "X=build; (X=/); rm -rf \"$X\"/*"),
            (Allowed, "X=build; X=/ | cat; rm -rf \"$X\"/*"),
            (Allowed, "cd build && (cd /; ls) && rm -rf *"),
            (Protected, "(cd / && rm -rf *)"),
            (Protected, "(cd build); cd / && rm -rf *"),
            (Protected, "cd / && (cd build) && rm -rf *"),
            (Outside, "ls | cd /; rm -rf *"),
            // A `case` pattern's `)` closes no subshell.
            (Protected, "(cd /; case $1 in a) ls;; esac; rm -rf *)"),
            (Protected, "case $1 in (a) cd /;; esac; rm -rf *"),
            (Allowed, "case $1 in a) (cd /);; esac; rm -rf *"),
            // Only known when it runs, and possibly empty.
            (Protected, "X=build; ls | X=/; rm -rf \"$X\"/*"),
        ],
    );
}

fn scratch_rules() -> [CommandRule; 2] {
    [
        CommandRule::allow("git status *"),
        CommandRule::allow("git *").within("scratch"),
    ]
}

#[test]
fn a_background_list_runs_in_a_subshell() {
    // `… &` runs the and-or list before it in a subshell.
    expect_ruled(
        &scratch_rules(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "cd scratch && git commit -m x &"),
            (true, "cd scratch; git commit -m x & wait"),
            (
                true,
                "(cd scratch & ls; git status) && cd scratch && git commit -m x",
            ),
            (false, "cd scratch & git commit -m x"),
            (false, "cd scratch && ls & git commit -m x"),
            (false, "cd scratch &\ngit commit -m x"),
            (false, "cd scratch && (ls) & git commit -m x"),
            (false, "(cd scratch & git commit -m x)"),
            (false, "ls | cd scratch & git commit -m x"),
        ],
    );
    // The working directory is $HOME/proj.
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "cd build & rm -rf *"),
            (Allowed, "cd / & rm -rf *"),
            (Allowed, "cd /tmp && ls & rm -rf build"),
            (Protected, "cd ..; cd proj & rm -rf *"),
            (Protected, "cd ..; cd proj && ls & rm -rf *"),
            (Protected, "cd ..; X=proj & rm -rf ${X:-.}/*"),
            (Protected, "cd / && rm -rf * &"),
        ],
    );
}

#[test]
fn a_group_or_compound_command_in_a_pipeline_keeps_its_directory() {
    // A pipeline's member runs in a subshell whatever it is: a subshell, a
    // `{ … }` group or a loop. In the shell itself, a group's `cd` lasts.
    expect_ruled(
        &scratch_rules(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "{ cd scratch; }; git commit -m x"),
            (true, "{ cd scratch; } && git commit -m x"),
            (true, "if cd scratch; then git commit -m x; fi"),
            (true, "for f in a; do cd scratch; done; git commit -m x"),
            (true, "{ cd scratch; git commit -m x; } | cat"),
            (false, "cd scratch | (ls); git commit -m x"),
            (false, "cd scratch | (git commit -m x)"),
            (false, "cd scratch | ( cat ) | cat; git commit -m x"),
            (false, "{ cd scratch; } | cat; git commit -m x"),
            (false, "{ cd scratch; } & git commit -m x"),
            (
                false,
                "for f in a; do cd scratch; done | cat; git commit -m x",
            ),
            (false, "while true; do cd scratch; done & git commit -m x"),
            (false, "if true; then cd scratch; fi | cat; git commit -m x"),
            // As a pipeline's last member, where it runs depends on the shell.
            (false, "ls | { cd scratch; }; git commit -m x"),
            (
                false,
                "ls | while read f; do cd scratch; done; git commit -m x",
            ),
        ],
    );
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "{ cd build; } && rm -rf *"),
            (Allowed, "cd / | (ls); rm -rf *"),
            (Protected, "{ cd /; } && rm -rf *"),
            (Protected, "cd ..; { cd proj; } | cat; rm -rf *"),
            (Protected, "cd ..; cd proj | (ls); rm -rf *"),
            (Protected, "cd ..; cd proj | ( ls ) | cat; rm -rf *"),
            (
                Protected,
                "cd ..; for d in proj; do cd $d; done | cat; rm -rf *",
            ),
            (Protected, "ls | { cd /; rm -rf *; }"),
            (
                Outside,
                "cd /; ls | while read f; do cd ~/proj; done; rm -rf *",
            ),
        ],
    );
}

#[test]
fn sudo_takes_assignments_like_env() {
    expect_ruled(
        &read_only_git(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "sudo X=1 git status"),
            (false, "sudo X=1 git push"),
            (false, "sudo DEBIAN_FRONTEND=noninteractive git push"),
            (false, "sudo -E X=1 git push"),
            (false, "sudo -u luke X=1 git push"),
        ],
    );
    expect_ruled(
        &scratch_rules(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "sudo X=1 git -C /home/luke/proj/scratch commit -m x"),
            (
                false,
                "sudo GIT_DIR=/home/luke/proj/.git git -C /home/luke/proj/scratch commit -m x",
            ),
        ],
    );
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "sudo X=1 rm -rf build"),
            (Protected, "sudo FOO=1 rm -rf /"),
            (Protected, "sudo -u root HOME=/root rm -rf ~"),
        ],
    );
}

#[test]
fn more_wrappers_run_the_command_after_them() {
    expect_ruled(
        &read_only_git(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "setsid -f git status"),
            (true, "flock -w 5 /tmp/lock git status"),
            (true, "flock /tmp/lock -c 'git status'"),
            (true, "taskset -c 0-3 git log"),
            (true, "taskset -p 03 1234"),
            (true, "chronic -e git status"),
            (true, "unshare --propagation private -m git status"),
            (true, "strace -f -o trace.txt -e trace=file git status"),
            (true, "ltrace -o calls.txt git log"),
            (true, "watch -n 5 git status"),
            (true, "watch -n 5 'git status && git log -1'"),
            (true, "watch -x git status"),
            (true, "script -q -c 'git log' /dev/null"),
            (true, "su luke -c 'git status'"),
            (false, "setsid git push"),
            (false, "setsid -w git push"),
            (false, "flock /tmp/lock git push"),
            (false, "flock -n -w 5 /tmp/lock git push"),
            (false, "flock /tmp/lock -c 'git push'"),
            (false, "flock /tmp/lock --command 'git push'"),
            (false, "taskset 0x3 git push"),
            (false, "taskset -c 0-3 git push"),
            (false, "chronic git push"),
            (false, "unshare -r git push"),
            (false, "unshare --map-user 0 --propagation private git push"),
            (false, "strace -f -o trace.txt git push"),
            (false, "strace -e trace=file -s 200 git push"),
            (false, "ltrace -l libgit2.so git push"),
            (false, "watch git push"),
            (false, "watch -n 5 'git push'"),
            (false, "watch -d -n 5 git status \\&\\& git push"),
            (false, "watch -x git push"),
            (false, "script -c 'git push'"),
            (false, "script -qc 'git push' log.txt"),
            (false, "script --command='git push'"),
            (false, "su -c 'git push'"),
            (false, "su luke -c 'git push'"),
            (false, "su -lc 'git push' luke"),
            (false, "su --session-command 'git push'"),
            // Text only known when it runs can't be checked.
            (false, "watch \"$(cat command.txt)\""),
            (false, "su -c \"$(cat command.txt)\""),
            (false, "flock /tmp/lock -c \"$(cat command.txt)\""),
        ],
    );
    // Without rules, the path checks follow the commands too.
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "watch \"$(cat command.txt)\""),
            (Allowed, "setsid rm -rf build"),
            (Protected, "setsid rm -rf ~"),
            (Protected, "flock /tmp/lock rm -rf /"),
            (Protected, "flock /tmp/lock -c 'rm -rf /'"),
            (Protected, "taskset -c 1 rm -rf /"),
            (Protected, "chronic rm -rf ~"),
            (Protected, "unshare -r rm -rf /etc"),
            (Protected, "strace -o /tmp/trace rm -rf ~"),
            (Protected, "ltrace -o /tmp/calls rm -rf ~"),
            (Protected, "watch 'rm -rf ~'"),
            (Protected, "script -c 'rm -rf ~' /dev/null"),
            (Protected, "su -c 'rm -rf /'"),
        ],
    );
    // `unshare -w` runs the command in another directory.
    expect_ruled(
        &scratch_rules(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "cd scratch && unshare -r git commit -m x"),
            (false, "cd scratch && unshare -w .. git commit -m x"),
            (false, "cd scratch && unshare --root=/ git commit -m x"),
        ],
    );
}

#[test]
fn git_location_variables_mean_another_repository() {
    let variables = [
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_DIR",
        "GIT_WORK_TREE",
    ];
    for variable in variables {
        let value = "/home/luke/proj/.git/x";
        expect_ruled(
            &scratch_rules(),
            &linux(),
            ShellDialect::Posix,
            &[
                (true, "git -C /home/luke/proj/scratch/toy add ."),
                (
                    false,
                    &format!("{variable}={value} git -C /home/luke/proj/scratch/toy add ."),
                ),
                (
                    false,
                    &format!("export {variable}={value}; git -C /home/luke/proj/scratch/toy add ."),
                ),
                (
                    false,
                    &format!("env {variable}={value} git -C /home/luke/proj/scratch/toy add ."),
                ),
                (
                    false,
                    &format!(
                        "strace -E {variable}={value} git -C /home/luke/proj/scratch/toy add ."
                    ),
                ),
            ],
        );
        // Set in the job's environment.
        let host = host(
            PathStyle {
                windows: false,
                ignore_case: false,
            },
            "/home/luke/proj",
            "/home/luke",
            &["/tmp"],
            &[("HOME", "/home/luke"), (variable, value)],
        );
        expect_ruled(
            &scratch_rules(),
            &host,
            ShellDialect::Posix,
            &[(false, "git -C /home/luke/proj/scratch/toy add .")],
        );
    }
}

#[test]
fn a_wrapper_without_its_command_runs_nothing() {
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "timeout"),
            (Allowed, "timeout 5"),
            (Allowed, "sudo -u"),
            (Allowed, "flock -w"),
            (Allowed, "taskset 03"),
        ],
    );
}

#[test]
fn a_listing_reaches_the_next_member_from_a_subshell_or_group() {
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "(find . -name '*.o') | xargs rm -f"),
            (Allowed, "{ ls build; } | xargs rm -rf"),
            (Protected, "find ~ -type f | xargs rm -f"),
            (Protected, "(find ~ -type f) | xargs rm -f"),
            (Protected, "{ find ~ -type f; } | xargs rm -f"),
            (Protected, "{ ls build; find ~; } | xargs rm -f"),
            (
                Protected,
                "for d in a; do find ~ -type f; done | xargs rm -f",
            ),
            (Protected, "(ls /) | (xargs rm -rf)"),
            (Protected, "(find ~) | grep x | xargs rm -f"),
            (Protected, "{ (find ~) | xargs rm -f; } | cat"),
            // The listing is checked along with the working directory.
            (Protected, "cd ~; (ls proj/build) | xargs rm -rf"),
        ],
    );
}

#[test]
fn sudos_long_options_take_their_values() {
    expect_ruled(
        &read_only_git(),
        &linux(),
        ShellDialect::Posix,
        &[
            (true, "sudo --user root git status"),
            (false, "sudo --user root git push"),
            (false, "sudo --user=root git push"),
            (false, "sudo --group wheel --prompt x git push"),
            (
                false,
                "sudo --host h --other-user u --role r --type t git push",
            ),
            (false, "sudo --close-from 3 --command-timeout 10 git push"),
            (false, "sudo --chdir /tmp --chroot / git push"),
        ],
    );
    expect(
        &linux(),
        ShellDialect::Posix,
        &[(Protected, "sudo --user root rm -rf /")],
    );
}

#[test]
fn a_wrappers_directory_is_where_its_command_runs() {
    // The working directory is $HOME/proj.
    expect(
        &linux(),
        ShellDialect::Posix,
        &[
            (Allowed, "env -C build rm -rf *"),
            (Allowed, "env -C /tmp rm -rf x"),
            (Allowed, "cd / && env -C /tmp rm -rf x"),
            (Allowed, "env -C / ls"),
            (Allowed, "env -C / true; rm -rf *"),
            (Protected, "env -C / rm -rf *"),
            (Protected, "env -C/ rm -rf *"),
            (Protected, "env --chdir=/ rm -rf *"),
            (Protected, "env --chdir / rm -rf *"),
            (Protected, "env -C .. rm -rf *"),
            (Protected, "env -C ~ rm -rf *"),
            (Protected, "sudo -D / rm -rf *"),
            (Protected, "sudo --chdir=/ rm -rf *"),
            (Protected, "sudo --chdir / rm -rf *"),
            (Protected, "sudo -u root -D / rm -rf *"),
            (Protected, "unshare -w / rm -rf *"),
            (Protected, "unshare --wd=/ rm -rf *"),
            (Protected, "unshare --wd / rm -rf *"),
            (Protected, "env -C /tmp sudo -D .. rm -rf *"),
            (Protected, "env -C / ls | xargs rm -rf"),
            // Only known when it runs: outside the working directory.
            (Outside, "env -C \"$(cat dir.txt)\" rm -rf *"),
            (Outside, "sudo -D \"$(cat dir.txt)\" rm -rf build"),
        ],
    );
}

#[test]
fn a_program_known_only_when_the_command_runs_is_refused_only_under_rules() {
    let unknown = [
        "$(which git) push",
        "source env.sh; $TOOL push",
        "bash -c \"$(cat script.sh)\"",
        "eval \"$(cat command.txt)\"",
        "ls | xargs \"$(which git)\" add",
    ];
    let rules = [CommandRule::deny("cargo * publish *")];
    for command in unknown {
        expect(&linux(), ShellDialect::Posix, &[(Allowed, command)]);
        expect_ruled(&rules, &linux(), ShellDialect::Posix, &[(false, command)]);
    }
    let refusal = CommandGuard {
        rules: rules.to_vec(),
        ..CommandGuard::default()
    }
    .check_on(&linux(), ShellDialect::Posix, "$(which git) push")
    .unwrap_err();
    assert!(
        refusal
            .message
            .contains("its program is only known when the command runs"),
        "{}",
        refusal.message
    );
}

#[test]
fn rules_leave_the_path_checks_in_place() {
    let guard = CommandGuard {
        rules: vec![CommandRule::allow("rm *")],
        ..CommandGuard::default()
    };
    let host = linux();
    assert_eq!(
        verdict(&guard, &host, ShellDialect::Posix, "rm -rf ~"),
        Protected
    );
    assert_eq!(
        verdict(&guard, &host, ShellDialect::Posix, "rm -rf ../other"),
        Outside
    );
    assert_eq!(
        verdict(&guard, &host, ShellDialect::Posix, "rm -rf build"),
        Allowed
    );
    // Turned off, the guard applies no rules either.
    let off = CommandGuard {
        dangerously_allow_all_commands: true,
        rules: vec![CommandRule::deny("ls *")],
        ..CommandGuard::default()
    };
    assert_eq!(verdict(&off, &host, ShellDialect::Posix, "ls -la"), Allowed);
}

#[test]
fn rules_apply_in_cmd_and_powershell() {
    let rules = read_only_git();
    expect_ruled(
        &rules,
        &windows(),
        ShellDialect::Cmd,
        &[
            (true, "git status"),
            (true, "git.exe log -1"),
            (true, "dir & git diff"),
            (false, "git push"),
            (false, "cd sub && GIT.EXE commit -m x"),
            (false, "cmd /c git push"),
        ],
    );
    expect_ruled(
        &rules,
        &windows(),
        ShellDialect::PowerShell,
        &[
            (true, "git status"),
            (true, "& git log -1"),
            (true, "$files = Get-ChildItem; $files | Measure-Object"),
            (
                true,
                "foreach ($f in $files) { Write-Output \"$($f.Name)\" }",
            ),
            (true, "Write-Output \"$(git log -1 --format=%H)\""),
            (false, "git push"),
            (false, "& 'C:\\Program Files\\Git\\bin\\git.exe' push"),
            (false, "$tool = (Get-Command git).Source; & $tool push"),
            (false, "Write-Output \"$(git push)\""),
            (false, "if (Test-Path x) { git commit -m x }"),
            (false, "pwsh -Command \"git push\""),
        ],
    );
}

#[test]
fn rule_refusals_explain_the_rule() {
    let guard = CommandGuard {
        rules: vec![
            CommandRule::allow("git status *"),
            CommandRule::allow("git *").within("scratch"),
            CommandRule::deny("cargo * publish *").because("Publishing is the user's."),
        ],
        ..CommandGuard::default()
    };
    let refusal = guard
        .check_on(
            &linux(),
            ShellDialect::Posix,
            "cd sub && git push origin main",
        )
        .unwrap_err();
    assert!(!refusal.protected);
    assert_eq!(refusal.target, "git push origin main");
    assert_eq!(
        refusal.message,
        "Refused: `git push origin main` isn't one of the `git` commands allowed here: `git status *`, `git *` (in `scratch`). Use one of those, or do the work another way. Other commands need the user's permission in the terminal settings."
    );
    let refusal = guard
        .check_on(&linux(), ShellDialect::Posix, "cargo publish --dry-run")
        .unwrap_err();
    assert_eq!(
        refusal.message,
        "Refused: `cargo publish --dry-run` matches the command rule `cargo * publish *`. Publishing is the user's."
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

/// A Windows workspace right under a drive's root.
fn windows_top_level() -> Host {
    host(
        PathStyle {
            windows: true,
            ignore_case: true,
        },
        r"D:\RoboEnv",
        r"C:\Users\luke",
        &[r"C:\Users\luke\AppData\Local\Temp"],
        &[
            ("USERPROFILE", r"C:\Users\luke"),
            ("TEMP", r"C:\Users\luke\AppData\Local\Temp"),
        ],
    )
}

/// A workspace right under a drive's root, such as `D:\RoboEnv`, holds its
/// files like any other: commands write, move and delete inside it, while
/// other top-level directories and the drive's root stay protected.
#[test]
fn a_workspace_at_the_top_of_a_drive_holds_its_files() {
    let host = windows_top_level();
    expect(
        &host,
        ShellDialect::PowerShell,
        &[
            (
                Allowed,
                r"Set-Content -Path 'D:\RoboEnv\guardtest.tmp' -Value 'x'",
            ),
            (
                Allowed,
                r"Set-Content -Path '.scratch\trademark\raw\tmview_index.html' -Value $r.Content -Encoding utf8",
            ),
            (
                Allowed,
                r"$r.Content | Out-File -Encoding utf8 D:\RoboEnv\.scratch\trademark\raw\tmview_index.html",
            ),
            (
                Allowed,
                r"python -I .scratch\trademark\scripts\tmview_run.py > .scratch\trademark\raw\tmview_studios_out.txt 2>&1",
            ),
            (
                Allowed,
                r"Move-Item 'Moonkite Studios Crescent Logo.png' 'assets\brand\moonkite-logo.png'",
            ),
            (Allowed, r"Remove-Item -Recurse -Force D:\RoboEnv\build"),
            (Outside, r"Remove-Item -Recurse -Force D:\Other\build"),
            (Protected, r"Remove-Item -Recurse -Force D:\Other"),
            (Protected, r"Remove-Item -Recurse -Force D:\"),
        ],
    );
    expect(
        &host,
        ShellDialect::Cmd,
        &[
            (Allowed, r"del D:\RoboEnv\out\old.txt"),
            (Allowed, r"rd /s /q build"),
            (Outside, r"del D:\Other\old.txt"),
        ],
    );
}

/// PowerShell's provider drives that aren't files (Env:, Variable:,
/// Function:, Alias:) aren't paths in the working directory. The registry
/// and the certificate store are outside it.
#[test]
fn powershell_provider_drives() {
    expect(
        &windows_top_level(),
        ShellDialect::PowerShell,
        &[
            (Allowed, r"Remove-Item Env:\SSL_CERT_FILE"),
            (
                Allowed,
                r"Remove-Item Env:GH_TOKEN -ErrorAction SilentlyContinue",
            ),
            (Allowed, r"Remove-Item env:\GH_TOKEN"),
            (Allowed, r"Set-Content Env:\MODE -Value x"),
            (Allowed, r"Remove-Item Variable:\old"),
            (Allowed, r"Remove-Item Function:\prompt"),
            (Allowed, r"Remove-Item Alias:\ls"),
            (Outside, r"Remove-Item HKCU:\Software\Example -Recurse"),
            (Outside, r"Remove-Item -Path HKLM:\SOFTWARE\Example"),
            (
                Outside,
                r"Remove-Item Registry::HKEY_CURRENT_USER\Software\Example",
            ),
            (Outside, r"Remove-Item Cert:\CurrentUser\My\ABC123"),
            (Allowed, r"Remove-Item Temp:\arko-build.log"),
            (
                Allowed,
                r"Remove-Item Microsoft.PowerShell.Core\FileSystem::D:\RoboEnv\out.txt",
            ),
            (Outside, r"Remove-Item FileSystem::D:\Other\out.txt"),
        ],
    );
}

/// A top-level workspace on Linux holds its files too, while a system
/// directory as the workspace still grants nothing.
#[test]
fn top_level_workspaces_and_system_directories() {
    let linux_style = PathStyle {
        windows: false,
        ignore_case: false,
    };
    expect(
        &host(linux_style, "/data", "/home/luke", &["/tmp"], &[]),
        ShellDialect::Posix,
        &[
            (Allowed, "rm -rf build && rm -f /data/out/old.log"),
            (Allowed, "mv a.txt b.txt"),
            (Outside, "rm -rf /srv/build"),
            (Protected, "rm -rf /srv"),
        ],
    );
    let windows_style = PathStyle {
        windows: true,
        ignore_case: true,
    };
    expect(
        &host(
            windows_style,
            r"C:\Windows",
            r"C:\Users\luke",
            &[r"C:\Users\luke\AppData\Local\Temp"],
            &[],
        ),
        ShellDialect::PowerShell,
        &[(Protected, r"Remove-Item -Recurse C:\Windows\Temp\x")],
    );
}

/// After `cd` into a drive that isn't the filesystem, relative paths aren't
/// known, so destructive commands on them need permission.
#[test]
fn powershell_cd_into_a_provider_drive() {
    expect(
        &windows_top_level(),
        ShellDialect::PowerShell,
        &[
            (Outside, r"cd HKCU:\Software; Remove-Item Example -Recurse"),
            (Outside, r"Set-Location Env:; Remove-Item GH_TOKEN"),
            (
                Allowed,
                r"cd FileSystem::D:\RoboEnv\out; Remove-Item old.txt",
            ),
            (Allowed, r"cd D:\RoboEnv\out; Remove-Item old.txt"),
        ],
    );
}

#[test]
fn powershell_provider_paths_parse() {
    let temp = Some(r"C:\Temp");
    let parse = |text: &str| powershell_provider(text, temp);
    assert_eq!(parse(r"C:\work\x.txt"), None);
    assert_eq!(parse(r"x.txt"), None);
    assert_eq!(parse(r"..\x.txt"), None);
    assert_eq!(parse(r"Env:\PATH"), Some(ProviderPath::NotFile));
    assert_eq!(parse(r"ENV:PATH"), Some(ProviderPath::NotFile));
    assert_eq!(parse(r"variable:x"), Some(ProviderPath::NotFile));
    assert_eq!(parse(r"HKLM:\SOFTWARE"), Some(ProviderPath::Elsewhere));
    assert_eq!(parse(r"Cert:\CurrentUser"), Some(ProviderPath::Elsewhere));
    assert_eq!(parse(r"MyDrive:\x"), Some(ProviderPath::Elsewhere));
    assert_eq!(
        parse(r"Temp:\a\b.log"),
        Some(ProviderPath::File(r"C:\Temp\a\b.log".to_string()))
    );
    assert_eq!(
        powershell_provider("Temp:/a.log", Some("/tmp")),
        Some(ProviderPath::File("/tmp/a.log".to_string()))
    );
    assert_eq!(
        powershell_provider(r"Temp:\a", None),
        Some(ProviderPath::Elsewhere)
    );
    assert_eq!(
        parse(r"Microsoft.PowerShell.Core\FileSystem::C:\x"),
        Some(ProviderPath::File(r"C:\x".to_string()))
    );
    assert_eq!(
        parse(r"Registry::HKEY_CURRENT_USER\Software"),
        Some(ProviderPath::Elsewhere)
    );
    assert_eq!(parse(r"Environment::PATH"), Some(ProviderPath::NotFile));
    assert_eq!(parse(r"C:\a\b::c"), None);
}
