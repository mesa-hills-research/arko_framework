//! The private Antigravity profile (`agy --gemini-dir DIR`) that runs Arko's
//! agent with Arko's tools only, and the bodies of its two hooks.
//!
//! [`Profile::write`] makes DIR with these files.
//!
//! - `config/agents/arko.md` is the agent `--agent arko` starts. Its body
//!   replaces agy's whole system prompt, and its settings leave out agy's
//!   built-in tools, skills and the project's customizations.
//! - `config/mcp_config.json` starts the host's relay (`HELPER relay
//!   DIR/relay.json`) as agy's MCP server and marks each tool eager, so the
//!   model sees it as `mcp_arko_<tool>`.
//! - `config/hooks.json` loads only when agy falls back to its default agent.
//!   Its `SessionStart` hook ([`run_tripwire`]) and its `PreToolUse` hook
//!   ([`run_gate`], which denies every tool but Arko's) each add a line to
//!   `DIR/tripped`, which [`Profile::tripped`] checks.
//!
//! No file holds the relay's token. It reaches the relay through agy's
//! environment.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// The name of Arko's agent in the profile, given to agy as `--agent`.
pub const AGENT_NAME: &str = "arko";

/// The MCP server's name in `mcp_config.json`. The model sees each of its
/// tools as `mcp_arko_<tool>`.
pub const MCP_SERVER_NAME: &str = "arko";

/// How long agy waits for each hook, in seconds.
const HOOK_TIMEOUT_SECONDS: u64 = 10;

/// The most hook input the gate reads, in bytes.
const MAX_HOOK_INPUT: u64 = 64 * 1024 * 1024;

/// The name the model, agy's hooks and its permission requests use for
/// `tool`.
pub fn qualified_tool_name(tool: &str) -> String {
    format!("mcp_{MCP_SERVER_NAME}_{tool}")
}

/// The host executable and the arguments before `relay`, `gate` or
/// `tripwire`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HelperCommand {
    pub program: PathBuf,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
}

impl HelperCommand {
    /// The arguments that run `subcommand` on `path`.
    fn args_for(&self, subcommand: &str, path: &Path) -> io::Result<Vec<String>> {
        let mut args = self.args.clone();
        args.push(subcommand.to_string());
        args.push(utf8(path)?.to_string());
        Ok(args)
    }

    /// The command line `hooks.json` runs, each word quoted for a POSIX
    /// shell.
    fn command_line(&self, subcommand: &str, path: &Path) -> io::Result<String> {
        let mut words = vec![shell_quote(utf8(&self.program)?)];
        for arg in self.args_for(subcommand, path)? {
            words.push(shell_quote(&arg));
        }
        Ok(words.join(" "))
    }
}

/// What goes in a profile.
#[derive(Debug, Clone, Copy)]
pub struct ProfileSpec<'a> {
    /// Replaces agy's whole system prompt.
    pub system_prompt: &'a str,
    /// The names of the tools the relay serves, each offered to the model.
    pub tools: &'a [String],
    /// Runs the relay and the hooks.
    pub helper: &'a HelperCommand,
}

/// A profile directory that [`Profile::write`] made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Profile {
    dir: PathBuf,
}

impl Profile {
    /// Makes the directory `dir` (its parents too, as needed) that only the
    /// user can open, and writes the profile in it. `dir` must not exist yet.
    /// On an error nothing of it is left.
    ///
    /// The files are small, but this is blocking file I/O, so async code
    /// calls it through `spawn_blocking`.
    pub fn write(dir: impl Into<PathBuf>, spec: &ProfileSpec<'_>) -> io::Result<Self> {
        let dir = dir.into();
        if let Some(parent) = dir.parent() {
            fs::create_dir_all(parent)?;
        }
        private_dir(&dir)?;
        let profile = Self { dir };
        if let Err(error) = profile.fill(spec) {
            let _ = fs::remove_dir_all(&profile.dir);
            return Err(error);
        }
        Ok(profile)
    }

    fn fill(&self, spec: &ProfileSpec<'_>) -> io::Result<()> {
        let config = self.dir.join("config");
        let agents = config.join("agents");
        private_dir(&config)?;
        private_dir(&agents)?;
        write_private(
            &agents.join(format!("{AGENT_NAME}.md")),
            agent_file(spec.system_prompt).as_bytes(),
        )?;
        write_private(
            &config.join("mcp_config.json"),
            mcp_config(spec, &self.relay_file())?.to_string().as_bytes(),
        )?;
        write_private(
            &config.join("hooks.json"),
            hooks(spec.helper, &self.tripwire_file())?
                .to_string()
                .as_bytes(),
        )
    }

    /// The directory, given to agy as `--gemini-dir`.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Where agy writes its log, given to agy as `--log-file`.
    pub fn log_file(&self) -> PathBuf {
        self.dir.join("acp.log")
    }

    /// Where the relay finds the host's address, which `write_address_file`
    /// writes.
    pub fn relay_file(&self) -> PathBuf {
        self.dir.join("relay.json")
    }

    /// The file a hook adds a line to whenever it runs.
    pub fn tripwire_file(&self) -> PathBuf {
        self.dir.join("tripped")
    }

    /// Whether a hook ran, which happens only when agy fell back to its
    /// default agent with its own tools and the project's customizations.
    /// It looks for the tripwire file, a blocking call that returns at once
    /// on a local disk.
    pub fn tripped(&self) -> bool {
        self.tripwire_file().exists()
    }

    /// Deletes the directory and everything in it, with blocking file I/O.
    pub fn remove(&self) -> io::Result<()> {
        fs::remove_dir_all(&self.dir)
    }
}

/// The body of the profile's `PreToolUse` hook. It reads agy's hook input
/// from stdin and prints a decision that allows Arko's tools and denies the
/// rest. The hook runs only after agy fell back to its default agent, so it
/// also adds a line to the tripwire file at `path`.
///
/// Input it can't read is denied. The decision is printed even when the
/// line can't be added, and that error is returned after it.
pub fn run_gate(path: &Path) -> io::Result<()> {
    let mut bytes = Vec::new();
    let input = io::stdin()
        .lock()
        .take(MAX_HOOK_INPUT)
        .read_to_end(&mut bytes)
        .ok()
        .and_then(|_| serde_json::from_slice::<Value>(&bytes).ok());
    let (decision, recorded) = gate(input.as_ref(), path);
    let mut stdout = io::stdout().lock();
    stdout.write_all(format!("{decision}\n").as_bytes())?;
    stdout.flush()?;
    recorded
}

/// The body of the profile's `SessionStart` hook. It adds a line to the
/// tripwire file at `path`, as the hook runs only after agy fell back to its
/// default agent. It prints nothing and leaves stdin unread.
pub fn run_tripwire(path: &Path) -> io::Result<()> {
    record(path, &json!({"hook": "SessionStart"}))
}

/// The gate's decision on agy's hook `input`, and whether its line reached
/// the tripwire file at `path`.
fn gate(input: Option<&Value>, path: &Path) -> (Value, io::Result<()>) {
    let tool = input
        .and_then(|input| input.pointer("/toolCall/name"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let decision = if is_arko_tool(tool) {
        json!({"decision": "allow"})
    } else {
        json!({
            "decision": "deny",
            "reason": format!("Only the {} tools run here.", qualified_tool_name("")),
        })
    };
    let recorded = record(
        path,
        &json!({"hook": "PreToolUse", "tool": tool, "decision": decision["decision"]}),
    );
    (decision, recorded)
}

/// Whether `tool` is one of the relay's tools as agy names them.
fn is_arko_tool(tool: &str) -> bool {
    let prefix = qualified_tool_name("");
    tool.len() > prefix.len() && tool.starts_with(&prefix)
}

/// Adds `event` as a line to the file at `path`, made if needed for the
/// user alone.
fn record(path: &Path, event: &Value) -> io::Result<()> {
    let mut options = fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    // One write, so lines from hooks that run together don't interleave.
    options
        .open(path)?
        .write_all(format!("{event}\n").as_bytes())
}

/// The agent file, with the settings agy reads from its front matter and
/// `system_prompt` as its body.
fn agent_file(system_prompt: &str) -> String {
    let mut file = format!(
        "---\n\
         name: {AGENT_NAME}\n\
         description: Arko main agent.\n\
         excludeDefaultComponents: true\n\
         inheritCustomizations: false\n\
         mainAgent: true\n\
         subagent: false\n\
         ---\n\
         \n\
         {system_prompt}"
    );
    if !file.ends_with('\n') {
        file.push('\n');
    }
    file
}

/// `mcp_config.json`, with the relay as its one server. Each tool is listed
/// by name, as agy offers the model only the eager ones directly.
fn mcp_config(spec: &ProfileSpec<'_>, relay_file: &Path) -> io::Result<Value> {
    let tools: Map<String, Value> = spec
        .tools
        .iter()
        .map(|tool| (tool.clone(), json!({"eager": true})))
        .collect();
    let mut servers = Map::new();
    servers.insert(
        MCP_SERVER_NAME.to_string(),
        json!({
            "command": utf8(&spec.helper.program)?,
            "args": spec.helper.args_for("relay", relay_file)?,
            "tools": tools,
        }),
    );
    Ok(json!({"mcpServers": servers}))
}

/// `hooks.json`, with the tripwire at session start and the gate before
/// every tool call.
fn hooks(helper: &HelperCommand, tripwire_file: &Path) -> io::Result<Value> {
    let tripwire = helper.command_line("tripwire", tripwire_file)?;
    let gate = helper.command_line("gate", tripwire_file)?;
    Ok(json!({
        "arko-gate": {
            "SessionStart": [
                {"type": "command", "command": tripwire, "timeout": HOOK_TIMEOUT_SECONDS},
            ],
            "PreToolUse": [{
                "matcher": "*",
                "hooks": [{"type": "command", "command": gate, "timeout": HOOK_TIMEOUT_SECONDS}],
            }],
        },
    }))
}

/// `word` in single quotes, each quote in it closed, escaped and reopened.
fn shell_quote(word: &str) -> String {
    format!("'{}'", word.replace('\'', r"'\''"))
}

fn utf8(path: &Path) -> io::Result<&str> {
    path.to_str().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "agy's settings need UTF-8 paths, and {} isn't one.",
                path.display()
            ),
        )
    })
}

/// Makes the directory `path` for the user alone. It must not exist yet.
fn private_dir(path: &Path) -> io::Result<()> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(false);
    #[cfg(unix)]
    std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
    builder.create(path)
}

/// Writes `contents` to a new file beside `path` that only the user can
/// read, then renames it to `path`, so a reader finds the old file or the
/// whole new one.
pub(super) fn write_private(path: &Path, contents: &[u8]) -> io::Result<()> {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    let temporary = path.with_file_name(name);
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    let written = options
        .open(&temporary)
        .and_then(|mut file| file.write_all(contents))
        .and_then(|()| fs::rename(&temporary, path));
    if written.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    fn helper() -> HelperCommand {
        HelperCommand {
            program: "/opt/example/arko".into(),
            args: vec!["agy-helper".into()],
        }
    }

    fn tools() -> Vec<String> {
        vec!["shell".into(), "read_file".into()]
    }

    fn write(dir: &Path) -> io::Result<Profile> {
        Profile::write(
            dir,
            &ProfileSpec {
                system_prompt: "You are Arko.\n\n---\n\nKeep going.",
                tools: &tools(),
                helper: &helper(),
            },
        )
    }

    fn json_file(path: &Path) -> Value {
        serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn the_profile_holds_the_agent_the_relay_and_the_hooks() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("sessions").join("one");
        let profile = write(&dir).unwrap();
        assert_eq!(profile.dir(), dir.as_path());
        assert_eq!(profile.log_file(), dir.join("acp.log"));

        let agent = fs::read_to_string(dir.join("config/agents/arko.md")).unwrap();
        assert_eq!(
            agent,
            "---\nname: arko\ndescription: Arko main agent.\nexcludeDefaultComponents: true\n\
             inheritCustomizations: false\nmainAgent: true\nsubagent: false\n---\n\n\
             You are Arko.\n\n---\n\nKeep going.\n"
        );

        let relay_file = dir.join("relay.json");
        assert_eq!(
            json_file(&dir.join("config/mcp_config.json")),
            json!({"mcpServers": {"arko": {
                "command": "/opt/example/arko",
                "args": ["agy-helper", "relay", relay_file.to_str().unwrap()],
                "tools": {"shell": {"eager": true}, "read_file": {"eager": true}},
            }}})
        );

        let tripped = shell_quote(dir.join("tripped").to_str().unwrap());
        assert_eq!(
            json_file(&dir.join("config/hooks.json")),
            json!({"arko-gate": {
                "SessionStart": [{
                    "type": "command",
                    "command": format!("'/opt/example/arko' 'agy-helper' 'tripwire' {tripped}"),
                    "timeout": 10,
                }],
                "PreToolUse": [{"matcher": "*", "hooks": [{
                    "type": "command",
                    "command": format!("'/opt/example/arko' 'agy-helper' 'gate' {tripped}"),
                    "timeout": 10,
                }]}],
            }})
        );

        // Nothing but the profile's own files, and no temporary ones.
        let mut names: Vec<_> = walk(&dir)
            .into_iter()
            .map(|path| path.strip_prefix(&dir).unwrap().to_path_buf())
            .collect();
        names.sort();
        let expected: Vec<PathBuf> = [
            "config",
            "config/agents",
            "config/agents/arko.md",
            "config/hooks.json",
            "config/mcp_config.json",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        assert_eq!(names, expected);
    }

    fn walk(dir: &Path) -> Vec<PathBuf> {
        let mut found = Vec::new();
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                found.extend(walk(&path));
            }
            found.push(path);
        }
        found
    }

    #[cfg(unix)]
    #[test]
    fn only_the_user_can_open_the_profile() {
        use std::os::unix::fs::PermissionsExt;

        fn mode(path: &Path) -> u32 {
            fs::metadata(path).unwrap().permissions().mode() & 0o777
        }

        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("profile");
        let profile = write(&dir).unwrap();
        super::super::relay::write_address_file(
            &profile.relay_file(),
            "127.0.0.1:4000".parse().unwrap(),
        )
        .unwrap();
        record(&profile.tripwire_file(), &json!({"hook": "SessionStart"})).unwrap();
        assert_eq!(mode(&dir), 0o700);
        for path in walk(&dir) {
            let expected = if path.is_dir() { 0o700 } else { 0o600 };
            assert_eq!(mode(&path), expected, "{}", path.display());
        }
    }

    #[test]
    fn a_profile_takes_a_new_directory() {
        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("profile");
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("keep"), "mine").unwrap();
        let error = write(&dir).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read_to_string(dir.join("keep")).unwrap(), "mine");
    }

    #[cfg(unix)]
    #[test]
    fn a_profile_that_fails_leaves_nothing() {
        use std::os::unix::ffi::OsStrExt;

        let root = tempfile::tempdir().unwrap();
        let dir = root.path().join("profile");
        let helper = HelperCommand {
            program: PathBuf::from(std::ffi::OsStr::from_bytes(b"/opt/\xff/arko")),
            args: Vec::new(),
        };
        let error = Profile::write(
            &dir,
            &ProfileSpec {
                system_prompt: "Hi.",
                tools: &tools(),
                helper: &helper,
            },
        )
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(!dir.exists());
    }

    /// The hook commands give the helper its words unchanged, quotes and
    /// spaces included, when a shell runs them.
    #[cfg(unix)]
    #[test]
    fn hook_commands_survive_the_shell() {
        let helper = HelperCommand {
            program: "printf".into(),
            args: vec!["%s|".into(), "it's a \"test\" $HOME".into()],
        };
        let line = helper
            .command_line("gate", Path::new("/home/ada/my profile/tripped"))
            .unwrap();
        let output = std::process::Command::new("sh")
            .args(["-c", &line])
            .output()
            .unwrap();
        assert!(output.status.success());
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            "it's a \"test\" $HOME|gate|/home/ada/my profile/tripped|"
        );
    }

    #[test]
    fn the_gate_allows_only_arko_tools_and_records_every_call() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("tripped");
        let call = |name: &str| json!({"hookEventName": "PreToolUse", "toolCall": {"name": name}});
        let deny = json!({"decision": "deny", "reason": "Only the mcp_arko_ tools run here."});
        let cases = [
            (Some(call("mcp_arko_shell")), json!({"decision": "allow"})),
            (Some(call("run_command")), deny.clone()),
            (Some(call("mcp_arko_")), deny.clone()),
            (Some(call("mcp_other_shell")), deny.clone()),
            (Some(json!({"toolCall": {}})), deny.clone()),
            (None, deny.clone()),
        ];
        for (input, expected) in &cases {
            let (decision, recorded) = gate(input.as_ref(), &path);
            recorded.unwrap();
            assert_eq!(&decision, expected, "{input:?}");
        }
        let lines: Vec<Value> = fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), cases.len());
        assert_eq!(
            lines[0],
            json!({"hook": "PreToolUse", "tool": "mcp_arko_shell", "decision": "allow"})
        );
        assert_eq!(
            lines[1],
            json!({"hook": "PreToolUse", "tool": "run_command", "decision": "deny"})
        );
    }

    #[test]
    fn the_gate_still_decides_when_it_cant_record() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("missing").join("tripped");
        let input = json!({"toolCall": {"name": "run_command"}});
        let (decision, recorded) = gate(Some(&input), &path);
        assert_eq!(decision["decision"], "deny");
        assert!(recorded.is_err());
    }

    #[test]
    fn a_hook_that_ran_trips_the_profile() {
        let root = tempfile::tempdir().unwrap();
        let profile = write(&root.path().join("profile")).unwrap();
        assert!(!profile.tripped());
        run_tripwire(&profile.tripwire_file()).unwrap();
        assert!(profile.tripped());
        assert_eq!(
            fs::read_to_string(profile.tripwire_file()).unwrap(),
            "{\"hook\":\"SessionStart\"}\n"
        );
        profile.remove().unwrap();
        assert!(!profile.dir().exists());
    }

    #[test]
    fn qualified_names_match_what_agy_shows_the_model() {
        assert_eq!(qualified_tool_name("shell"), "mcp_arko_shell");
    }

    #[test]
    fn helper_commands_load_without_arguments() {
        let default_args: HelperCommand =
            serde_json::from_str(r#"{"program": "/opt/example/arko"}"#).unwrap();
        assert!(default_args.args.is_empty());
        assert_eq!(
            serde_json::to_value(&default_args).unwrap(),
            json!({"program": "/opt/example/arko"})
        );
        assert_eq!(
            serde_json::from_value::<HelperCommand>(serde_json::to_value(helper()).unwrap())
                .unwrap(),
            helper()
        );
    }
}
