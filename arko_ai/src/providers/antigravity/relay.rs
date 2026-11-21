//! The MCP relay between agy and the host's tools.
//!
//! agy starts its MCP servers as child processes on stdio, so the profile
//! names the host's helper (`HELPER relay DIR/relay.json`) as its server.
//! The helper ([`run_relay`]) connects to the [`RelayHost`] the host runs on
//! 127.0.0.1, gives it the session's token, and then copies bytes both ways
//! between its stdin and stdout and the connection. The host serves a
//! [`ToolServer`] to each connection that gave the token. It answers each
//! request in a task of its own, so calls run at the same time, and a
//! `notifications/cancelled` from agy stops its call.
//!
//! The token goes only in agy's environment ([`RelayHost::env`]), which the
//! helper inherits. The relay file holds just the host's address.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::net::{Ipv4Addr, SocketAddr};
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::FutureExt;
use serde_json::{Value, json};
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio::task::{AbortHandle, JoinSet};
use uuid::Uuid;

use super::profile::write_private;
use crate::providers::claude_code::ToolServer;

/// The environment variable that carries the relay's token from the host
/// through agy to the helper.
pub const RELAY_TOKEN_ENV: &str = "ARKO_AGY_RELAY_TOKEN";

/// How long a new connection has to give the token.
const HELLO_TIMEOUT: Duration = Duration::from_secs(10);

/// The longest first line a connection may send, in bytes.
const MAX_HELLO: usize = 1024;

/// The longest message agy may send, in bytes.
const MAX_FRAME: usize = 64 * 1024 * 1024;

/// How long the host waits after a failed accept (too many open files, for
/// one) before it accepts again.
const ACCEPT_RETRY: Duration = Duration::from_millis(100);

/// Serves a [`ToolServer`] on 127.0.0.1 to the relays that give its token.
/// Dropping it stops listening and closes every connection. The calls still
/// running are dropped, as a cancelled call is.
pub struct RelayHost {
    address: SocketAddr,
    token: String,
    serving: AbortHandle,
}

impl fmt::Debug for RelayHost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RelayHost")
            .field("address", &self.address)
            .finish_non_exhaustive()
    }
}

impl RelayHost {
    /// Listens on a free port of 127.0.0.1 with a new random token. The
    /// server's calls run on the current tokio runtime.
    pub async fn bind(server: Arc<ToolServer>) -> io::Result<Self> {
        Self::bind_with(server, HELLO_TIMEOUT).await
    }

    async fn bind_with(server: Arc<ToolServer>, hello_timeout: Duration) -> io::Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await?;
        let address = listener.local_addr()?;
        let token = new_token();
        let serving = tokio::spawn(serve(
            listener,
            server,
            Arc::from(token.as_str()),
            hello_timeout,
        ))
        .abort_handle();
        Ok(Self {
            address,
            token,
            serving,
        })
    }

    /// The address for [`write_address_file`].
    pub fn address(&self) -> SocketAddr {
        self.address
    }

    /// The environment variable (name and value) that agy needs so the relay
    /// it starts can connect. Set it after the caller's extra environment,
    /// so the caller can't replace it. No file and no command line holds it.
    pub fn env(&self) -> (&'static str, &str) {
        (RELAY_TOKEN_ENV, &self.token)
    }
}

impl Drop for RelayHost {
    fn drop(&mut self) {
        self.serving.abort();
    }
}

/// Writes `address` to the relay file at `path`, readable by the user
/// alone, for [`run_relay`]. This is small blocking file I/O, so async code
/// calls it through `spawn_blocking`.
pub fn write_address_file(path: impl AsRef<Path>, address: SocketAddr) -> io::Result<()> {
    write_private(
        path.as_ref(),
        format!("{}\n", json!({"address": address.to_string()})).as_bytes(),
    )
}

/// The helper's side of the relay, as agy's MCP server. It reads the host's
/// address from the relay file at `path` and the token from
/// [`RELAY_TOKEN_ENV`], connects, and copies stdin to the host and the
/// host's replies to stdout until the host closes the connection. It needs a
/// tokio runtime with IO enabled.
pub async fn run_relay(path: &Path) -> io::Result<()> {
    let token = std::env::var(RELAY_TOKEN_ENV)
        .ok()
        .filter(|token| !token.is_empty())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("{RELAY_TOKEN_ENV} isn't set. Arko sets it for the agy it starts."),
            )
        })?;
    let stream = connect(path, &token).await?;
    let (chunks, input) = mpsc::channel(16);
    // tokio reads stdin on a thread it can't stop, and a read still waiting
    // there would hold up the runtime's shutdown after the host closes the
    // connection. A thread of its own ends with the process.
    std::thread::Builder::new()
        .name("relay-stdin".into())
        .spawn(move || forward_stdin(chunks))?;
    pump(stream, input, tokio::io::stdout()).await
}

/// Sends what arrives on stdin to `chunks` until stdin ends or the relay
/// stops taking it.
fn forward_stdin(chunks: mpsc::Sender<Vec<u8>>) {
    use std::io::Read as _;

    let mut stdin = io::stdin().lock();
    let mut buffer = vec![0; 64 * 1024];
    loop {
        match stdin.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => {
                if chunks.blocking_send(buffer[..read].to_vec()).is_err() {
                    break;
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
}

/// Connects to the host named in the relay file at `path` and gives it
/// `token`.
async fn connect(path: &Path, token: &str) -> io::Result<TcpStream> {
    let text = tokio::fs::read_to_string(path).await.map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("Couldn't read {}. {error}", path.display()),
        )
    })?;
    let address = parse_address(&text).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{} has no loopback address.", path.display()),
        )
    })?;
    let mut stream = TcpStream::connect(address).await?;
    stream.set_nodelay(true)?;
    stream
        .write_all(format!("{}\n", json!({"token": token})).as_bytes())
        .await?;
    Ok(stream)
}

/// The address in a relay file, if it's on this machine.
fn parse_address(text: &str) -> Option<SocketAddr> {
    let address: SocketAddr = serde_json::from_str::<Value>(text)
        .ok()?
        .get("address")?
        .as_str()?
        .parse()
        .ok()?;
    address.ip().is_loopback().then_some(address)
}

/// Copies `input` to the host and the host's bytes to `output`, until the
/// host closes the connection. When `input` ends first, the host hears that
/// agy is done, ends the connection's calls and closes it.
async fn pump(
    stream: TcpStream,
    mut input: mpsc::Receiver<Vec<u8>>,
    mut output: impl AsyncWrite + Unpin,
) -> io::Result<()> {
    let (mut from_host, mut to_host) = stream.into_split();
    let upstream = async move {
        while let Some(chunk) = input.recv().await {
            to_host.write_all(&chunk).await?;
        }
        to_host.shutdown().await
    };
    let downstream = async move {
        tokio::io::copy(&mut from_host, &mut output).await?;
        output.flush().await
    };
    tokio::pin!(upstream, downstream);
    let mut sending = true;
    loop {
        tokio::select! {
            ended = &mut downstream => return ended,
            // A send fails only when the connection is gone, which
            // `downstream` sees too.
            _ = &mut upstream, if sending => sending = false,
        }
    }
}

/// Accepts connections until the task is aborted.
async fn serve(
    listener: TcpListener,
    server: Arc<ToolServer>,
    token: Arc<str>,
    hello_timeout: Duration,
) {
    // The connections go with this task, so aborting it ends them too.
    let mut connections = JoinSet::new();
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _)) => stream,
            Err(error) => {
                log::warn!("The Antigravity relay couldn't accept a connection. {error}");
                tokio::time::sleep(ACCEPT_RETRY).await;
                continue;
            }
        };
        while connections.try_join_next().is_some() {}
        connections.spawn(connection(
            stream,
            server.clone(),
            token.clone(),
            hello_timeout,
        ));
    }
}

/// Serves one relay, once it has given `token` within `hello_timeout`.
async fn connection(
    stream: TcpStream,
    server: Arc<ToolServer>,
    token: Arc<str>,
    hello_timeout: Duration,
) {
    let _ = stream.set_nodelay(true);
    let (read, mut write) = stream.into_split();
    let mut reader = BufReader::new(read);
    let hello = tokio::time::timeout(hello_timeout, read_frame(&mut reader, MAX_HELLO)).await;
    let given = match hello {
        Ok(Ok(Some(line))) => line,
        _ => Vec::new(),
    };
    if !hello_matches(&given, &token) {
        log::warn!("The Antigravity relay closed a connection that didn't give its token.");
        return;
    }
    let (replies, mut outgoing) = mpsc::unbounded_channel::<String>();
    let reading = async move {
        // The calls go with the connection. Those still running when it
        // closes are aborted, as their answers can't reach agy.
        let mut calls = JoinSet::new();
        let mut by_id: HashMap<String, AbortHandle> = HashMap::new();
        loop {
            let line = match read_frame(&mut reader, MAX_FRAME).await {
                Ok(Some(line)) => line,
                Ok(None) => break,
                Err(error) => {
                    log::warn!("The Antigravity relay closed a connection. {error}");
                    break;
                }
            };
            let Ok(message) = serde_json::from_slice::<Value>(&line) else {
                if !line.trim_ascii().is_empty() {
                    log::debug!("The Antigravity relay skipped a line that isn't JSON.");
                }
                continue;
            };
            // Calls that ended leave as the next message comes.
            while calls.try_join_next().is_some() {}
            by_id.retain(|_, call| !call.is_finished());
            let method = message
                .get("method")
                .and_then(Value::as_str)
                .map(str::to_string);
            let id = message.get("id").cloned();
            match (method.as_deref(), id) {
                (Some("notifications/cancelled"), None) => {
                    // agy gave up on the call, and it expects no answer.
                    let cancelled = message
                        .pointer("/params/requestId")
                        .and_then(|id| by_id.remove(&id.to_string()));
                    if let Some(call) = cancelled {
                        call.abort();
                    }
                }
                (Some(_), Some(id)) => {
                    let key = id.to_string();
                    let (server, replies) = (server.clone(), replies.clone());
                    let call = calls.spawn(async move {
                        // A tool that panics still answers, or agy would wait
                        // for it until its own limit.
                        let answered = AssertUnwindSafe(server.handle(&message))
                            .catch_unwind()
                            .await;
                        let reply = match answered {
                            Ok(Some(reply)) => reply,
                            Ok(None) => return,
                            Err(_) => json!({
                                "jsonrpc": "2.0",
                                "id": id,
                                "error": {"code": -32603, "message": "The host failed while answering."},
                            }),
                        };
                        let _ = replies.send(format!("{reply}\n"));
                    });
                    by_id.insert(key, call);
                }
                // Other notifications, and replies, as the host asks agy
                // nothing.
                _ => {}
            }
        }
    };
    let writing = async move {
        while let Some(line) = outgoing.recv().await {
            if write.write_all(line.as_bytes()).await.is_err() {
                break;
            }
        }
        let _ = write.shutdown().await;
    };
    tokio::join!(reading, writing);
}

/// Reads a line without its newline, or `None` at the end of the stream. A
/// line over `max` bytes is an error, as nothing after it can be read.
async fn read_frame<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    max: usize,
) -> io::Result<Option<Vec<u8>>> {
    let mut line = Vec::new();
    let limit = u64::try_from(max).unwrap_or(u64::MAX).saturating_add(1);
    let mut limited = AsyncReadExt::take(&mut *reader, limit);
    if limited.read_until(b'\n', &mut line).await? == 0 {
        return Ok(None);
    }
    if line.last() == Some(&b'\n') {
        line.pop();
    } else if line.len() > max {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("A message was over {max} bytes."),
        ));
    }
    Ok(Some(line))
}

/// Whether a connection's first line gives `token`.
fn hello_matches(line: &[u8], token: &str) -> bool {
    let Ok(hello) = serde_json::from_slice::<Value>(line) else {
        return false;
    };
    hello
        .get("token")
        .and_then(Value::as_str)
        .is_some_and(|given| same_bytes(given.as_bytes(), token.as_bytes()))
}

/// Whether `a` and `b` are equal, in a time that doesn't depend on where
/// they differ.
fn same_bytes(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |differ, (x, y)| differ | (x ^ y)) == 0
}

/// 244 random bits from the OS, as two version 4 UUIDs in 64 hex digits.
fn new_token() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use tokio::io::Lines;
    use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
    use tokio::sync::Barrier;

    use super::*;
    use crate::providers::claude_code::{ClaudeTool, ToolOutput};

    /// How long a test waits for each step before it fails.
    const PROGRESS: Duration = Duration::from_secs(5);

    /// Returns its `text`.
    struct Echo;

    #[async_trait]
    impl ClaudeTool for Echo {
        fn name(&self) -> String {
            "echo".into()
        }
        fn description(&self) -> String {
            "Returns its text.".into()
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {"text": {"type": "string"}}})
        }
        async fn call(&self, arguments: Value) -> ToolOutput {
            ToolOutput::text(arguments["text"].as_str().unwrap_or_default())
        }
    }

    /// Returns once as many calls wait as the barrier counts, so calls that
    /// run one at a time never return.
    struct Gather(Arc<Barrier>);

    #[async_trait]
    impl ClaudeTool for Gather {
        fn name(&self) -> String {
            "gather".into()
        }
        fn description(&self) -> String {
            "Waits for the other calls.".into()
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        async fn call(&self, _arguments: Value) -> ToolOutput {
            self.0.wait().await;
            ToolOutput::text("gathered")
        }
    }

    /// Runs until it's dropped, counting the calls running.
    struct Hang(Arc<AtomicUsize>);

    #[async_trait]
    impl ClaudeTool for Hang {
        fn name(&self) -> String {
            "hang".into()
        }
        fn description(&self) -> String {
            "Runs until it's stopped.".into()
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        async fn call(&self, _arguments: Value) -> ToolOutput {
            struct Running(Arc<AtomicUsize>);
            impl Drop for Running {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            self.0.fetch_add(1, Ordering::SeqCst);
            let _running = Running(self.0.clone());
            std::future::pending().await
        }
    }

    struct Panics;

    #[async_trait]
    impl ClaudeTool for Panics {
        fn name(&self) -> String {
            "panics".into()
        }
        fn description(&self) -> String {
            "Panics.".into()
        }
        fn input_schema(&self) -> Value {
            json!({"type": "object", "properties": {}})
        }
        async fn call(&self, _arguments: Value) -> ToolOutput {
            panic!("the tool failed")
        }
    }

    /// A server whose `gather` returns once `gathering` calls wait.
    fn server(running: &Arc<AtomicUsize>, gathering: usize) -> Arc<ToolServer> {
        Arc::new(
            ToolServer::new("arko")
                .tool(Arc::new(Echo))
                .tool(Arc::new(Gather(Arc::new(Barrier::new(gathering)))))
                .tool(Arc::new(Hang(running.clone())))
                .tool(Arc::new(Panics)),
        )
    }

    fn call(id: Value, tool: &str, arguments: Value) -> Value {
        json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "tools/call",
            "params": {
                "_meta": {"antigravity.google/conversation_id": "c1"},
                "name": tool,
                "arguments": arguments,
            },
        })
    }

    fn cancel(id: Value) -> Value {
        json!({
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"reason": "context canceled", "requestId": id},
        })
    }

    /// Whether `running` reaches `count` in time.
    async fn reaches(running: &AtomicUsize, count: usize) -> bool {
        for _ in 0..500 {
            if running.load(Ordering::SeqCst) == count {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        false
    }

    /// A relay talking to the host directly.
    struct Peer {
        lines: Lines<BufReader<OwnedReadHalf>>,
        write: OwnedWriteHalf,
    }

    impl Peer {
        async fn connect(address: SocketAddr, hello: &str) -> Self {
            let stream = TcpStream::connect(address).await.unwrap();
            let (read, write) = stream.into_split();
            let mut peer = Self {
                lines: BufReader::new(read).lines(),
                write,
            };
            peer.send_line(hello).await;
            peer
        }

        async fn with_token(host: &RelayHost) -> Self {
            Self::connect(host.address(), &json!({"token": host.env().1}).to_string()).await
        }

        async fn send_line(&mut self, line: &str) {
            self.write
                .write_all(format!("{line}\n").as_bytes())
                .await
                .unwrap();
        }

        async fn send(&mut self, message: Value) {
            self.send_line(&message.to_string()).await;
        }

        /// The next message, or `None` once the host closed the connection.
        async fn receive(&mut self) -> Option<Value> {
            let line = tokio::time::timeout(PROGRESS, self.lines.next_line())
                .await
                .expect("the host neither answered nor closed the connection")
                .ok()??;
            Some(serde_json::from_str(&line).unwrap())
        }
    }

    async fn send(input: &mpsc::Sender<Vec<u8>>, bytes: &[u8]) {
        input.send(bytes.to_vec()).await.unwrap();
    }

    async fn next(lines: &mut Lines<BufReader<tokio::io::DuplexStream>>) -> Option<Value> {
        let line = tokio::time::timeout(PROGRESS, lines.next_line())
            .await
            .expect("no reply in time")
            .unwrap()?;
        Some(serde_json::from_str(&line).unwrap())
    }

    /// The helper's side and the host's side together carry agy's MCP
    /// session, a message split across reads included, and the relay ends
    /// when agy closes its stdin.
    #[tokio::test]
    async fn the_relay_carries_mcp_between_agy_and_the_host() {
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind(server(&running, 1)).await.unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("relay.json");
        write_address_file(&file, host.address()).unwrap();
        let stream = connect(&file, host.env().1).await.unwrap();
        let (agy_stdin, input) = mpsc::channel(16);
        let (relay_stdout, agy_reads) = tokio::io::duplex(1 << 20);
        let relay = tokio::spawn(pump(stream, input, relay_stdout));
        let mut replies = BufReader::new(agy_reads).lines();

        send(
            &agy_stdin,
            b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"server/discover\",\"params\":{}}\n",
        )
        .await;
        let reply = next(&mut replies).await.unwrap();
        assert_eq!(reply["id"], 1);
        assert_eq!(reply["error"]["code"], -32601);

        let initialize = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25", "capabilities": {}},
        });
        let initialized = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let list = json!({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}});
        send(
            &agy_stdin,
            format!("{initialize}\n{initialized}\n{list}\n").as_bytes(),
        )
        .await;
        // Requests are answered on their own, so replies may come in any
        // order.
        let mut both = [
            next(&mut replies).await.unwrap(),
            next(&mut replies).await.unwrap(),
        ];
        both.sort_by_key(|reply| reply["id"].as_u64());
        let [started, listed] = both;
        assert_eq!(started["id"], 2);
        assert_eq!(started["result"]["protocolVersion"], "2025-11-25");
        assert_eq!(started["result"]["serverInfo"]["name"], "arko");
        assert_eq!(listed["id"], 3);
        let names: Vec<&str> = listed["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|tool| tool["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["echo", "gather", "hang", "panics"]);

        let line = format!("{}\n", call(json!(4), "echo", json!({"text": "hi there"})));
        let (first, second) = line.as_bytes().split_at(line.len() / 2);
        send(&agy_stdin, first).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        send(&agy_stdin, second).await;
        assert_eq!(
            next(&mut replies).await.unwrap(),
            json!({
                "jsonrpc": "2.0",
                "id": 4,
                "result": {"content": [{"type": "text", "text": "hi there"}], "isError": false},
            })
        );

        drop(agy_stdin);
        tokio::time::timeout(PROGRESS, relay)
            .await
            .expect("the relay didn't end with agy's stdin")
            .unwrap()
            .unwrap();
        assert_eq!(next(&mut replies).await, None);
    }

    /// Calls run at the same time. Every round's calls return only once all
    /// of them wait together, which calls answered one at a time never do.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn calls_run_at_the_same_time() {
        const CALLS: u64 = 16;
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind(server(&running, CALLS as usize))
            .await
            .unwrap();
        let mut peer = Peer::with_token(&host).await;
        for round in 0..20 {
            let ids: BTreeSet<u64> = (0..CALLS).map(|call| round * 100 + call).collect();
            for id in &ids {
                peer.send(call(json!(id), "gather", json!({}))).await;
            }
            let mut answered = BTreeSet::new();
            for _ in 0..CALLS {
                let reply = peer.receive().await.unwrap();
                assert_eq!(reply["result"]["content"][0]["text"], "gathered");
                answered.insert(reply["id"].as_u64().unwrap());
            }
            assert_eq!(answered, ids);
        }
    }

    /// A cancel stops the call it names and no other, the call gets no
    /// answer, and the connection goes on.
    #[tokio::test]
    async fn a_cancel_stops_its_call_without_an_answer() {
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind(server(&running, 1)).await.unwrap();
        let mut peer = Peer::with_token(&host).await;
        peer.send(call(json!(1), "hang", json!({}))).await;
        peer.send(call(json!("1"), "hang", json!({}))).await;
        assert!(reaches(&running, 2).await);

        peer.send(cancel(json!("1"))).await;
        assert!(reaches(&running, 1).await);
        peer.send(cancel(json!(1))).await;
        assert!(reaches(&running, 0).await);

        peer.send(json!({"jsonrpc": "2.0", "id": 2, "method": "ping"}))
            .await;
        assert_eq!(
            peer.receive().await.unwrap(),
            json!({"jsonrpc": "2.0", "id": 2, "result": {}})
        );
        // A cancel for a call that already ended changes nothing.
        peer.send(cancel(json!(2))).await;
        peer.send(call(json!(3), "echo", json!({"text": "still here"})))
            .await;
        assert_eq!(
            peer.receive().await.unwrap()["result"]["content"][0]["text"],
            "still here"
        );
    }

    #[tokio::test]
    async fn closing_a_connection_stops_its_calls() {
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind(server(&running, 1)).await.unwrap();
        let mut peer = Peer::with_token(&host).await;
        peer.send(call(json!(1), "hang", json!({}))).await;
        peer.send(call(json!(2), "hang", json!({}))).await;
        assert!(reaches(&running, 2).await);
        drop(peer);
        assert!(reaches(&running, 0).await);
    }

    #[tokio::test]
    async fn dropping_the_host_stops_its_calls_and_connections() {
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind(server(&running, 1)).await.unwrap();
        let address = host.address();
        let mut peer = Peer::with_token(&host).await;
        peer.send(call(json!(1), "hang", json!({}))).await;
        assert!(reaches(&running, 1).await);
        drop(host);
        assert!(reaches(&running, 0).await);
        assert_eq!(peer.receive().await, None);
        assert!(TcpStream::connect(address).await.is_err());
    }

    /// A connection without the token is closed before it reaches a tool,
    /// and one with it is still served.
    #[tokio::test]
    async fn only_connections_with_the_token_reach_the_tools() {
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind(server(&running, 1)).await.unwrap();
        let token = host.env().1.to_string();
        let mut wrong = token.clone().into_bytes();
        wrong[0] = if wrong[0] == b'0' { b'1' } else { b'0' };
        let wrong = String::from_utf8(wrong).unwrap();
        for hello in [
            json!({"token": wrong}).to_string(),
            json!({"token": &token[..63]}).to_string(),
            json!({}).to_string(),
            token.clone(),
            String::new(),
        ] {
            let mut peer = Peer::connect(host.address(), &hello).await;
            // The host may have closed the connection already, so the write
            // may fail.
            let line = format!("{}\n", call(json!(1), "hang", json!({})));
            let _ = peer.write.write_all(line.as_bytes()).await;
            assert_eq!(peer.receive().await, None, "{hello}");
        }
        assert_eq!(running.load(Ordering::SeqCst), 0);

        let mut peer = Peer::with_token(&host).await;
        peer.send(call(json!(1), "echo", json!({"text": "in"})))
            .await;
        assert_eq!(
            peer.receive().await.unwrap()["result"]["content"][0]["text"],
            "in"
        );
    }

    #[tokio::test]
    async fn a_connection_that_stays_silent_is_closed() {
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind_with(server(&running, 1), Duration::from_millis(100))
            .await
            .unwrap();
        let stream = TcpStream::connect(host.address()).await.unwrap();
        let mut lines = BufReader::new(stream).lines();
        let closed = tokio::time::timeout(PROGRESS, lines.next_line())
            .await
            .expect("the host kept a silent connection open");
        assert!(matches!(closed, Ok(None) | Err(_)));
    }

    #[tokio::test]
    async fn a_tool_that_panics_gets_an_error_answer() {
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind(server(&running, 1)).await.unwrap();
        let mut peer = Peer::with_token(&host).await;
        peer.send(call(json!(1), "panics", json!({}))).await;
        assert_eq!(
            peer.receive().await.unwrap(),
            json!({
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32603, "message": "The host failed while answering."},
            })
        );
        peer.send(call(json!(2), "echo", json!({"text": "after"})))
            .await;
        assert_eq!(peer.receive().await.unwrap()["id"], 2);
    }

    #[tokio::test]
    async fn frames_are_lines_up_to_the_cap() {
        let mut input: &[u8] = b"12345\n123456\n";
        assert_eq!(
            read_frame(&mut input, 5).await.unwrap(),
            Some(b"12345".to_vec())
        );
        assert_eq!(
            read_frame(&mut input, 5).await.unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        let mut input: &[u8] = b"{}\n\nlast";
        assert_eq!(
            read_frame(&mut input, 64).await.unwrap(),
            Some(b"{}".to_vec())
        );
        assert_eq!(read_frame(&mut input, 64).await.unwrap(), Some(Vec::new()));
        assert_eq!(
            read_frame(&mut input, 64).await.unwrap(),
            Some(b"last".to_vec())
        );
        assert_eq!(read_frame(&mut input, 64).await.unwrap(), None);
    }

    /// The relay file holds the address and nothing secret, and the relay
    /// connects only to this machine.
    #[tokio::test]
    async fn the_relay_file_holds_only_a_loopback_address() {
        let running = Arc::new(AtomicUsize::new(0));
        let host = RelayHost::bind(server(&running, 1)).await.unwrap();
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("relay.json");
        write_address_file(&file, host.address()).unwrap();
        let text = std::fs::read_to_string(&file).unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&text).unwrap(),
            json!({"address": host.address().to_string()})
        );
        assert_eq!(parse_address(&text), Some(host.address()));

        std::fs::write(&file, r#"{"address": "192.0.2.1:4000"}"#).unwrap();
        let error = connect(&file, host.env().1).await.unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        let error = connect(&dir.path().join("missing.json"), host.env().1)
            .await
            .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
    }

    #[tokio::test]
    async fn each_host_has_its_own_token_and_never_shows_it() {
        let running = Arc::new(AtomicUsize::new(0));
        let first = RelayHost::bind(server(&running, 1)).await.unwrap();
        let second = RelayHost::bind(server(&running, 1)).await.unwrap();
        let (name, token) = first.env();
        assert_eq!(name, RELAY_TOKEN_ENV);
        assert_eq!(token.len(), 64);
        assert!(token.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(token, second.env().1);
        assert!(first.address().ip().is_loopback());
        assert!(!format!("{first:?}").contains(token));
    }
}
