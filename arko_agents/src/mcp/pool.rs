//! The servers shared by the hubs of a process (`"shared": true`): one
//! process per command line, kept while a hub uses it.
//!
//! The pool is process-wide, behind a static, rather than an object a host
//! passes to each hub. A process can have only one meaningful pool, and a
//! host that opens several sessions (an app with a session per window)
//! would otherwise need a static of its own to hand the same pool to each.
//! Its servers run on a runtime of their own, since the runtime of the
//! session that started a server may end while other sessions still use
//! it.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::Duration;

use rmcp::service::{Peer, RoleClient};

use super::config::McpServerConfig;
use super::connection::Connection;

/// What makes two servers the same: the command, arguments and
/// environment, as expanded.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    command: String,
    args: Vec<String>,
    env: Vec<(String, String)>,
}

impl Key {
    fn of(config: &McpServerConfig) -> Self {
        let mut env: Vec<(String, String)> = config
            .env
            .iter()
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect();
        env.sort();
        Self {
            command: config.command.clone(),
            args: config.args.clone(),
            env,
        }
    }
}

struct Pool {
    runtime: tokio::runtime::Runtime,
    servers: Mutex<HashMap<Key, Weak<SharedServer>>>,
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool {
        runtime: tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("arko-mcp-shared")
            .enable_all()
            .build()
            .expect("the runtime of shared MCP servers can be built"),
        servers: Mutex::new(HashMap::new()),
    })
}

/// The pool's server for `config`'s command line, the one other hubs use
/// if any does. The server stops when the last hub drops it.
pub(crate) fn acquire(config: &McpServerConfig) -> Arc<SharedServer> {
    let key = Key::of(config);
    let mut servers = pool().servers.lock().unwrap();
    servers.retain(|_, server| server.strong_count() > 0);
    if let Some(server) = servers.get(&key).and_then(Weak::upgrade) {
        return server;
    }
    let server = Arc::new(SharedServer {
        config: config.clone(),
        lifecycle: tokio::sync::Mutex::new(()),
        connection: Mutex::new(None),
    });
    servers.insert(key, Arc::downgrade(&server));
    server
}

/// How many hubs use the server `config` names, for tests.
#[cfg(all(test, unix))]
pub(crate) fn users(config: &McpServerConfig) -> usize {
    pool()
        .servers
        .lock()
        .unwrap()
        .get(&Key::of(config))
        .map_or(0, Weak::strong_count)
}

/// One server of the pool. It runs while any hub holds it.
pub(crate) struct SharedServer {
    config: McpServerConfig,
    /// Held while the server starts.
    lifecycle: tokio::sync::Mutex<()>,
    connection: Mutex<Option<Connection>>,
}

impl SharedServer {
    /// The running server's connection, or `None` when it isn't running.
    pub(crate) fn peer(&self) -> Option<Peer<RoleClient>> {
        let connection = self.connection.lock().unwrap();
        connection
            .as_ref()
            .filter(|connection| connection.exited().is_none())
            .map(Connection::peer)
    }

    /// How the server ended on its own, while no hub has started it again.
    pub(crate) fn exited(&self) -> Option<String> {
        self.connection.lock().unwrap().as_ref()?.exited()
    }

    pub(crate) fn instructions(&self) -> Option<String> {
        self.connection.lock().unwrap().as_ref()?.instructions()
    }

    /// How many hubs use it.
    pub(crate) fn users(self: &Arc<Self>) -> usize {
        Arc::strong_count(self)
    }

    /// Starts the server on the pool's runtime unless it runs, within
    /// `timeout`, and returns its connection. One that ended is started
    /// again.
    pub(crate) async fn start(&self, timeout: Duration) -> Result<Peer<RoleClient>, String> {
        let _starting = self.lifecycle.lock().await;
        if let Some(peer) = self.peer() {
            return Ok(peer);
        }
        let ended = self.connection.lock().unwrap().take();
        let config = self.config.clone();
        let opened = pool()
            .runtime
            .spawn(async move {
                if let Some(mut ended) = ended {
                    ended.stop().await;
                }
                match tokio::time::timeout(timeout, Connection::open(&config)).await {
                    Ok(opened) => opened.map_err(|error| error.reason()),
                    Err(_) => Err(format!("it didn't start within {} s", timeout.as_secs())),
                }
            })
            .await
            .map_err(|error| {
                log::warn!(
                    "MCP server '{}': its start was stopped ({error})",
                    self.config.name
                );
                "its start was stopped".to_string()
            })??;
        let peer = opened.peer();
        *self.connection.lock().unwrap() = Some(opened);
        Ok(peer)
    }
}

impl Drop for SharedServer {
    /// The last hub let go: the server stops, with what it started, on the
    /// pool's runtime.
    fn drop(&mut self) {
        if let Some(mut connection) = self.connection.get_mut().unwrap().take() {
            pool().runtime.spawn(async move { connection.stop().await });
        }
    }
}
