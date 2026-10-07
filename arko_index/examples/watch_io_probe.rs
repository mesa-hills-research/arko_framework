//! Local diagnostic, not the host timing metric. Uses existing sources read-only.
use arko_index::bench_v1::{Error, Options, open_watch};
use std::{
    path::Path,
    time::{Duration, Instant},
};
fn main() -> Result<(), Error> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: watch_io_probe ROOT NEW_STATE_DIR".into());
    }
    let root = Path::new(&args[1]);
    let state = Path::new(&args[2]);
    let start = Instant::now();
    let mut w = open_watch(root, state, &Options::default())?;
    let open_s = start.elapsed().as_secs_f64();
    let snapshot = w.snapshot()?;
    let start = Instant::now();
    let delta = w
        .next_change(Duration::ZERO)?
        .ok_or("use a fresh state directory")?;
    w.ack(delta.sequence)?;
    let ack_s = start.elapsed().as_secs_f64();
    drop(w);
    let start = Instant::now();
    let mut w = open_watch(root, state, &Options::default())?;
    let reopen_s = start.elapsed().as_secs_f64();
    assert_eq!(snapshot, w.snapshot()?);
    assert!(w.next_change(Duration::ZERO)?.is_none());
    println!(
        "{}",
        serde_json::json!({"chunks":snapshot.len(),"open_s":open_s,"ack_s":ack_s,"reopen_s":reopen_s})
    );
    Ok(())
}
