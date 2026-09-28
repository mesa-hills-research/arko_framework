//! The opt-in stream idle timeout: the longest a streamed response may go
//! without receiving any data. Off unless set, since models can reason for a
//! long time without sending anything; see
//! [`LLMClient::with_stream_idle_timeout`](crate::client::LLMClient::with_stream_idle_timeout).

use super::StreamEvent;
use crate::error::Error;
use futures::{Stream, StreamExt, stream::BoxStream};
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

/// Parses a streamed response body with `parse`. With a `timeout`, the body
/// is cut off once no data has arrived for that long (any data counts,
/// keep-alive comments and ping events included). The parser then reports
/// the early end, usually as a response that ended before completing (or, if
/// the cut split an event, as a malformed one); either way the error becomes
/// [`Error::StreamTruncated`], which is transient, so the caller's usual
/// retries apply, and its message names the timeout. The wait for the
/// response's status and headers happens before this and is not covered.
pub(crate) fn parse_with_idle_timeout<S, B, E>(
    bytes: S,
    timeout: Option<Duration>,
    parse: impl FnOnce(
        BoxStream<'static, Result<B, E>>,
    ) -> BoxStream<'static, Result<StreamEvent, Error>>,
) -> BoxStream<'static, Result<StreamEvent, Error>>
where
    S: Stream<Item = Result<B, E>> + Send + 'static,
    B: Send + 'static,
    E: Send + 'static,
{
    let Some(timeout) = timeout else {
        return parse(bytes.boxed());
    };
    let timed_out = Arc::new(AtomicBool::new(false));
    let flag = timed_out.clone();
    let limited = async_stream::stream! {
        let mut bytes = Box::pin(bytes);
        loop {
            match tokio::time::timeout(timeout, bytes.next()).await {
                Ok(Some(item)) => yield item,
                Ok(None) => break,
                Err(_) => {
                    flag.store(true, Ordering::SeqCst);
                    break;
                }
            }
        }
    };
    parse(limited.boxed())
        .map(move |event| match event {
            // Every error after the cut comes from the cut.
            Err(_) if timed_out.load(Ordering::SeqCst) => Err(Error::StreamTruncated(format!(
                "no data arrived for {timeout:?}, the configured stream idle timeout, so the response was cut off"
            ))),
            other => other,
        })
        .boxed()
}
