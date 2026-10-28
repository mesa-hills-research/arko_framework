//! The ChatGPT backend's load-balancer routing cookies, `__oailb` and
//! `__cflb`. Codex keeps them and sends them back, so that later requests
//! reach the servers that served the earlier ones
//! (openai/codex `codex-rs/http-client/src/chatgpt_cloudflare_cookies.rs`).
//! A client keeps them in memory only, each for the host that set it, and no
//! other cookie.

use reqwest::header::{HeaderMap, HeaderValue, SET_COOKIE};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

/// The cookies kept. Every other cookie is ignored.
const ROUTING_COOKIE_NAMES: [&str; 2] = ["__oailb", "__cflb"];

/// The longest `Set-Cookie` header read, as in Codex.
const MAX_SET_COOKIE_BYTES: usize = 8192;

struct RoutingCookie {
    host: String,
    name: &'static str,
    value: String,
    expires: Option<SystemTime>,
    secure: bool,
}

impl RoutingCookie {
    fn is_live(&self, now: SystemTime) -> bool {
        self.expires.is_none_or(|at| at > now)
    }
}

/// The routing cookies of one client and its clones.
#[derive(Default)]
pub(super) struct RoutingCookies(Mutex<Vec<RoutingCookie>>);

impl RoutingCookies {
    /// Keeps the routing cookies a response from `url` sets, for its host,
    /// and forgets those it expires.
    pub(super) fn record(&self, url: &reqwest::Url, headers: &HeaderMap) {
        let Some(host) = url.host_str() else {
            return;
        };
        let now = SystemTime::now();
        let mut cookies = self.0.lock().unwrap();
        for header in headers.get_all(SET_COOKIE) {
            let Some(mut cookie) = header
                .to_str()
                .ok()
                .filter(|header| header.len() <= MAX_SET_COOKIE_BYTES)
                .and_then(|header| parse_set_cookie(header, now))
            else {
                continue;
            };
            cookies.retain(|kept| !(kept.host == host && kept.name == cookie.name));
            if cookie.is_live(now) {
                cookie.host = host.to_string();
                cookies.push(cookie);
            }
        }
    }

    /// The `Cookie` header for a request to `url`, when a routing cookie is
    /// kept for its host. Secure cookies go only over HTTPS.
    pub(super) fn header(&self, url: &reqwest::Url) -> Option<HeaderValue> {
        let host = url.host_str()?;
        let https = url.scheme() == "https";
        let now = SystemTime::now();
        let mut cookies = self.0.lock().unwrap();
        cookies.retain(|cookie| cookie.is_live(now));
        let pairs: Vec<String> = cookies
            .iter()
            .filter(|cookie| cookie.host == host && (https || !cookie.secure))
            .map(|cookie| format!("{}={}", cookie.name, cookie.value))
            .collect();
        if pairs.is_empty() {
            return None;
        }
        let mut value = HeaderValue::from_str(&pairs.join("; ")).ok()?;
        value.set_sensitive(true);
        Some(value)
    }
}

/// A routing cookie from a `Set-Cookie` header, without its host. `None`
/// for any other cookie.
fn parse_set_cookie(header: &str, now: SystemTime) -> Option<RoutingCookie> {
    let mut parts = header.split(';');
    let (name, value) = parts.next()?.split_once('=')?;
    let name = *ROUTING_COOKIE_NAMES
        .iter()
        .find(|known| **known == name.trim())?;
    let (mut max_age, mut expires, mut secure) = (None, None, false);
    for attribute in parts {
        let (key, argument) = attribute
            .split_once('=')
            .map_or((attribute.trim(), ""), |(key, argument)| {
                (key.trim(), argument.trim())
            });
        if key.eq_ignore_ascii_case("max-age") {
            max_age = argument.parse::<i64>().ok();
        } else if key.eq_ignore_ascii_case("expires") {
            expires = httpdate::parse_http_date(argument).ok();
        } else if key.eq_ignore_ascii_case("secure") {
            secure = true;
        }
    }
    // Max-Age comes before Expires (RFC 6265), and zero or less deletes it.
    let expires = match max_age {
        Some(seconds) => u64::try_from(seconds)
            .ok()
            .filter(|seconds| *seconds > 0)
            .map_or(Some(SystemTime::UNIX_EPOCH), |seconds| {
                now.checked_add(Duration::from_secs(seconds))
            }),
        None => expires,
    };
    Some(RoutingCookie {
        host: String::new(),
        name,
        value: value.trim().to_string(),
        expires,
        secure,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(set_cookies: &[&str]) -> HeaderMap {
        let mut headers = HeaderMap::new();
        for cookie in set_cookies {
            headers.append(SET_COOKIE, HeaderValue::from_str(cookie).unwrap());
        }
        headers
    }

    fn url(text: &str) -> reqwest::Url {
        reqwest::Url::parse(text).unwrap()
    }

    /// Only the two routing cookies are kept, for their host only, Secure
    /// ones only over HTTPS, and a later `Set-Cookie` replaces or deletes one.
    #[test]
    fn routing_cookies_are_kept_per_host_and_sent_back() {
        let backend = url("https://chatgpt.com/backend-api/codex/responses");
        let cookies = RoutingCookies::default();
        cookies.record(
            &backend,
            &response(&[
                "__oailb=pin-1; Path=/; Secure; HttpOnly; Max-Age=3600",
                "__cflb=lb-1; Path=/; Secure; HttpOnly",
                "__Secure-next-auth.session-token=secret; Path=/; Secure",
                "oai-did=device; Path=/",
            ]),
        );
        let header = cookies.header(&backend).unwrap();
        assert!(header.is_sensitive());
        assert_eq!(header, "__oailb=pin-1; __cflb=lb-1");
        assert_eq!(
            cookies.header(&url("https://api.openai.com/v1/responses")),
            None
        );
        assert_eq!(cookies.header(&url("http://chatgpt.com/backend-api")), None);

        cookies.record(
            &backend,
            &response(&["__oailb=pin-2; Path=/; Secure", "__cflb=gone; Max-Age=0"]),
        );
        assert_eq!(cookies.header(&backend).unwrap(), "__oailb=pin-2");
        cookies.record(
            &backend,
            &response(&["__oailb=old; Expires=Thu, 01 Jan 1970 00:00:01 GMT"]),
        );
        assert_eq!(cookies.header(&backend), None);

        // A cookie without Secure goes over plain HTTP too (a local proxy).
        let local = url("http://127.0.0.1:8080/responses");
        cookies.record(&local, &response(&["__oailb=local"]));
        assert_eq!(cookies.header(&local).unwrap(), "__oailb=local");
        assert_eq!(cookies.header(&backend), None);
    }

    #[test]
    fn expiry_comes_from_max_age_before_expires() {
        let now = SystemTime::now();
        let cookie = parse_set_cookie(
            "__cflb=x; Expires=Thu, 01 Jan 1970 00:00:01 GMT; Max-Age=60",
            now,
        )
        .unwrap();
        assert_eq!(cookie.expires, Some(now + Duration::from_secs(60)));
        let cookie =
            parse_set_cookie("__cflb=x; expires=Wed, 21 Oct 2099 07:28:00 GMT", now).unwrap();
        assert!(cookie.is_live(now));
        assert!(parse_set_cookie("__cflb=x; Max-Age=-1", now).is_some_and(|c| !c.is_live(now)));
        assert!(parse_set_cookie("session=x", now).is_none());
        assert!(parse_set_cookie("__oailb", now).is_none());
    }
}
