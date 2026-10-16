use super::oauth_recovery_tests::{api_key_client, completed_response, reply, server};
use super::*;
use crate::model::Model;

/// OpenAI's answer to a request for reasoning summaries from an
/// organization that isn't verified for them.
const UNVERIFIED: &str = r#"{"error":{"message":"Your organization must be verified to generate reasoning summaries. Please go to: https://platform.openai.com/settings/organization/general and click on Verify Organization. If you just verified, it can take up to 15 minutes for access to propagate.","type":"invalid_request_error","param":"reasoning.summary","code":"unsupported_value"}}"#;

fn asking_for_summaries() -> GenerationConfig {
    GenerationConfig {
        reasoning: Some(crate::reasoning::ReasoningConfig::openai(
            OpenAIEffort::High,
            true,
        )),
        ..GenerationConfig::default()
    }
}

fn conversation() -> Conversation {
    let mut conversation =
        Conversation::new(Model::get(ProviderType::OpenAI, "gpt-6-astra").unwrap());
    conversation.add_user_message("hi");
    conversation
}

/// The summary a request asked for, if any.
fn summary(body: &Value) -> Option<&str> {
    body["reasoning"]["summary"].as_str()
}

/// Refused reasoning summaries are dropped: the request goes again without
/// them, its response says so, and the client's later requests ask for
/// none, streamed or not.
#[tokio::test]
async fn refused_reasoning_summaries_are_dropped_for_good() {
    tokio::time::timeout(Duration::from_secs(5), async {
        for streaming in [false, true] {
            let completed = if streaming {
                format!(
                    "data: {}\n\n",
                    serde_json::json!({"type": "response.completed", "response": completed_response()})
                )
            } else {
                completed_response().to_string()
            };
            let (url, server) = server(vec![
                reply("/responses", 400, UNVERIFIED),
                reply("/responses", 200, &completed),
                reply("/responses", 200, &completed),
            ])
            .await;
            let client = api_key_client(&url);
            let generate = |client: OpenAIClient| async move {
                if streaming {
                    let mut stream = client
                        .generate_stream(&conversation(), &asking_for_summaries())
                        .await
                        .unwrap();
                    let mut end = None;
                    while let Some(event) = stream.next().await {
                        if let StreamEvent::End(response) = event.unwrap() {
                            end = Some(*response);
                        }
                    }
                    end.unwrap()
                } else {
                    client
                        .generate_text(&conversation(), &asking_for_summaries())
                        .await
                        .unwrap()
                }
            };
            let first = generate(client.clone()).await;
            assert!(first.model_call_usage.reasoning_summaries_refused, "{streaming}");
            assert_eq!(first.text(), "ok");
            let second = generate(client.clone()).await;
            assert!(!second.model_call_usage.reasoning_summaries_refused);

            let requests = server.await.unwrap();
            let summaries: Vec<Option<&str>> =
                requests.iter().map(|(_, body)| summary(body)).collect();
            assert_eq!(summaries, [Some("auto"), None, None], "{streaming}");
            // The retry is the same request otherwise.
            assert_eq!(requests[0].1["input"], requests[1].1["input"]);
            assert_eq!(requests[1].1["reasoning"]["effort"], "high");
        }
    })
    .await
    .unwrap();
}

/// Another 400 is the request's error, and a request without summaries
/// isn't sent again.
#[tokio::test]
async fn other_refusals_are_not_retried() {
    tokio::time::timeout(Duration::from_secs(5), async {
        let other = r#"{"error":{"message":"Invalid value for 'input'.","type":"invalid_request_error","param":"input","code":"invalid_value"}}"#;
        let (url, server) = server(vec![
            reply("/responses", 400, other),
            reply("/responses", 400, UNVERIFIED),
        ])
        .await;
        let client = api_key_client(&url);
        let error = client
            .generate_text(&conversation(), &asking_for_summaries())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("Invalid value"), "{error}");
        // Without summaries asked for, a refusal naming them is an error too.
        let error = client
            .generate_text(&conversation(), &GenerationConfig::default())
            .await
            .unwrap_err();
        assert!(error.to_string().contains("verified"), "{error}");
        let requests = server.await.unwrap();
        assert_eq!(requests.len(), 2);
        assert!(!client.summaries_refused.load(Ordering::Relaxed));
    })
    .await
    .unwrap();
}

#[test]
fn the_refusal_is_known_by_its_param_or_its_message() {
    assert!(refuses_reasoning_summaries(UNVERIFIED));
    assert!(refuses_reasoning_summaries(
        r#"{"error":{"message":"Your organization must be verified to generate reasoning summaries."}}"#
    ));
    assert!(!refuses_reasoning_summaries(
        r#"{"error":{"message":"Invalid value for 'input'.","param":"input"}}"#
    ));
    assert!(!refuses_reasoning_summaries("not json"));
}
