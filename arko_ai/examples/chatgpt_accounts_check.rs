//! Live check of failover between ChatGPT accounts ([`ChatGptAccounts`]).
//!
//! ```text
//! cargo run -p arko_ai --example chatgpt_accounts_check -- \
//!     --auth ~/.codex-200/auth.json --auth ~/.codex-100/auth.json \
//!     [--model gpt-6-luna] [--user-agent UA] [--request] [--portability]
//! ```
//!
//! 1. Reads each account's usage (`GET backend-api/wham/usage`) and prints
//!    it: whether requests are allowed, the windows, the credits, the manual
//!    resets left, and whether the list would count the account usable.
//! 2. `--request`: sends one short request through the account list and
//!    prints the account that served it, its plan usage and credits headers,
//!    and the list's events.
//! 3. `--portability`: has the first account answer with reasoning, then
//!    sends the follow-up, that reasoning replayed, straight to the second
//!    account (not through the list, which would leave it out), and prints
//!    whether the backend accepts one account's encrypted reasoning from
//!    another.
//!
//! Each `auth.json` is kept in step with the file (rotated tokens are written
//! back). Accounts are labelled by their file's directory name. Tokens and
//! account ids are never printed.

use arko_ai::{
    ChatGptAccounts, CodexAuthFile, Conversation, GenerationConfig, LLMClient, Model, OpenAIEffort,
    Provider, ReasoningConfig, providers::ProviderClient,
};
use std::path::PathBuf;

struct Args {
    auth: Vec<PathBuf>,
    model: String,
    user_agent: Option<String>,
    request: bool,
    portability: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        auth: Vec::new(),
        model: "gpt-6-luna".into(),
        user_agent: None,
        request: false,
        portability: false,
    };
    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        let mut value = || iter.next().ok_or(format!("{arg} needs a value"));
        match arg.as_str() {
            "--auth" => args.auth.push(PathBuf::from(value()?)),
            "--model" => args.model = value()?,
            "--user-agent" => args.user_agent = Some(value()?),
            "--request" => args.request = true,
            "--portability" => args.portability = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if args.auth.is_empty() || (args.portability && args.auth.len() < 2) {
        return Err("give --auth PATH once per account (two for --portability)".into());
    }
    Ok(args)
}

fn label(path: &std::path::Path) -> String {
    path.parent()
        .and_then(|dir| dir.file_name())
        .map(|name| name.to_string_lossy().trim_start_matches('.').to_string())
        .unwrap_or_else(|| path.display().to_string())
}

#[tokio::main]
async fn main() {
    if let Err(error) = run().await {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let args = parse_args()?;
    let files = args
        .auth
        .iter()
        .map(|path| {
            Ok((
                label(path),
                CodexAuthFile::load(path).map_err(|e| e.to_string())?,
            ))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let accounts = ChatGptAccounts::from_codex_auth_files(
        files.iter().map(|(label, file)| (label.as_str(), file)),
    )
    .map_err(|e| e.to_string())?
    .with_observer(|event| println!("event: {event}"));
    let mut client = LLMClient::new().with_chatgpt_accounts(accounts);
    if let Some(user_agent) = &args.user_agent {
        client = client.with_openai_user_agent(user_agent);
    }
    let accounts = client.chatgpt_accounts().expect("installed above").clone();

    for (label, _) in &files {
        match accounts.read_usage(label).await {
            Ok(status) => println!(
                "{label}: plan {:?}, allowed {:?}, usable {:?}, primary {:?}, secondary {:?}, credits {:?}, resets left {:?}, reached type {:?}",
                status.plan_type,
                status.allowed,
                status.usable(),
                status.primary,
                status.secondary,
                status.credits,
                status.reset_credits_available,
                status.limit_reached_type,
            ),
            Err(error) => println!("{label}: usage could not be read: {error}"),
        }
    }

    let model = Model::get(Provider::OpenAI, &args.model).map_err(|e| e.to_string())?;
    if args.request {
        let mut conversation = Conversation::new(model.clone());
        conversation.add_user_message("Reply with the single word OK.");
        let response = client
            .generate_text(&conversation, GenerationConfig::default())
            .await
            .map_err(|e| e.to_string())?;
        let usage = &response.model_call_usage;
        println!(
            "served by {:?}: plan usage {:?}",
            usage.account, usage.plan_usage
        );
    }

    if args.portability {
        let first = &accounts.accounts()[0];
        let second = &accounts.accounts()[1];
        let config = GenerationConfig::default()
            .with_reasoning(ReasoningConfig::openai(OpenAIEffort::Medium, false));
        let mut conversation = Conversation::new(model);
        conversation.add_user_message("Think it through, then give just the number: 17 * 23 + 4.");
        let answer = first
            .client()
            .generate_text(&conversation, &config)
            .await
            .map_err(|e| format!("{} could not answer: {e}", first.label()))?;
        let reasoning_items = match &answer.message.reasoning_artifact {
            Some(arko_ai::ReasoningArtifact::OpenAIOutputItems { items }) => items
                .iter()
                .filter(|item| item["type"] == "reasoning")
                .count(),
            _ => 0,
        };
        println!(
            "{} answered with {reasoning_items} reasoning item(s)",
            first.label()
        );
        conversation.add_assistant_message_from_response(answer);
        conversation.add_user_message("Now add 1 and give just the number.");
        match second.client().generate_text(&conversation, &config).await {
            Ok(_) => println!(
                "{} accepted {}'s encrypted reasoning",
                second.label(),
                first.label()
            ),
            Err(error) => println!(
                "{} refused {}'s encrypted reasoning: {error}",
                second.label(),
                first.label()
            ),
        }
    }
    Ok(())
}
