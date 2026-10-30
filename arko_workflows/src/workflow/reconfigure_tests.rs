//! A run continued with a new configuration (`ContinueOptions::reconfigure`):
//! the switch between a chat and an agent, in the same conversation.

use std::sync::Arc;

use serde_json::Value;

use super::*;
use crate::persistence::PersistenceConfig;
use crate::test_support::{
    ProbeTool, ResponsesServer, Workspace, agent_config, text_turn, tool_call_turn,
};

/// The agent as a chat (`AgentToolConfig::chat`): offered no tools, its
/// reply in text ends its task.
fn chat() -> AgentConfig {
    let mut config = agent_config("Main");
    config.tool_config.chat = true;
    config.tool_config.available_tool_names.clear();
    config.tool_config.text_reply_ends_task = true;
    config.instructions = "Chat with the user.".into();
    config
}

/// The agent as an agent: offered the probe.
fn agent() -> AgentConfig {
    let mut config = chat();
    config.tool_config.chat = false;
    config.tool_config.available_tool_names = vec!["probe".into()];
    config.instructions = "Work on the user's task.".into();
    config
}

fn builder(server: &ResponsesServer, workspace: &Workspace) -> WorkflowBuilder {
    Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", chat())
        .agent_tool("main", ProbeTool::new("probe", false))
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
}

fn tool_names(body: &Value) -> Vec<String> {
    body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| tool["name"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// A chat continued as an agent: the same run and conversation, with its
/// prompt cache key, now offered its tools (the workflow's added, as for a
/// new run) under a system prompt from the new configuration. As a chat it
/// had none of them. A resume keeps the new configuration, though the
/// agent is registered as a chat.
#[tokio::test]
async fn a_chat_continued_as_an_agent_keeps_its_conversation_and_gets_tools() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let workflow = builder(&server, &workspace).build().unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let mut events = workflow.subscribe_events();
    let second = workflow
        .continue_agent_with(
            &first,
            "Now work on it.",
            ContinueOptions::new().reconfigure(agent()),
        )
        .await
        .unwrap();
    assert_eq!(second.agent_run_id, first.agent_run_id);
    assert!(second.task().unwrap().reconfigured);
    // Reported as the run goes on, before its own events.
    assert!(matches!(
        events.try_recv().unwrap().event,
        arko_agents::AgentEvent::Reconfigured {
            chat: false,
            tools: 1
        }
    ));

    let bodies: Vec<Value> = server
        .bodies()
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect();
    assert!(tool_names(&bodies[0]).is_empty(), "{}", bodies[0]);
    let chat_prompt = bodies[0]["instructions"].as_str().unwrap();
    assert!(chat_prompt.contains("Chat with the user."));
    assert!(!chat_prompt.contains("## Tool Usage"));
    assert_eq!(tool_names(&bodies[1]), ["probe"]);
    let agent_prompt = bodies[1]["instructions"].as_str().unwrap();
    assert!(agent_prompt.contains("Work on the user's task."));
    assert!(agent_prompt.contains("## Tool Usage"));
    assert_eq!(bodies[0]["prompt_cache_key"], bodies[1]["prompt_cache_key"]);
    let sent = bodies[1]["input"].to_string();
    assert!(sent.contains("Hello.") && sent.contains("Sure.") && sent.contains("Now work on it."));

    // Resumed with the agent registered as a chat, the run keeps the
    // configuration it was given, and continues with it.
    workflow.checkpoint().await.unwrap();
    let run = workspace.0.join(".run");
    let latest = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    let resumed = builder(&server, &workspace)
        .resume_from(run.join(latest.trim()))
        .build()
        .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    assert!(restored.task().unwrap().reconfigured);
    resumed.continue_agent(&restored, "And now?").await.unwrap();
    let last: Value = serde_json::from_str(server.bodies().last().unwrap()).unwrap();
    assert_eq!(tool_names(&last), ["probe"]);
}

/// The model stays: another one is refused before anything runs.
#[tokio::test]
async fn a_run_is_reconfigured_with_its_model() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let workflow = builder(&server, &workspace).build().unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    let mut other = agent();
    other.llm_config.model = arko_ai::Model::anthropic_claude_opus_5_5();
    let error = workflow
        .continue_agent_with(&first, "Go on.", ContinueOptions::new().reconfigure(other))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, WorkflowError::BuilderInvalid(message) if message.contains("keeps its model")),
        "{error}"
    );
    assert_eq!(server.requests(), 1);
}

/// Only an agent set as a chat goes without the workflow's tools: one that
/// lists no tools of its own (its reply in text ending its task) still gets
/// the tools registered for it, as before.
#[tokio::test]
async fn only_a_chat_goes_without_the_workflow_s_tools() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let mut quiet = chat();
    quiet.tool_config.chat = false;
    let workflow = Workflow::builder()
        .name("chat-flag-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("quiet", quiet)
        .agent("chat", chat())
        .agent_tool("quiet", ProbeTool::new("probe", false))
        .agent_tool("chat", ProbeTool::new("probe", false))
        .build()
        .unwrap();
    workflow.run_agent("quiet", "Hello.").await.unwrap();
    workflow.run_agent("chat", "Hello.").await.unwrap();
    let bodies: Vec<Value> = server
        .bodies()
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect();
    // The workflow's tools: the one registered for it, and messaging.
    let quiet_tools = tool_names(&bodies[0]);
    assert!(
        quiet_tools.contains(&"probe".to_string()),
        "{quiet_tools:?}"
    );
    assert!(
        quiet_tools.contains(&"send_message".to_string()),
        "{quiet_tools:?}"
    );
    assert!(tool_names(&bodies[1]).is_empty());
}

/// The latest checkpoint of the run directory.
fn latest_checkpoint(workspace: &Workspace) -> std::path::PathBuf {
    let run = workspace.0.join(".run");
    let name = std::fs::read_to_string(run.join("latest-checkpoint")).unwrap();
    run.join(name.trim())
}

/// The system prompt a request body was sent with.
fn instructions(body: &str) -> String {
    serde_json::from_str::<Value>(body).unwrap()["instructions"]
        .as_str()
        .unwrap()
        .to_string()
}

/// With `keep_saved_configs`, an agent registered with another
/// configuration resumes: its saved runs keep theirs (a continued run is
/// sent its own instructions) and its new runs take the new one. Without
/// it, the change refuses the resume.
#[tokio::test]
async fn saved_runs_keep_their_configuration_and_new_runs_take_the_registered_one() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let workflow = builder(&server, &workspace).build().unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let mut brief = chat();
    brief.instructions = "Chat briefly with the user.".into();
    let resume = || {
        Workflow::builder()
            .name("reconfigure-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("main", brief.clone())
            .agent_tool("main", ProbeTool::new("probe", false))
            .persistence(PersistenceConfig::new(workspace.0.join(".run")))
            .resume_from(latest_checkpoint(&workspace))
    };
    let refused = resume().build().err().unwrap().to_string();
    assert!(
        refused.contains("configuration changed since its checkpoint (instructions changed)"),
        "{refused}"
    );
    let unknown = resume()
        .keep_saved_configs(["helper"])
        .build()
        .err()
        .unwrap()
        .to_string();
    assert!(
        unknown.contains("keep_saved_configs names unknown agent 'helper'"),
        "{unknown}"
    );

    let resumed = resume().keep_saved_configs(["main"]).build().unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    resumed.continue_agent(&restored, "And now?").await.unwrap();
    let continued = instructions(server.bodies().last().unwrap());
    assert!(continued.contains("Chat with the user."), "{continued}");
    assert!(!continued.contains("Chat briefly"), "{continued}");
    resumed.run_agent("main", "Start over.").await.unwrap();
    let fresh = instructions(server.bodies().last().unwrap());
    assert!(fresh.contains("Chat briefly with the user."), "{fresh}");
}

/// A saved run kept with its own configuration still needs the tools it
/// offers: a resume without one is refused, naming it.
#[tokio::test]
async fn a_kept_configuration_needs_its_tools() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let workflow = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", agent())
        .agent_tool("main", ProbeTool::new("probe", false))
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let resume = |with_probe: bool| {
        let mut builder = Workflow::builder()
            .name("reconfigure-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("main", chat())
            .persistence(PersistenceConfig::new(workspace.0.join(".run")))
            .resume_from(latest_checkpoint(&workspace))
            .keep_saved_configs(["main"]);
        if with_probe {
            builder = builder.agent_tool("main", ProbeTool::new("probe", false));
        }
        builder.build()
    };
    let refused = resume(false).err().unwrap().to_string();
    assert!(
        refused.contains(&format!(
            "agent 'main' run {} offers tool 'probe', which isn't registered",
            first.agent_run_id
        )),
        "{refused}"
    );
    let resumed = resume(true).unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    resumed.continue_agent(&restored, "And now?").await.unwrap();
    let last: Value = serde_json::from_str(server.bodies().last().unwrap()).unwrap();
    assert!(tool_names(&last).contains(&"probe".to_string()), "{last}");
}

/// A tool registered for saved runs only (`resume_only_agent_tool`) lets a
/// kept run resume. The run is offered it and calls it. A new run, and a run
/// continued with a configuration that doesn't list it, are not offered it,
/// as they are a tool registered as usual.
#[tokio::test]
async fn a_resume_only_tool_serves_a_kept_run_and_is_not_offered_to_others() {
    let workspace = Workspace::new();
    // The run's continuation calls the tool once.
    let server = ResponsesServer::start(|index, _| {
        Some(match index {
            1 => tool_call_turn("c1", "probe", serde_json::json!({})),
            _ => text_turn(&format!("t{index}"), "Sure."),
        })
    })
    .await;
    let workflow = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", agent())
        .agent_tool("main", ProbeTool::new("probe", false))
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    // The agent is registered without the tool in its configuration.
    let mut quiet = chat();
    quiet.tool_config.chat = false;
    let stand_in = ProbeTool::new("probe", false);
    let resume = |probe: Option<Arc<ProbeTool>>| {
        let mut builder = Workflow::builder()
            .name("reconfigure-test")
            .workspace(&workspace.0)
            .llm_client(Arc::new(server.client()))
            .agent("main", quiet.clone())
            .agent_tool("main", ProbeTool::new("other", false))
            .persistence(PersistenceConfig::new(workspace.0.join(".run")))
            .resume_from(latest_checkpoint(&workspace))
            .keep_saved_configs(["main"]);
        if let Some(probe) = probe {
            builder = builder.resume_only_agent_tool("main", probe);
        }
        builder.build()
    };
    let refused = resume(None).err().unwrap().to_string();
    assert!(
        refused.contains("offers tool 'probe', which isn't registered"),
        "{refused}"
    );
    let resumed = resume(Some(stand_in.clone())).unwrap();

    // The kept run is offered the tool and calls it.
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    resumed.continue_agent(&restored, "And now?").await.unwrap();
    let bodies = || -> Vec<Value> {
        server
            .bodies()
            .iter()
            .map(|body| serde_json::from_str(body).unwrap())
            .collect()
    };
    assert!(tool_names(&bodies()[1]).contains(&"probe".to_string()));
    assert_eq!(stand_in.calls(), 1);

    // A new run is offered the tool registered as usual, and not it.
    resumed.run_agent("main", "Fresh.").await.unwrap();
    let new = tool_names(bodies().last().unwrap());
    assert!(new.contains(&"other".to_string()), "{new:?}");
    assert!(!new.contains(&"probe".to_string()), "{new:?}");

    // So is the run continued with a configuration that doesn't list it.
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    resumed
        .continue_agent_with(
            &restored,
            "Again.",
            ContinueOptions::new().reconfigure(quiet.clone()),
        )
        .await
        .unwrap();
    let reconfigured = tool_names(bodies().last().unwrap());
    assert!(
        reconfigured.contains(&"other".to_string()),
        "{reconfigured:?}"
    );
    assert!(
        !reconfigured.contains(&"probe".to_string()),
        "{reconfigured:?}"
    );
    assert_eq!(stand_in.calls(), 1);
}

/// A kept Claude Code run needs its host tools too: only Claude Code's own
/// tools are the CLI's, and a host tool that is gone refuses the resume.
#[tokio::test]
async fn a_kept_claude_code_configuration_needs_its_host_tools() {
    use super::recovery_tests::FakeClaude;
    let fake = FakeClaude::new(serde_json::json!([
        [{"tool": "mcp__arko__complete_task", "input": {"summary": "Done."}}],
    ]));
    let workspace = Workspace::new();
    let mut claude = fake.agent();
    claude.tool_config.available_tool_names = vec!["probe".into(), "Read".into()];
    let workflow = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(arko_ai::LLMClient::new()))
        .agent("main", claude.clone())
        .agent_tool("main", ProbeTool::new("probe", false))
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let mut without_probe = claude.clone();
    without_probe.tool_config.available_tool_names = vec!["Read".into()];
    let refused = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(arko_ai::LLMClient::new()))
        .agent("main", without_probe)
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .resume_from(latest_checkpoint(&workspace))
        .keep_saved_configs(["main"])
        .build()
        .err()
        .unwrap()
        .to_string();
    assert!(
        refused.contains(&format!(
            "agent 'main' run {} offers tool 'probe', which isn't registered",
            first.agent_run_id
        )),
        "{refused}"
    );
}

/// A new run started on a configuration of its own (`RunOptions::config`),
/// here on another model, without rebuilding the workflow: its requests go
/// to that model with that configuration's tools and instructions. The
/// task keeps it, also across a resume from a checkpoint that doesn't keep
/// saved configurations, while new runs take the registered one.
#[tokio::test]
async fn a_new_run_starts_on_another_model_and_keeps_it_across_a_resume() {
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let workflow = builder(&server, &workspace).build().unwrap();
    let sol = arko_ai::Model::openai_gpt_6_1_sol();
    let luna = chat().llm_config.model;
    let mut own = agent();
    own.llm_config.model = sol.clone();
    let first = workflow
        .run_agent_with("main", "Hello.", RunOptions::new().config(own))
        .await
        .unwrap();
    assert!(first.task().unwrap().reconfigured);
    let bodies = || -> Vec<Value> {
        server
            .bodies()
            .iter()
            .map(|body| serde_json::from_str(body).unwrap())
            .collect()
    };
    let started = &bodies()[0];
    assert_eq!(started["model"], sol.id.as_str(), "{started}");
    assert_eq!(tool_names(started), ["probe"]);
    assert!(
        started["instructions"]
            .as_str()
            .unwrap()
            .contains("Work on the user's task.")
    );

    workflow.checkpoint().await.unwrap();
    drop(workflow);
    let resumed = builder(&server, &workspace)
        .resume_from(latest_checkpoint(&workspace))
        .build()
        .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    assert_eq!(restored.info().agent_config.llm_config.model.id, sol.id);
    resumed.continue_agent(&restored, "And now?").await.unwrap();
    let continued = bodies().pop().unwrap();
    assert_eq!(continued["model"], sol.id.as_str(), "{continued}");
    assert_eq!(tool_names(&continued), ["probe"]);
    resumed.run_agent("main", "Start over.").await.unwrap();
    let fresh = bodies().pop().unwrap();
    assert_eq!(fresh["model"], luna.id.as_str(), "{fresh}");
    assert!(tool_names(&fresh).is_empty(), "{fresh}");
}

/// A run started on a Claude Code configuration while its agent is
/// registered with an API model runs in Claude Code, and the other way
/// round. Each resumes from a checkpoint on its own configuration.
#[tokio::test]
async fn a_new_run_moves_between_an_api_model_and_claude_code() {
    use super::recovery_tests::FakeClaude;
    let fake = FakeClaude::new(serde_json::json!([
        [{"tool": "mcp__arko__complete_task", "input": {"summary": "Done in Claude Code."}}],
    ]));
    let workspace = Workspace::new();
    let server =
        ResponsesServer::start(|index, _| Some(text_turn(&format!("t{index}"), "Sure."))).await;
    let workflow = builder(&server, &workspace).build().unwrap();
    let in_claude_code = workflow
        .run_agent_with("main", "Hello.", RunOptions::new().config(fake.agent()))
        .await
        .unwrap();
    assert_eq!(in_claude_code.status, AgentRunStatus::Completed);
    assert_eq!(in_claude_code.summary, "Done in Claude Code.");
    assert_eq!(server.requests(), 0);
    assert_eq!(fake.logged("user").len(), 1);
    workflow.checkpoint().await.unwrap();
    drop(workflow);
    let resumed = builder(&server, &workspace)
        .resume_from(latest_checkpoint(&workspace))
        .build()
        .unwrap();
    let restored = resumed.agent_result(in_claude_code.agent_run_id).unwrap();
    assert_eq!(
        restored.info().agent_config.llm_config.model.provider,
        arko_ai::Provider::ClaudeCode
    );
    drop(resumed);

    // Registered in Claude Code, a run starts on an API model.
    let workspace = Workspace::new();
    let workflow = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", fake.agent())
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .build()
        .unwrap();
    let on_the_api = workflow
        .run_agent_with("main", "Hello.", RunOptions::new().config(chat()))
        .await
        .unwrap();
    assert_eq!(on_the_api.summary, "Sure.");
    assert_eq!(server.requests(), 1);
    assert_eq!(fake.logged("user").len(), 1);
    workflow.checkpoint().await.unwrap();
    drop(workflow);
    let resumed = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", fake.agent())
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .resume_from(latest_checkpoint(&workspace))
        .build()
        .unwrap();
    let restored = resumed.agent_result(on_the_api.agent_run_id).unwrap();
    resumed.continue_agent(&restored, "And now?").await.unwrap();
    assert_eq!(server.requests(), 2);
    assert_eq!(fake.logged("user").len(), 1);
}

/// A handover of a run that keeps the configuration it was saved with
/// (`keep_saved_configs`) starts the fresh conversation on the agent's
/// configuration as registered now: a run saved with an old tool set comes
/// out with the current one, and the current instructions.
#[tokio::test]
async fn a_handover_of_a_saved_run_takes_the_registered_configuration() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, body| {
        Some(if body.contains("# Handover report") {
            text_turn(&format!("t{index}"), "Done.")
        } else if body.contains("Your conversation is getting long") {
            tool_call_turn(
                "report",
                "handover_report",
                serde_json::json!({"report": "REPORT: probed once."}),
            )
        } else if body.contains("Keep going.") {
            // Over the handover size.
            tool_call_turn("p1", "probe", serde_json::json!({}))
                .replace("\"input_tokens\":10", "\"input_tokens\":20000")
        } else {
            text_turn(&format!("t{index}"), "Sure.")
        })
    })
    .await;
    let mut old = agent();
    old.handover = Some(arko_agents::HandoverConfig::at_input_tokens(10_000));
    let workflow = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", old)
        .agent_tool("main", ProbeTool::new("probe", false))
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .build()
        .unwrap();
    let first = workflow.run_agent("main", "Hello.").await.unwrap();
    workflow.checkpoint().await.unwrap();
    drop(workflow);

    let mut current = agent();
    current.handover = Some(arko_agents::HandoverConfig::at_input_tokens(10_000));
    current.tool_config.available_tool_names = vec!["probe".into(), "fresh".into()];
    current.instructions = "Work with the fresh tool.".into();
    let resumed = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", current)
        .agent_tool("main", ProbeTool::new("probe", false))
        .agent_tool("main", ProbeTool::new("fresh", false))
        .persistence(PersistenceConfig::new(workspace.0.join(".run")))
        .resume_from(latest_checkpoint(&workspace))
        .keep_saved_configs(["main"])
        .build()
        .unwrap();
    let restored = resumed.agent_result(first.agent_run_id).unwrap();
    let result = resumed
        .continue_agent(&restored, "Keep going.")
        .await
        .unwrap();
    assert_eq!(result.summary, "Done.");
    let task = result.task().unwrap();
    assert_eq!(task.handovers.len(), 1);
    assert!(!task.reconfigured);
    assert!(
        task.agent_config
            .tool_config
            .available_tool_names
            .contains(&"fresh".to_string())
    );

    let bodies: Vec<Value> = server
        .bodies()
        .iter()
        .map(|body| serde_json::from_str(body).unwrap())
        .collect();
    // Before the handover, the saved configuration.
    let continued = &bodies[1];
    assert!(!tool_names(continued).contains(&"fresh".to_string()));
    assert!(
        continued["instructions"]
            .as_str()
            .unwrap()
            .contains("Work on the user's task.")
    );
    // The fresh conversation, the registered one.
    let fresh = bodies.last().unwrap();
    assert!(fresh.to_string().contains("REPORT: probed once."));
    let tools = tool_names(fresh);
    assert!(
        tools.contains(&"fresh".to_string()) && tools.contains(&"probe".to_string()),
        "{tools:?}"
    );
    let instructions = fresh["instructions"].as_str().unwrap();
    assert!(
        instructions.contains("Work with the fresh tool."),
        "{instructions}"
    );
}

/// A run the host started on a configuration of its own keeps it across a
/// handover.
#[tokio::test]
async fn a_handover_of_a_run_on_the_host_s_configuration_keeps_it() {
    let workspace = Workspace::new();
    let server = ResponsesServer::start(|index, body| {
        Some(if body.contains("# Handover report") {
            text_turn(&format!("t{index}"), "Done.")
        } else if body.contains("Your conversation is getting long") {
            tool_call_turn(
                "report",
                "handover_report",
                serde_json::json!({"report": "REPORT: probed once."}),
            )
        } else {
            // Under the handover size, then over it: a conversation that
            // starts over it doesn't hand over.
            let size = match body.matches("\"function_call_output\"").count() {
                0 => "\"input_tokens\":5000",
                _ => "\"input_tokens\":20000",
            };
            tool_call_turn(&format!("p{index}"), "probe", serde_json::json!({}))
                .replace("\"input_tokens\":10", size)
        })
    })
    .await;
    let mut registered = agent();
    registered.handover = Some(arko_agents::HandoverConfig::at_input_tokens(10_000));
    registered.instructions = "Work as registered.".into();
    let mut own = registered.clone();
    own.instructions = "Work as the host says.".into();
    let workflow = Workflow::builder()
        .name("reconfigure-test")
        .workspace(&workspace.0)
        .llm_client(Arc::new(server.client()))
        .agent("main", registered)
        .agent_tool("main", ProbeTool::new("probe", false))
        .build()
        .unwrap();
    let result = workflow
        .run_agent_with("main", "Hello.", RunOptions::new().config(own))
        .await
        .unwrap();
    assert_eq!(result.task().unwrap().handovers.len(), 1);
    let fresh: Value = serde_json::from_str(server.bodies().last().unwrap()).unwrap();
    assert!(fresh.to_string().contains("REPORT: probed once."));
    let instructions = fresh["instructions"].as_str().unwrap();
    assert!(
        instructions.contains("Work as the host says."),
        "{instructions}"
    );
}
