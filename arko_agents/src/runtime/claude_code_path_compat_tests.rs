use super::*;

#[tokio::test]
async fn a_non_string_path_reaches_the_arko_tool_for_validation() {
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__read_file", "input": {"path": 13}},
        complete("Done")
    ]]));
    let mut cfg = config(&fake, &["read_file"]);
    cfg.tool_config.default_out_of_workspace_path_policy = PathSafetyPolicy::AutoReject;
    let task = Task::new(cfg, "Read the example.".into(), fake.workspace(), None);
    let (task, events) = start(task).finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
    assert!(
        events.iter().any(
            |event| matches!(event, AgentEvent::ErrorEncountered { error_message, .. }
        if error_message.contains("Parameter 'path' must be a string"))
        ),
        "{events:#?}"
    );
    assert!(
        fake.logged("post")[0]["output"]
            .as_str()
            .unwrap()
            .contains("must be a string")
    );
}

#[tokio::test]
async fn an_outside_path_requires_approval_with_the_existing_reason() {
    let outside = std::env::temp_dir().join(format!("example-{}.txt", uuid::Uuid::new_v4()));
    let fake = Fake::new(json!([[
        {"tool": "mcp__arko__read_file", "input": {"path": outside}},
        complete("Done")
    ]]));
    let mut cfg = config(&fake, &["read_file"]);
    cfg.tool_config.default_out_of_workspace_path_policy = PathSafetyPolicy::RequireApproval;
    let task = Task::new(cfg, "Read the example.".into(), fake.workspace(), None);
    let mut run = start(task);
    let events = run
        .events_until(|event| matches!(event, AgentEvent::ToolCallAwaitingApproval { .. }))
        .await;
    let Some(AgentEvent::ToolCallAwaitingApproval {
        tool_call_id,
        approval_reason,
        ..
    }) = events.last()
    else {
        panic!("{events:#?}")
    };
    assert_eq!(approval_reason.path_outside_workspace, None);
    assert!(!approval_reason.tool_not_auto_approved);
    run.commands
        .send(UserCommand::RejectToolUse {
            tool_call_id: *tool_call_id,
            reason: None,
        })
        .await
        .unwrap();
    let (task, _) = run.finish().await;
    assert!(matches!(task.current_state, TaskState::Completed { .. }));
}
