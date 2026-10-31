//! Keeping a copy of a running task up to date without cloning its images
//! again at every step.

use super::Task;

/// `copy` made equal to `source`, kept as it is when it already is.
fn same<T: Clone + PartialEq>(copy: &mut T, source: &T) {
    if copy != source {
        copy.clone_from(source);
    }
}

impl Task {
    /// Makes `copy` equal to this task, as `*copy = self.clone()` does, but
    /// keeps what `copy` holds of its messages and images when it is the
    /// same, instead of cloning it again. A host that saves a copy of the
    /// task at each step (the workflow's task journal) refreshes the same
    /// copy each time. That costs the step's new messages and a comparison
    /// of the old ones, which reads their images but doesn't copy them.
    pub fn refresh_copy(&self, copy: &mut Task) {
        // Every field is named, so that a new one has to be copied here too.
        let Task {
            id,
            agent_config_id,
            agent_config,
            initial_task_prompt,
            initial_task_images,
            conversation,
            current_state,
            start_time,
            end_time,
            developer_env_details,
            usage_history,
            model_call_usage_history,
            current_working_directory,
            user_message_count,
            pending_native_tool_results,
            pending_native_tool_calls,
            pending_tool_images,
            pending_text_tool_results,
            deferred_tool_errors,
            refused_input,
            runtime_messages,
            runtime_text_messages,
            fallback_for,
            fallback_calls,
            plan_usage_threshold_reported,
            monitor_pause,
            pending_context_injections,
            next_context_item,
            context_records,
            context_records_from,
            initial_prompt_ids,
            pending_context_images,
            unavailable_images,
            handover_requested,
            handovers,
            conversation_usage_start,
            conversation_model_call_start,
            context_size_unknown,
            latest_request,
            cache_window_start,
            claude_code_session,
            claude_code_directory,
            claude_code_synced,
            claude_code_prompt_changed,
            claude_code_entries,
            claude_code_resume_at,
            claude_code_fork_of,
            reconfigured,
            running_tool_calls,
            parent,
            questions,
            context_marks,
        } = self;
        copy.id = *id;
        copy.agent_config_id.clone_from(agent_config_id);
        copy.agent_config.clone_from(agent_config);
        copy.initial_task_prompt.clone_from(initial_task_prompt);
        same(&mut copy.initial_task_images, initial_task_images);
        copy.conversation.refresh_from(conversation);
        copy.current_state.clone_from(current_state);
        copy.start_time = *start_time;
        copy.end_time = *end_time;
        copy.developer_env_details.clone_from(developer_env_details);
        copy.usage_history.clone_from(usage_history);
        copy.model_call_usage_history
            .clone_from(model_call_usage_history);
        copy.current_working_directory
            .clone_from(current_working_directory);
        copy.user_message_count = *user_message_count;
        same(
            &mut copy.pending_native_tool_results,
            pending_native_tool_results,
        );
        copy.pending_native_tool_calls
            .clone_from(pending_native_tool_calls);
        same(&mut copy.pending_tool_images, pending_tool_images);
        copy.pending_text_tool_results
            .clone_from(pending_text_tool_results);
        copy.deferred_tool_errors.clone_from(deferred_tool_errors);
        copy.refused_input.clone_from(refused_input);
        copy.runtime_messages.clone_from(runtime_messages);
        copy.runtime_text_messages.clone_from(runtime_text_messages);
        copy.fallback_for.clone_from(fallback_for);
        copy.fallback_calls.clone_from(fallback_calls);
        copy.plan_usage_threshold_reported
            .clone_from(plan_usage_threshold_reported);
        copy.monitor_pause = *monitor_pause;
        copy.pending_context_injections
            .clone_from(pending_context_injections);
        copy.next_context_item = *next_context_item;
        copy.context_records.clone_from(context_records);
        copy.context_records_from = *context_records_from;
        copy.initial_prompt_ids.clone_from(initial_prompt_ids);
        same(&mut copy.pending_context_images, pending_context_images);
        copy.unavailable_images.clone_from(unavailable_images);
        copy.handover_requested.clone_from(handover_requested);
        copy.handovers.clone_from(handovers);
        copy.conversation_usage_start = *conversation_usage_start;
        copy.conversation_model_call_start = *conversation_model_call_start;
        copy.context_size_unknown = *context_size_unknown;
        copy.latest_request.clone_from(latest_request);
        copy.cache_window_start = *cache_window_start;
        copy.claude_code_session.clone_from(claude_code_session);
        copy.claude_code_directory.clone_from(claude_code_directory);
        copy.claude_code_synced = *claude_code_synced;
        copy.claude_code_prompt_changed = *claude_code_prompt_changed;
        copy.claude_code_entries.clone_from(claude_code_entries);
        copy.claude_code_resume_at.clone_from(claude_code_resume_at);
        copy.claude_code_fork_of.clone_from(claude_code_fork_of);
        copy.reconfigured = *reconfigured;
        same(&mut copy.running_tool_calls, running_tool_calls);
        copy.parent = *parent;
        copy.questions.clone_from(questions);
        copy.context_marks.clone_from(context_marks);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent_config::{AgentConfig, AgentLLMConfig};
    use crate::handover::HandoverRecord;
    use arko_ai::{Content, Image, Message, ToolCall, ToolResult};

    fn task() -> Task {
        let config = AgentConfig::builder(
            "worker".into(),
            "Worker".into(),
            AgentLLMConfig {
                model: arko_ai::Model::anthropic_claude_opus_5_5(),
                generation_config: Default::default(),
            },
        )
        .role_description("r".into())
        .instructions("i".into())
        .build()
        .unwrap();
        Task::new(config, "Look.".into(), "/w".into(), None)
    }

    fn image(fill: u8) -> Image {
        Image::from_bytes(&vec![fill; 4096], "image/png")
    }

    /// A user message with a screenshot, then an assistant reply, the shape
    /// of a step.
    fn step(task: &mut Task, fill: u8) {
        task.conversation
            .messages
            .push(Message::new_user_with_contents(vec![
                Content::Text(format!("Screenshot {fill}.")),
                Content::Image(image(fill)),
            ]));
        task.conversation
            .add_assistant_message(&format!("Seen {fill}."));
    }

    /// Everything a clone would hold, compared as saved and as printed.
    fn assert_same(copy: &Task, source: &Task) {
        let cloned = source.clone();
        assert_eq!(
            serde_json::to_value(copy).unwrap(),
            serde_json::to_value(&cloned).unwrap()
        );
        assert_eq!(format!("{copy:?}"), format!("{cloned:?}"));
    }

    /// Refreshes `copy` from `source` and checks it against a clone.
    fn refresh(copy: &mut Task, source: &Task) {
        source.refresh_copy(copy);
        assert_same(copy, source);
    }

    /// The address of each image's data in the task's messages.
    fn image_addresses(task: &Task) -> Vec<*const u8> {
        task.conversation
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .filter_map(|content| match content {
                Content::Image(Image::Base64 { data, .. }) => Some(data.as_ptr()),
                _ => None,
            })
            .collect()
    }

    /// A copy refreshed after each step equals a clone, and the images it
    /// held already are not copied again.
    #[test]
    fn a_refreshed_copy_equals_a_clone_and_keeps_the_images_it_has() {
        let mut task = task();
        step(&mut task, 1);
        let mut copy = task.clone();
        assert_same(&copy, &task);
        let held = image_addresses(&copy);
        for fill in 2..6 {
            step(&mut task, fill);
            task.usage_history.push(Default::default());
            task.pending_text_tool_results
                .push(format!("result {fill}"));
            refresh(&mut copy, &task);
            assert_eq!(image_addresses(&copy)[..held.len()], held[..]);
        }
        assert_eq!(image_addresses(&copy).len(), 5);
    }

    /// Whatever changes in the old messages shows in the copy: an edit in
    /// the text, an image replaced by another of the same size, one added
    /// and one removed.
    #[test]
    fn a_refreshed_copy_follows_edits_to_old_messages_and_images() {
        let mut task = task();
        for fill in 1..5 {
            step(&mut task, fill);
        }
        let mut copy = task.clone();

        task.conversation.messages[0].content[0] = Content::Text("Edited.".into());
        refresh(&mut copy, &task);

        task.conversation.messages[2].content[1] = Content::Image(image(9));
        refresh(&mut copy, &task);
        let Content::Image(Image::Base64 { data, .. }) = &copy.conversation.messages[2].content[1]
        else {
            panic!("no image");
        };
        assert_eq!(&**data, image_data(9));

        task.conversation.messages[4]
            .content
            .push(Content::Image(image(7)));
        refresh(&mut copy, &task);

        task.conversation.messages[4].content.truncate(2);
        refresh(&mut copy, &task);
        task.conversation.messages[6].content.remove(1);
        refresh(&mut copy, &task);

        // A tool result with an image, a pending image, and the task's own.
        let call = ToolCall {
            id: Some("c1".into()),
            name: "view".into(),
            arguments: serde_json::json!({}),
            thought_signature: None,
            provider_id: None,
            invalid_arguments: None,
        };
        let mut result = ToolResult::from_tool_call(&call, "A picture.");
        result.images = vec![arko_ai::ToolResultImage::new("a.png", image(3))];
        task.conversation.add_tool_results(vec![result]);
        task.initial_task_images = vec![image(4)];
        task.pending_tool_images = vec![crate::tool::ToolImage::base64(
            "b.png",
            image_data(5),
            "image/png",
        )];
        task.pending_context_images = vec![crate::runtime::ContextImage::new(
            "Image of note 1:",
            image(6),
        )];
        refresh(&mut copy, &task);

        task.initial_task_images.clear();
        task.pending_tool_images.clear();
        task.pending_context_images.clear();
        refresh(&mut copy, &task);
    }

    fn image_data(fill: u8) -> String {
        match image(fill) {
            Image::Base64 { data, .. } => data.to_string(),
            Image::Url(_) => unreachable!(),
        }
    }

    /// A cut back, a task moved to a Claude Code session, a handover that
    /// replaces the conversation, and a copy of another task altogether.
    #[test]
    fn a_refreshed_copy_follows_cut_backs_handovers_and_other_tasks() {
        let mut task = task();
        for fill in 1..4 {
            step(&mut task, fill);
        }
        let mut copy = task.clone();

        let turns = task.user_turns();
        task.cut_back(turns[1].index).unwrap();
        refresh(&mut copy, &task);
        step(&mut task, 8);
        refresh(&mut copy, &task);

        // A handover: a conversation of its own, with the report kept.
        task.handovers.push(HandoverRecord {
            at: chrono::Utc::now(),
            input_tokens: 1000,
            replaced_messages: task.conversation.messages.len(),
            report: "Report.".into(),
        });
        task.conversation.messages.clear();
        task.conversation.system_message = Some("New prompt.".into());
        task.conversation.add_user_message("Carry on.");
        step(&mut task, 9);
        task.conversation_usage_start = task.usage_history.len();
        refresh(&mut copy, &task);

        task.claude_code_session = Some("session".into());
        task.move_to("/elsewhere".into());
        refresh(&mut copy, &task);

        let mut other = self::task();
        step(&mut other, 5);
        other.id = uuid::Uuid::new_v4();
        refresh(&mut copy, &other);

        // Back to an empty copy and a task without messages.
        let empty = self::task();
        refresh(&mut copy, &empty);
        let mut fresh = self::task();
        refresh(&mut fresh, &task);
    }
}
