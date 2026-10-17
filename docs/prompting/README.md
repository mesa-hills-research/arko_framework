# Writing prompts for framework agents

An agent's prompt should describe its job and the result you need. Put the current
request in the task, and keep reusable role instructions in the agent configuration.
See the [agent guide](../arko_agents-guide.md) for configuration and task examples.

Choose tools and permissions in the configuration rather than relying on a prompt
to enforce them. The [defaults reference](../defaults-reference.md) explains approval,
recovery, tool-result limits and task completion.

For a workflow, tell each agent which part of the job it owns and what its report
should contain. Use the workflow's messaging and completion APIs to coordinate runs.
See the [workflow guide](../arko_workflows-guide.md) for those APIs and the
[agent-design guide](../guides/agent-design.md) for choosing roles and models.

When a prompt needs changing, try it on representative tasks and inspect both the
result and the tool calls. Keep instructions that solve a concrete problem in those
runs.
