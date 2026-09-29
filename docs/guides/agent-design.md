# General Agent Design Guide for Arko Framework

Author: Luke Erickson | Mesa Hills Research
Purpose: To provide a concise, general guide to creating high performance and efficient agents and workflows with arko_framework.

Sections:

---

## Fundamentals

As LLMs have gotten more intelligent and effectively trained over time, the best practices for agent design have shifted. Previously, frontier models would often still require a lot of steering, safeguards, etc.
This has shifted in the opposite direction. Often times the most effective and simple approach to designing an agent is to give them terminal access and set them loose. This does not mean that you shouldn't
introduce other tools for the LLM, but there really is no need for the previously necessary tools such as `read_file`, `write_file`, `apply_diff`, `list_files`, etc.

Some tools and techniques have survived this shift. In particular, `ask_user` is still quite relevant. `ask_user` allows agents to reach out to the user to clarify any ambiguities, what path should be taken, etc.
Another technique that has survived and gotten even better are subagents. This is where you give one agent the ability to spawn one or more agents with instructions and even the ability to communicate with them.
Memory and progress tracking are also useful. Since it's generally a good practice to keep the context window from growing too large, you can set a specific limit - usually 20-30% of the model's context window.
Once the model hits or crosses this threshold, you can inject a message telling the model to document their progress. Once it's done, start with a fresh conversation. **Do not use hard limits for this. This risks
losing progress.**

Summary:
- Prefer terminal over a large set of tools that can be achieved by terminal tools such as `cat`, `grep`, `sed`, `ls`, etc.
- Consider allowing the agent to spawn subagents. Determine what mode of communication should be provided to the top level model and its subagents.

## Prompting

Follow the prompting guide for the model used by each agent. Use the [Opus 5.5 prompting guide](../prompting/opus-5.5.md) for Opus 5.5; it also applies to Fable 5.1. Use the [Astra 6 prompting guide](../prompting/astra-6.md) for Astra 6.

## Model Choice

Opus 5.5 is the default choice for most work. It is a strong general-purpose model and usually offers the best balance of capability, speed, and cost. Use Medium for straightforward tasks that still need more intelligence than Luna.

Fable 5.1 and Astra 6 are also strong choices, but they cost more. They are especially useful as tag-in models when Opus 5.5 would benefit from a different perspective or an independent pass over the work. Use XHigh reasoning effort by default for Opus 5.5, Fable 5.1, and Astra 6. High is also a good choice and usually makes little practical difference. Max should rarely, if ever, be used with these models: it adds substantial time and cost for little additional benefit.

Use GPT-6 Luna for high-volume, low-intelligence work such as converting unstructured data into structured data. Max reasoning effort is appropriate for Luna.

Sonnet 5.5 is a capable workhorse and a reasonable alternative for general work. Use Medium when a task needs more intelligence than Luna but does not need a higher effort level. Use High reasoning effort as the normal maximum. XHigh can be appropriate when the task needs it, but avoid Max; when Max would be justified, Opus 5.5 at XHigh is usually the better value.

GPT-6 Sol occupies a less favorable position for most users: it is not yet as capable as Opus 5.5 while costing as much or more. It is reasonable when users want to use existing OAuth access through a ChatGPT subscription; otherwise, Opus 5.5 is the clearer default.

Do not restrict a model's output token limit (`model.max_output_tokens`) unless there is a clear reason. Leave it at the model's maximum as the safe default.

## Inter-agent Communication

With `arko_framework`, there are several ways to enable inter-agent communication. What the best way is depends on the topology of your workflow. You can also enable more than one way.

### Shared Memory

`arko_workflows` has a shared memory system that allows you to share a document across multiple agents with several access control configurations. This helps to coordinate across several agents,
including sharing progress between them. You can also have multiple memory documents each having their own access control configurations. For example, you can have a memory document with global
read, where every agent in the workflow can read it, while only some can write it. You can have lower level memory tiers documents shared between a select few agents. You can also have memory
documents that are accessible by only a single agent.

### Messaging Tools

`arko_workflows` has built in primitives for providing agents with tools to message each other. Tools are queued for the next user input opportunity. For example, if Agent A messages Agent B
while Agent B is currently calling a tool, the message will be delivered after the tool call is executed and a user message can be inserted. This tool also has access control configurations.
For example, you can have one way messaging where Agent A can message Agent B, but Agent B cannot message Agent A.

### Best Practices

Brainstorm a workflow's general topology before looking at the code in `arko_framework`. Consider the overall flow, what agents should exist, when they should be spawned, and if/how they
should be able to communicate. Then, set up the appropriate configurations for them to do so. Proper inter-agent messaging can help reduce or prevent duplicate work, speed up overall progress
by allowing agents to divide work and share their findings, etc.

Summary:
- Brainstorm a workflow's topology and logical flow before committing to a specific inter-agent messaging protocol. Only once you have determined that, then consider which inter-agent messaging
  protocol(s) suit your workflow best.
- Be diligent with access control configurations. Each one should have a purpose behind it backed by solid reasoning.

## Tooling

While we previously discussed the overall shift to terminal-first tooling, this section still applies regardless of if you provide non-terminal tools or terminal tools. This section is especially
important when a terminal tool is provided to an agent.

It is very important to brainstorm what tooling should be available for the agent to access before the workflow. This is especially true in controlled environments where agents may be restricted
from network access or installing their own tooling. For example, before writing a coding agent or coding workflow, consider if tools such as ast-grep, semgrep, etc. should be made available to
the agent. Often times agents can use these tools more effectively than pure grep. If you're having the agent write rust, ensure that rust is installed.

Summary:
- Before creating an agent or workflow, consider what tooling is necessary and beneficial to the agent(s).
- Ensure these tools are accessible beforehand, especially in controlled environments that prevent the agents from acquiring them.
