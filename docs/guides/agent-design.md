# General Agent Design Guide for Arko Framework

Author: Luke Erickson | Mesa Hills Research
Purpose: To provide a concise, general guide to creating high performance and efficient agents and workflows with arko_framework.

This guide offers ideas, not rules.

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
- Start with the simplest setup that could work, and add tools, subagents, or controls only when it falls short. Aim for autonomy rather than gating the agent by default.
- Prefer terminal over a large set of tools that can be achieved by terminal tools such as `cat`, `grep`, `sed`, `ls`, etc.
- Consider allowing the agent to spawn subagents. Determine what mode of communication should be provided to the top level model and its subagents.

## Prompting

Follow the prompting guide for the model used by each agent. Use the [Opus 5.5 prompting guide](../prompting/opus-5.5.md) for Opus 5.5; it also applies to Fable 5.1. Use the [Astra 6 prompting guide](../prompting/astra-6.md) for Astra 6.

Anything you put in front of an agent steers it. Mentioning a rule or concept, or making a tool available, often leads an agent to use it even when the task doesn't need it. Decide carefully what each prompt includes, leave out what the role doesn't need, and when something is optional, say so and say when it applies.

## Model Choice

Opus 5.5 and GPT-6.1 Sol are the default choices for most work. Both are strong general-purpose models that perform about as well as Astra 6. GPT-6.1 Sol has the lower list price ($2 input and $10 output per million tokens, against $4 and $20 for Opus 5.5), and it is the natural choice with existing OAuth access through a ChatGPT subscription. Use Medium for straightforward tasks that still need more intelligence than Luna.

Fable 5.1 and Astra 6 are also strong choices, but they cost more. They are especially useful as tag-in models when the default model would benefit from a different perspective or an independent pass over the work. Use XHigh reasoning effort by default for Opus 5.5, Fable 5.1, and Astra 6. High is also a good choice and usually makes little practical difference. Max should rarely, if ever, be used with these models: it adds substantial time and cost for little additional benefit.

Astra 6 and Fable 5.1 have complementary strengths in 3D work: Astra 6 is good at 3D design, while Fable 5.1 is good at 3D interaction. Split 3D projects along that line. For a game, you might have Astra 6 work as the 3D modeler and Fable 5.1 as the model that ties everything together.

Use GPT-6 Luna for high-volume, low-intelligence work such as converting unstructured data into structured data. Max reasoning effort is appropriate for Luna.

Sonnet 5.5 really shouldn't be used except for a select few tasks; for general work, use Opus 5.5 or GPT-6.1 Sol. The main exception is a researcher subagent that dives deep into a question and returns a quality analysis to the agent that spawned it. For that role, use High reasoning effort, or XHigh when the question needs it. Avoid Max; when Max would be justified, Opus 5.5 at XHigh is usually the better value.

GPT-6.1 Sol replaces GPT-6 Sol at the same price. GPT-6 Sol costs less per token than Opus 5.5 but about the same per task in practice, and Opus 5.5 clearly outperforms it, so there is no reason to choose it for new work.

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

Offering a tool or parameter is itself a nudge to use it. For example, give agents an optional timeout parameter on a tool and they will almost always set one. Offer only the tools and parameters a role needs, and describe optional ones as optional, with when they're worth using.

Summary:
- Before creating an agent or workflow, consider what tooling is necessary and beneficial to the agent(s).
- Ensure these tools are accessible beforehand, especially in controlled environments that prevent the agents from acquiring them.
- Every tool and parameter you offer steers the agent toward using it. Offer only what the role needs, and phrase optional ones as optional.
