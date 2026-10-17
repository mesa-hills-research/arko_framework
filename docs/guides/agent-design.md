# General Agent Design Guide for Arko Framework

A concise, general guide to creating high-performance, efficient agents and workflows with arko_framework. It offers ideas, not rules.

## Fundamentals

As LLMs have become more capable and better trained, best practices for agent design have shifted. Frontier models used to need a lot of steering and safeguards. The trend has since reversed, and often the simplest and most effective approach is to give an agent terminal access and set it loose. You can still give an agent other tools, but tools the terminal already covers, such as `read_file`, `write_file`, `apply_diff`, and `list_files`, are no longer needed.

Some tools and techniques have survived this shift. `ask_user` is still relevant. It lets an agent ask the user to clarify an ambiguity or choose which path to take. Subagents have not only survived but become more useful. One agent can spawn others with instructions, and can even communicate with them.

Memory and progress tracking are also useful. It is good practice to keep the context window from growing too large, so set a threshold, usually 20–30% of the model's context window. When the model reaches it, the model documents its progress and continues in a fresh conversation. `arko_agents` does this natively, with `AgentConfig::handover` (for example `HandoverConfig::at_context_fraction(0.25)`), and the agent writes a handover report at the threshold, then continues from the task and the report in a fresh conversation, in the same task. Its name, task ID and budget stay the same, so a workflow's messages still reach it. **Do not use hard limits for this. They risk losing progress.**

Summary:
- Start with the simplest setup that could work, and add tools, subagents, or controls only when it falls short. Aim for autonomy rather than gating the agent by default.
- Prefer a terminal over a large set of tools that duplicate terminal commands such as `cat`, `grep`, `sed`, and `ls`.
- Consider letting the agent spawn subagents, and decide how the top-level model and its subagents should communicate.

## Prompting

Follow the prompting guide for each agent's model. Use the [Opus 5.5 prompting guide](../prompting/opus-5.5.md) for Opus 5.5. It also applies to Fable 5.1. Use the [Astra 6 prompting guide](../prompting/astra-6.md) for Astra 6.

Anything you put in front of an agent steers it. Mentioning a rule or concept, or making a tool available, often leads an agent to use it even when the task doesn't need it. Decide carefully what each prompt includes, and leave out what the role doesn't need. When something is optional, say so, and say when it applies.

When an agent works under a policy that can refuse its calls (a command guard, a hook that rejects tool calls, path rules, approvals), tell it how to treat a refusal. The refusal is the policy working, not an obstacle. It shouldn't rephrase, split, or encode a command to get past the check. It should do the work another way, or report the refusal and why the work needed it. A refusal it believes is mistaken is a finding for whoever owns the policy, not something to route around. Agents that meet a refusal otherwise tend to treat it as a puzzle to solve, which hides the policy's gaps instead of reporting them.

## Model Choice

Opus 5.5 and GPT-6.1 Sol are the default choices for most work. Both are strong general-purpose models that perform about as well as Astra 6. GPT-6.1 Sol has the lower list price ($2 input and $10 output per million tokens, against $4 and $20 for Opus 5.5), and it is the natural choice when you already have OAuth access through a ChatGPT subscription. With a Claude subscription, Opus 5.5 can run through the installed Claude Code CLI instead (`Model::claude_code(Model::anthropic_claude_opus_5_5())`). Claude Code runs the agent loop under your login, and the agent keeps its arko tools, approvals, messaging and handovers, with Claude Code's own tools (`Read`, `Edit`, `Bash`, …) available when you list them. Use High or XHigh reasoning effort for GPT-6.1 Sol. Use Medium reasoning effort for straightforward tasks that still need more intelligence than Luna.

Fable 5.1 and Astra 6 are also strong choices, but they cost more. They are especially useful as tag-in models when the default model would benefit from a different perspective or an independent pass over the work. Use XHigh reasoning effort by default for Opus 5.5, Fable 5.1, and Astra 6. High is also a good choice and usually makes little practical difference. Max should rarely, if ever, be used with these models, since it adds substantial time and cost for little additional benefit.

Astra 6 and Fable 5.1 have complementary strengths in 3D work. Astra 6 is good at 3D design, while Fable 5.1 is good at 3D interaction. Split 3D projects along that line. For a game, you might have Astra 6 work as the 3D modeler and Fable 5.1 as the model that ties everything together.

Use GPT-6 Luna for high-volume, low-intelligence work, such as converting unstructured data into structured data. Max reasoning effort is appropriate for Luna.

GPT-6.1 Sol, at High or XHigh, has also largely taken over Sonnet 5.5's one remaining role, a subagent that explores a codebase in depth to answer a question and returns a quality analysis to the agent that spawned it. Otherwise, avoid Sonnet 5.5. For general work, use Opus 5.5 or GPT-6.1 Sol. If you do use Sonnet 5.5 for such a subagent, use High reasoning effort, or XHigh when the question needs it. Avoid Max. When Max would be justified, Opus 5.5 at XHigh is usually the better value.

GPT-6.1 Sol replaces GPT-6 Sol at the same price. GPT-6 Sol costs less per token than Opus 5.5 but about the same per task in practice, and Opus 5.5 clearly outperforms it, so there is no reason to choose it for new work.

Do not restrict a model's output token limit (`model.max_output_tokens`) unless there is a clear reason. Leaving it at the model's maximum is the safe default.

## Inter-agent Communication

`arko_framework` offers several ways for agents to communicate. The best one depends on your workflow's topology, and you can enable more than one.

### Shared Memory

`arko_workflows` has a shared memory system that lets several agents share a document, with configurable access control. This helps agents coordinate, including sharing their progress with one another. You can have several memory documents, each with its own access control. For example, a document can be readable by every agent in the workflow but writable by only some of them. Lower-level memory tiers can be shared among a select few agents, and a document can also be limited to a single agent.

### Messaging Tools

`arko_workflows` has built-in primitives that give agents tools to message each other. Messages are queued until the recipient's next opportunity to receive input. For example, if Agent A messages Agent B while Agent B is calling a tool, the message is delivered once the tool call finishes and a user message can be inserted. Messaging also has access control. For example, you can set up one-way messaging, where Agent A can message Agent B but Agent B cannot message Agent A. A running agent is told when an agent it can message starts or finishes. A message to a finished agent is refused by default, so the sender knows no reply is coming. `MessagingPolicy::keep_messages_for_finished()` queues it for that agent's next run instead, which suits workflows that continue or rerun agents.

### Best Practices

Brainstorm a workflow's general topology before looking at the code in `arko_framework`. Consider the overall flow, which agents should exist, when they should be spawned, and whether and how they should communicate. Then set up the configurations that let them do so. Good inter-agent messaging can reduce or prevent duplicate work, and it speeds up overall progress by letting agents divide the work and share their findings.

Summary:
- Work out a workflow's topology and logical flow before choosing how agents communicate. Only then decide which communication method or methods suit the workflow best.
- Be deliberate with access control. Each setting should have a clear, well-reasoned purpose.

## Tooling

The shift to terminal-first tooling described above doesn't make this section any less relevant, because it applies whether you provide terminal tools, other tools, or both, and it matters most when an agent has a terminal.

Decide which tools the agent needs before you build the workflow. This is especially important in controlled environments, where agents may be unable to reach the network or install their own tools. For example, before writing a coding agent or workflow, consider whether to make tools such as ast-grep or semgrep available. Agents often use them more effectively than plain grep. If the agent will write Rust, make sure Rust is installed.

Offering a tool or parameter is itself a nudge to use it. For example, give agents an optional timeout parameter on a tool, and they will almost always set one. Offer only the tools and parameters a role needs, and describe optional ones as optional, with when they're worth using.

Large files are the one place where reading through the terminal is more than a preference. A data file, a log or a minified bundle can be millions of tokens, more than any model takes in one request. An agent with a terminal should look at it in parts, with `wc -l` for its size, `grep -n` to find what it needs, `head` or `sed -n '100,200p'` for a range, or a short script that prints only what matters. If the role is likely to meet such files, say so in its instructions. `arko_agents` points the agent there too, since `read_file`'s description gains one sentence about it when the agent also has the terminal tools, and its overflow guard (on by default) shortens a turn's results only when they would certainly overflow the model's input limit, with a notice that names those commands. The guard is a safety net, not a strategy, so don't hand-roll a read cap in a hook. Set `ToolResultLimits::max_result_tokens` (or a tool's own `max_result_tokens`) only for a role that should never pull large outputs. `read_file` line ranges are opt-in (`read_file_line_ranges`) for agents without a terminal. Agents offered ranges tend to read in short pieces, so offer them only where they're needed.

Summary:
- Before creating an agent or workflow, consider which tools the agents need or would benefit from.
- Make sure those tools are available beforehand, especially in controlled environments that prevent agents from installing them.
- Every tool and parameter you offer steers the agent toward using it. Offer only what the role needs, and phrase optional ones as optional.
- Steer agents with a terminal to read large files in parts through it. Rely on the overflow guard as the safety net, and add caps only where a role needs a tighter budget.
