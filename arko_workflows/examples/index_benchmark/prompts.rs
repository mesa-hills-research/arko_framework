//! The benchmark's prompts. `{placeholders}` are filled by the functions at
//! the end of this file.
//!
//! The role markers (`[index-benchmark role: ...]`) at the start of each
//! role description tell the mock model which role it plays; a real model
//! reads them as a heading.

/// The solver's role, the same in every arm.
pub const SOLVER_ROLE: &str = "[index-benchmark role: solver]
You are an expert software engineer. You answer questions about unfamiliar codebases \
precisely, from the code itself, citing exact file paths and line numbers.";

/// The solver's instructions. `{tools}` holds the arm's tool section.
pub const SOLVER_INSTRUCTIONS: &str = r#"
You answer one question about the repository checked out in your working directory. The
checkout is read-only.

Use the `shell` tool to explore it: list files, search, and read code. {tools}

Rules:
- Base every claim on code you have read in this checkout, not on what you remember of the
  project: this is one exact version, and details differ between versions.
- Don't modify files, build the project, run its tests, or use the network.
- Paths are relative to the repository root. Line numbers are 1-based, as `rg -n`, `grep -n`
  and `sed -n` print them.
- Be complete where the question asks for a list (every caller, every place affected), and
  don't pad the answer with guesses: a wrong item counts against it.
- When you have the answer, call `complete_task` with it in the format below. Its summary is
  your whole answer, and only its content is judged: say what the answer is, not how you found
  it.

Answer format (the `summary` of `complete_task`):

ANSWER:
<the answer: complete and concise. For "how does it work", the mechanism step by step, naming
the functions and files involved.>
LOCATIONS:
<one location per line, `path:line` or `path:start-end`, most important first: the
definitions, call sites or code the answer relies on.>
"#;

/// The tool section of arm A (terminal only).
pub const TOOLS_TERMINAL: &str = "The usual command-line tools are installed{rg}.";

/// The tool section of arm B (terminal and arko_index).
pub const TOOLS_INDEX: &str = r#"The usual command-line tools are installed{rg}, and also
`arko_index`, a code search engine with a prebuilt index of this checkout:
- `arko_index search2 "$ARKO_INDEX_STATE" 'words or identifiers'` prints the 10 code chunks
  (functions, methods, types, ...) that best match the query, best first, one JSON line each,
  with `path`, `start_line`, `end_line`, `kind` and `symbol_path`. It ranks with BM25 over words
  and identifier parts (snake_case and camelCase are split), so a description such as
  'retry backoff delay' finds code that names those things. It prints locations, not code:
  read the lines with `sed -n`.
- `arko_index outline PATH` lists the chunks of a file, or of every file under a directory,
  as `path:start-end<TAB>kind<TAB>symbol` (paths relative to PATH): a quick table of contents."#;

/// What arm C adds to arm B's section (dense search).
pub const TOOLS_DENSE: &str = r#"
- `arko_index_dense 'a description of the code'` ranks the same chunks by meaning
  (embeddings) and prints the 10 best in the same JSON format, so it finds code that does what
  you describe even when it uses other words."#;

/// The answer guidance for each kind of question.
pub fn kind_guidance(kind: &str) -> &'static str {
    match kind {
        "definition" => {
            "Give the location of the definition (the line where it is declared, or its whole range) first under LOCATIONS, and say in ANSWER what it is and where."
        }
        "callers" => {
            "List every call site under LOCATIONS (the line of each call), and name the calling function of each in ANSWER. Count only call sites within the scope the question gives."
        }
        "behaviour" => {
            "Explain in ANSWER how it works, step by step, naming the functions and files involved; give the key definitions under LOCATIONS."
        }
        "impact" => {
            "List in ANSWER every place that would have to change or would behave differently, and why; give each place under LOCATIONS."
        }
        _ => "",
    }
}

/// The grader's role.
pub const GRADER_ROLE: &str = "[index-benchmark role: grader]
You grade answers to questions about a codebase against a verified answer key, strictly and \
fairly.";

pub const GRADER_INSTRUCTIONS: &str = r#"
You get a question about the repository in your working directory (a read-only checkout), its
answer key, and a candidate answer. Decide which of the key's required items the candidate
answer gets right, and which of its claims are wrong.

- The key was verified against this checkout. The candidate may still be right where its
  wording differs: judge meaning, not wording. A location a few lines off that is clearly the
  same code (the same function, the same call) matches.
- An item is met only when the answer states it, or something equivalent, and doesn't
  contradict it. A vague answer doesn't meet a specific item. A location item is met when the
  answer identifies that code (by a location, or by naming it unambiguously).
- Wrong claims are statements the code contradicts, or for a list question (callers, places
  affected) items listed that don't belong in the list. Extra details that are true are not
  wrong. A "known wrong answer" in the key given as the answer is a wrong claim.
- Use the `shell` tool to look at the code only when you need to: when the answer makes a claim
  the key doesn't settle, check it in the code before you call it wrong.
- How the answer was found doesn't matter, only what it says.

Verdict: "correct" when every item is met (or the misses are trivial) and there is no material
wrong claim; "partially_correct" when some items are met; otherwise "incorrect".

Then call `complete_task` with exactly one JSON object as its summary, and nothing else:
{"items": [{"id": "<item id>", "met": true, "note": "<short reason>"}],
 "wrong_claims": ["<each wrong claim, briefly>"],
 "verdict": "correct" | "partially_correct" | "incorrect",
 "notes": "<one or two sentences>"}
Every item of the key appears exactly once in "items".
"#;

/// The question writer's role.
pub const WRITER_ROLE: &str = "[index-benchmark role: writer]
You write benchmark questions about codebases, each with an answer key you have verified in \
the code.";

pub const WRITER_INSTRUCTIONS: &str = r#"
You write questions about the repository checked out in your working directory (read-only), for
a benchmark of coding agents that answer such questions by exploring the code. Each question
needs an answer key that you have checked in the code: an agent's answer will be graded
against it.

Use the `shell` tool to explore (`rg`, `grep`, `find`, `sed -n`, `git grep`{index}). Submit
each question with `submit_question` as soon as it is ready; it checks the key's locations
against the checkout and tells you what to fix. When you have submitted the number of
questions asked for, call `complete_task` with a one-line summary.

What makes a good question:
- A real question a developer new to this codebase would ask, with one clear answer that the
  code settles. No trivia about comments, formatting, licences or build files.
- Spread the questions over different parts of the repository (different directories,
  subsystems and features), and vary the difficulty.
- Describe the thing by what it does at least as often as by its name, so that answering takes
  understanding and not one search for a name given in the question. Never give away the file
  or line in the question.
- Keys must be exact and complete. For a list (callers, places affected), find every member:
  search for each name in every form it can be called or referenced (methods with the same
  name on other types don't count; say so in the scope when that matters), and keep lists
  short: questions whose complete answer has at most 12 entries.
- Scope: say what counts when it matters, for example "in non-test code" or "under src/".

The kinds of question:
- definition: where something is defined: a function, method, type, constant, or a field or
  variant of a type. Key: the definition's location (its declaration line to its end).
  Example: "Where is the function that decides how long to wait before retrying a failed
  request defined?"
- callers: who calls a function or method. Key: every call site in scope, each the line of the
  call, described by the calling function. Example: "Which functions call `flush_pending_writes`
  outside the tests?"
- behaviour: how a feature or mechanism works. Key: 3 to 6 facts an answer must state (each
  checkable, specific: names, conditions, order of steps, values) and the 1 to 4 functions
  or files that implement it, as locations. Example: "How does the client decide when an
  expired access token must be refreshed, and what happens when the refresh fails?"
- impact: what a change would affect. Key: the places that would have to change or would
  behave differently (as locations), and 1 to 4 facts about why. Example: "If `Options::max_bytes`
  were allowed to be 0, which code paths would break and how?"

`submit_question` takes the question and its key:
- `locations`: one per line, `path:start-end | what it is` (or `path:line | ...`), paths
  relative to the repository root. Definitions: the definition. Callers: each call site.
  Behaviour and impact: the functions or places in the key.
- `facts`: one per line (behaviour and impact; optional for the others).
- `answer`: the reference answer in prose, as a complete answer would give it.
- `evidence`: the commands and lines you checked the key with.
"#;

/// The verifier's role.
pub const VERIFIER_ROLE: &str = "[index-benchmark role: verifier]
You check benchmark questions about codebases: you answer each one yourself from the code, then \
compare your answer with the proposed answer key.";

pub const VERIFIER_INSTRUCTIONS: &str = r#"
You check one benchmark question about the repository in your working directory (a read-only
checkout). First you answer it yourself from the code; then you get the answer key another
engineer wrote, and you decide whether the key is right.

Use the `shell` tool to explore (`rg`, `grep`, `find`, `sed -n`, `git grep`{index}). Base every
claim on code you have read in this checkout. Don't modify files, build, or use the network.

First message: answer the question. Call `complete_task` with your answer as its summary:
ANSWER:
<the answer>
LOCATIONS:
<one `path:line` or `path:start-end` per line>

Second message: you get the answer key. Compare it with your answer and check every difference
in the code: each location and fact of the key, and for list questions (callers, places
affected) search again for entries either of you missed. Then call `complete_task` with exactly
one JSON object as its summary, and nothing else:
{"verdict": "agree" | "disagree" | "ambiguous",
 "problems": ["<each error or omission in the key, with its location>"],
 "notes": "<one or two sentences>"}
- "agree": the key is correct and complete for the question as asked (wording aside).
- "disagree": the key has an error or omission, even after you rechecked your own answer.
- "ambiguous": the question allows more than one reasonable answer, or its scope is unclear.
"#;

/// The first message to the verifier: the question, as a solver gets it.
pub fn verifier_question(repo: &str, kind: &str, question: &str, scope: Option<&str>) -> String {
    format!(
        "Repository: {repo}\nQuestion ({kind}): {question}\n{}{}",
        scope.map_or(String::new(), |scope| format!("Scope: {scope}\n")),
        kind_guidance(kind)
    )
}

/// The second message to the verifier: the key.
pub fn verifier_key(key_text: &str) -> String {
    format!(
        "ANSWER KEY to check (written by another engineer):\n{key_text}\n\nCompare it with your answer, check each difference in the code, and give your verdict as the JSON object described in your instructions."
    )
}

/// The writer's task prompt.
pub fn writer_prompt(
    repo: &str,
    description: &str,
    language: &str,
    kind: &str,
    count: usize,
    avoid: &[String],
) -> String {
    let avoid = if avoid.is_empty() {
        String::new()
    } else {
        format!(
            "\nThese questions exist already; don't repeat them or ask about the same code:\n{}\n",
            avoid
                .iter()
                .map(|question| format!("- {question}"))
                .collect::<Vec<_>>()
                .join("\n")
        )
    };
    format!(
        "Repository: {repo} ({description}; mainly {language}).\n\nWrite {count} questions of kind `{kind}` and submit each with `submit_question` (kind `{kind}`).\n{avoid}"
    )
}
