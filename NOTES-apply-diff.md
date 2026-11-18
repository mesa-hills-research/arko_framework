# apply_diff, the rules the change keeps

Deleted in the branch's last commit.

## The tool (arko_agents)

- All or nothing. Every block applies, or nothing is written. A diff that doesn't parse, a
  block not found, a block found more than once without `replace_all`, and blocks that overlap
  each leave the file byte-identical.
- Every block matches the file as it was before the call, never another block's result, so
  blocks must not overlap. Their order in the diff doesn't matter.
- Matching is exact, on whole lines. No fuzzy score, no search window around a line number,
  no re-indentation, no stripping of copied line numbers.
- An error names each failing block by its number in the diff, says why, and says the file is
  unchanged. Every failing block is reported in one error. A miss gets edit_file's hints
  (copied line numbers, a match apart from indentation) and names the first line that differs.
- Bytes outside the matched lines are kept: a byte-order mark, line breaks, a missing last line
  break. In a CRLF file the search lines match CRLF lines and the replacement lines get CRLF.
- A save never widens who can read a file. It keeps the owner, group, permissions, ACLs and hard
  links, writes through a symbolic link so the link stays a link, and refuses a file the process
  can't write, as an in-place write would. A temporary file renamed over the file is used only
  where it keeps all of that (on Linux, an own file with one link, a settable group and no
  extended attributes on either file), and the file is written in place otherwise. The temporary
  file is private until it has the file's permissions, before any byte is written.
- Anything but a regular file is refused before it is read.
- The output carries an `EditReceipt` covering every block, as edit_file's does.
- The description is a few plain sentences and one short example. Rare cases (a file line that
  is itself a marker) are explained by the error that meets them.
- Blocks in the old format (`:start_line:` and `-------`) still parse, so resumed sessions work.
- Tests check behaviour, not wording.

## Arko

- The main agent and workers are offered `apply_diff`, not `edit_file`. Every instruction,
  code-mode list, outside-access check, transcript row, Quiet label, doc and the CHANGELOG
  follow.
- An apply_diff row names the file and how many blocks applied. Opening it shows the changes
  as edit_file's row does.
- Arko takes the framework branch through a `[patch.crates-io]` git dependency at a full SHA,
  in its own commit marked TEMPORARY.

## Both

- Builds only through `cargo-locked`, one at a time. fmt and clippy `--all-targets -D warnings`
  clean.
- Small commits. Nothing pushed but the framework branch Arko's patch needs.
