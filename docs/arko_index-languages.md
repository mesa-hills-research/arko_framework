# arko_index languages

arko_index parses every file language that [mhr_tree_sitter](arko_index-reference.md#parsing)
has a grammar for and chunks it along that grammar's structure. This page lists which
files go to which grammar, how the rules tables work, what each language's rules are,
how Markdown's front matter is handled, and how the rules are checked.

Fifty-two grammars parse files. The port's other two have no file type of their own:
`markdown_inline` (Markdown's inline syntax, which chunk boundaries never need) and
`php_only` (PHP without the surrounding HTML, while `.php` files are parsed with `php`).

## Which grammar parses a file

The extension is lowercased first, except that only a lowercase `.tsx` gets the TSX grammar
and only a lowercase `.h` the C++ grammar (`.TSX` parses as TypeScript, `.H` as C).
Well-known file names match exactly, case included.

| Language (`language` field) | Extensions | File names |
|---|---|---|
| bash | `sh`, `bash`, `zsh` | `.bashrc`, `.bash_profile`, `.bash_login`, `.bash_logout`, `.profile`, `.zshrc`, `.zshenv`, `.zprofile`, `PKGBUILD`, `APKBUILD` |
| c | `c`, `h` (`.h` parsed with the C++ grammar) | |
| c_sharp | `cs`, `csx` | |
| cmake | `cmake` | `CMakeLists.txt` |
| cpp | `cc`, `cpp`, `hpp`, `cxx`, `hh`, `hxx`, `ipp`, `tpp` | |
| css | `css` | |
| cuda | `cu`, `cuh` | |
| dart | `dart` | |
| dtd | `dtd` | |
| elixir | `ex`, `exs` | |
| elm | `elm` | |
| embedded_template | `erb`, `ejs` | |
| erlang | `erl`, `hrl`, `escript` | |
| fortran | `f90`, `f95`, `f03`, `f08` | |
| fsharp | `fs`, `fsx` | |
| glsl | `glsl`, `vert`, `frag`, `geom`, `comp`, `tesc`, `tese`, `vsh`, `fsh` | |
| go | `go` | |
| haskell | `hs` | |
| hcl | `hcl`, `tf`, `tfvars` | |
| html | `html`, `htm` | |
| java | `java` | |
| javascript | `js`, `jsx`, `mjs`, `cjs` | |
| json | `json`, `jsonc`, `geojson`, `webmanifest` | |
| julia | `jl` | |
| kotlin | `kt`, `kts` | |
| lua | `lua` | |
| make | `mk`, `mak`, `make` | `Makefile`, `makefile`, `GNUmakefile` |
| markdown | `md`, `markdown`, `mdx` | |
| objc | `m` | |
| ocaml | `ml`, and `mli` (parsed with the OCaml interface grammar) | |
| odin | `odin` | |
| perl | `pm`, and `pl` and `t` when they look like Perl | |
| php | `php`, `phtml` | |
| powershell | `ps1`, `psm1`, `psd1` | |
| proto | `proto` | |
| python | `py`, `pyi`, `pyw` | |
| r | `r` | |
| ruby | `rb`, `rake`, `gemspec`, `ru` | `Gemfile`, `Rakefile`, `Podfile`, `Vagrantfile`, `Brewfile` |
| rust | `rs` | |
| scala | `scala`, `sc`, `sbt` | |
| solidity | `sol` | |
| sql | `sql` | |
| svelte | `svelte` | |
| swift | `swift` | |
| toml | `toml` | `Cargo.lock`, `Pipfile` |
| typescript | `ts`, `tsx` (TSX grammar for `.tsx`), `mts`, `cts` | |
| verilog | `v`, `vh`, `sv`, `svh` | |
| xml | `xml`, `svg`, `xsd`, `xsl`, `xslt`, `rng`, `plist`, `xaml`, `csproj`, `fsproj`, `vbproj`, `vcxproj`, `props`, `targets`, `nuspec`, `resx` | |
| yaml | `yaml`, `yml` | |
| zig | `zig`, `zon` | |

Everything else is `text`, split by the bounded fallback.

**Ambiguous extensions** are settled this way:

- `.h` stays C, parsed with the C++ grammar, as before the other grammars came.
- `.m` is Objective-C (MATLAB also uses it).
- `.v` is Verilog (Coq also uses it).
- `.fs` is F# (some GLSL fragment shaders use it).
- `.mdx` is Markdown. Its JSX blocks parse as HTML blocks, and its leading
  `import`/`export` lines are file metadata (see [Markdown](#markdown-and-mdx)).
- `.erb` and `.ejs` are embedded templates, `.cu` and `.cuh` CUDA, `.svelte` Svelte,
  `.php` PHP with HTML.
- `.pl` and `.t` are Perl only when the file starts like Perl: a `perl` shebang, or one
  of the first 50 lines (within 4 KB) starting with `use strict`, `use warnings`,
  `use v5`, `use 5.`, `use Test::`, `package Name;`, `package Name {`, `sub name` or
  `my $`/`@`/`%`/`(`.
  Otherwise they are text: Prolog uses `.pl`, other test frameworks `.t`. `.pm` is
  always Perl. A search filter by language decides from the path alone and counts
  every `.pl` and `.t` file as Perl.
- `.mli` reports the language `ocaml`, like `.tsx` reports `typescript`.

The nine languages arko_index parsed first keep their extensions and come first, so
their detection is unchanged. Their grammars also take the newer extensions above
(`mjs`, `cjs`, `mts`, `cts`, `pyi`, `pyw`, `cxx`, `hh`, `hxx`, `ipp`, `tpp`), files that
were text before.

## How the rules work

One engine chunks every language (`arko_index/src/syntax.rs`). Each grammar has a table
of rules (`arko_index/src/rules/tables.rs`) that names node kinds:

| Field | What it does |
|---|---|
| definitions | Preferred boundaries. A definition that fits stays one chunk, and its name joins the symbol path. It shares a chunk only as a function family, as merged clauses, or in a run of eight or more tiny definitions (same kind, at most 256 bytes each, grouped up to 768 bytes). Eight grammars' tables (C#, Dart, Elm, PHP, Ruby, Scala, Solidity and Swift) start from the kinds their `tags.scm` marks `@definition.*` (`rules/tags.rs`), some of which the table skips. |
| bindings | A kind that is a definition when the node at a path is a callable: `const f = () => {}`, R's `f <- function(x)`, Zig's `const Point = struct {}`, SQL's `CREATE TABLE` statements. |
| keyed | A kind that is a definition, a container or an attachment when the text at a path is one of a list: Elixir's `call` whose target is `def` or `defmodule`, its `@doc` attributes. |
| containers | Kinds that always expose their members, even when small (modules, class bodies). |
| compact | Containers kept whole below 700 bytes (a small class). |
| attach | Kinds that join the next unit (comments, doc comments, attributes, decorators, signatures, Markdown headings). Every kind whose name contains `comment` attaches. |
| attach back | Kinds that join the unit before them (`end module`, `endfunction()`, closing tags). |
| bodies | Where a kind's members are (a field or a child kind). The default is the `body` field. |
| names, labels | The symbol name and the scope label (`fn f`, `class C`, `@media (min-width: 800px)`, `<section id="x">`). A label is a keyword and the name, the name alone, the text up to the body, the first line, or the text at a path. |
| families | Callables that can pair as a `function_family`. |
| atomic | Kinds kept whole when they fit, even if they contain definitions (HTML `<script>` and `<style>`). |
| split small | On for most code and for HCL and Protocol Buffers: a node that fits splits into its definitions. Off for prose, markup, the other data formats, CSS, SQL, Make and CMake: what fits stays whole, and only what exceeds the bound splits. |
| merge clauses | Adjacent definitions of the same kind and name, in the same scope, merge while they fit (Haskell equations, Elixir and Erlang function clauses). |

Every table also has these behaviours, which the classic table leaves off: a whitespace
unit attaches forward, a short header (at most 256 bytes) before a split definition's
first member joins that member, and packing joins only units that stayed whole, so the
end of a large JSON value never shares a chunk with the next key.

The tests check every kind and field a table names, apart from the shared lists of the
classic tables, against its grammar (`rules::tests::every_kind_exists_in_its_grammar`), so a typo, or a grammar upgrade
that renames a kind, fails instead of silently matching nothing.

### The original nine languages

Rust, Python, TypeScript, TSX, JavaScript, Go, Java, C and C++ share one table,
`CLASSIC`, whose lists are the engine's original kind lists, and keep their original
scope labels (`syntax::classic_label`). Their output is byte-identical to the build
before the other grammars: the test `original_languages_match_the_golden_output`
compares it with a golden file from that build, and
[`language_check.py --baseline`](#checking-whole-corpora) compares whole corpora.
CUDA and GLSL extend C and C++ and use the same table and labels.

A scope is the chain of definitions that enclose the chunk, the chunk's own definition
included. Their labels:

- Rust: `mod`, `impl Trait for Type`, `trait`, `struct`, `enum`, `fn`, `macro_rules!`
- Python: `class`, `def`, `async def`, through decorators
- TypeScript and JavaScript: `namespace`, `module`, `class`, `interface`, `enum`, `type`,
  `function` (also an arrow function bound to a name), `method`, through `export`
- Go: `func`, `func (r *T) Name`, `type`
- Java: `class`, `interface`, `enum`, `record`, `method`, `constructor`
- C and C++: `namespace`, `class`, `struct`, `enum`, `function`, through templates

## Code

The rule source is `tags.scm` when the table starts from the grammar's tags (then
curated), hand-written otherwise.

| Language | Definitions (preferred boundaries) | Scope labels | Rule source |
|---|---|---|---|
| Bash | functions | `function f` | hand-written |
| C# | namespaces, classes, structs, records, interfaces, enums, methods, constructors, destructors, properties, indexers, events, delegates, operators, local functions | `namespace N › class C › method M` | tags.scm |
| CUDA, GLSL | as C and C++ | as C and C++ | classic |
| Dart | classes, mixins, enums, extensions, class members, functions, getters and setters, typedefs | `class Point › distanceTo` | tags.scm |
| Elixir | `defmodule`, `defprotocol`, `defimpl`, `describe` (containers), `def`, `defp`, `defmacro`, `defmacrop`, `defguard`, `defguardp`, `defdelegate`, `defn`, `defnp`, `test`. `@doc` and other attributes attach to what follows, and clauses of one function merge | `defmodule Shop.Cart › def add` | hand-written |
| Elm | values, types, type aliases, ports. Type annotations attach to their value | `update`, `type Msg` | tags.scm |
| Erlang | functions, records, types, macros. `-spec` and `-doc` attach, and clauses merge | `handle_call`, `record state` | hand-written |
| F# | modules, types, `let` bindings, members, exceptions | `namespace Shop › module Orders › let add` | hand-written |
| Fortran | modules, submodules, programs, functions, subroutines, derived types, interfaces | `module vectors › subroutine normalize` | hand-written |
| Haskell | functions, binds, data types, data families, newtypes, type synonyms and families, classes, instances, foreign imports, pattern synonyms. Signatures and Haddock comments attach, and equations of one function merge | `instance Pretty Instr › pretty` | hand-written |
| Julia | functions (also `f(x) = ...`), macros, structs, abstract and primitive types, modules. Docstrings attach | `module Geometry › function area` | hand-written |
| Kotlin | classes, objects, companion objects, functions, constructors, type aliases | `interface Pricing › fun priceOf` | hand-written |
| Lua | functions, and functions bound to names or table fields | `function Emitter:on` | hand-written |
| Objective-C | `@interface`, `@implementation`, `@protocol`, methods, C functions, structs, enums | `@implementation Rectangle › - (double)area` | hand-written |
| OCaml | `let`, `type`, `module`, `module type`, `class`, `class type`, `exception`, `external`, `val` | `module Queue › let dequeue` | hand-written |
| Odin | procedures, structs, unions, enums, bit fields | `proc area` | hand-written |
| Perl | subroutines. POD attaches | `sub new` | hand-written |
| PHP | namespaces, classes, interfaces, traits, enums, functions, methods | `class OrderTotals › method total` | tags.scm |
| PowerShell | functions, classes, methods, enums | `class Deployment › method Run` | hand-written |
| R | functions bound with `<-` or `=` | `function cv` | hand-written |
| Ruby | modules, classes, singleton classes, methods | `module Shop › class Cart › def add` | tags.scm |
| Scala | classes, objects, traits, enums, `def`, `given`, `type`, extensions | `class OrderTotals › def total` | tags.scm |
| Solidity | contracts, interfaces, libraries, functions, fallback and receive functions, modifiers, constructors, structs, enums | `contract Vault › function deposit` | tags.scm |
| Swift | classes, structs, enums, extensions, protocols, functions, initializers, deinitializers, subscripts, type aliases | `final class OrderTotals › func total` | tags.scm |
| Verilog | modules, interfaces, packages, classes, functions, tasks | `module counter` | hand-written |
| Zig | functions, tests, and `const` bindings of structs, unions, enums and opaques | `const Point › fn distance` | hand-written |

## Prose and markup

### Markdown and MDX

- **Sections.** A section (an ATX heading and what follows it, nested by level) is a
  definition. A section that fits is one chunk. A larger one splits into its heading
  with the content after it, and its subsections, recursively. Paragraphs, lists,
  tables and code blocks between headings pack together, and a code fence that fits
  stays whole.
- **MDX.** JSX blocks parse as HTML blocks, and `import`/`export` lines later in the
  file are ordinary content.
- **Scopes.** The root of every chunk's scope is the document's title: the front
  matter's `title`, or without one the first level-1 heading (`# H1` or a `===`
  underline). The heading chain at the chunk's start follows, for example
  `SQLite storage › ## Storage API › ### Transactions`, from ATX and setext headings.
  Fenced code blocks and the front matter hold no headings. A chunk that starts before
  the first heading but contains one takes that heading's chain. The heading that gives
  the root, and a level-1 heading that repeats the title, are left out of the chain. With
  `Options::front_matter_description` (the CLI's `--front-matter-description`, off by
  default) the root is `title — description`, for comparing the two in retrieval. Line
  ranges appear only where a result is shown (`Chunk::header`), so an edit that moves a
  chunk's lines leaves its embedded text and cached embedding unchanged.

**The metadata region.** Front matter is file metadata. A YAML
(`---` … `---` or `...`) or TOML (`+++` … `+++`) block at the top of a `.md`,
`.markdown` or `.mdx` file, and in MDX the `import`/`export` paragraphs right after it
(or at the top), form the file's metadata region. A byte order mark at the start
belongs to it. Bookkeeping fields (sidebar order, ids, content types) appear in no
chunk and no scope.

Every byte of a file is in exactly one chunk or in its metadata region, which starts
at byte 0. The region is recorded as a `FileMetadata`:

```json
{"file":"docs/storage.md","metadata_range":{"start_byte":0,"end_byte":112,"start_line":1,"end_line":6},"front_matter":"---\ntitle: SQLite storage\n...\n---\n","title":"SQLite storage","description":"How chunks are stored."}
```

`front_matter` is the raw block, delimiters included (empty when the region holds only
MDX lines). `title` and `description` are absent when the front matter has none. The
library returns it with `chunk_str_with_metadata` and `chunk_file_with_metadata` (and
to `chunk_dir_map`'s `map`). The `chunk` command writes it as a line before the file's
chunks (chunk format 3), which a reader tells apart by its `file` key. `outline`
prints it as a line with the path and lines, `metadata` and the title. Watcher deltas
and indexes hold chunks only.

### HTML, XML, Svelte and embedded templates

Elements are the units. An element that fits stays whole, a larger one splits into its
child elements and text, which pack together, and a closing tag stays with the content
before it. HTML `<script>` and `<style>` blocks (and Svelte's) stay whole when they
fit. Scopes are the chain of start tags, for example
`<html lang="en"> › <body> › <section id="summary">`. XML comments attach to the next
element. Embedded templates (ERB, EJS) split between their text and directives when a
file exceeds the bound. A file that fits is one chunk.

### CSS

Rule sets and at-rules (`@media`, `@supports`, `@keyframes`, `@font-face`) are
definitions, labelled by their selectors or their at-rule line:
`@media (min-width: 800px) › .grid`. A small stylesheet is one chunk.

## Data and configuration

A JSON, YAML or TOML file that fits is one chunk. A larger one splits along its
structure, and the scope is the key path:

- **JSON:** pairs, recursing into oversized values (`devDependencies › vitest`).
- **YAML:** documents, then mapping pairs (`jobs › lint`).
- **TOML:** tables and arrays of tables are definitions (`workers.limits`), and the
  top-level keys before them pack together.
- **HCL:** blocks are definitions labelled by their header
  (`resource "aws_lb" "web"`), and a file splits into them like code.
- **Protocol Buffers:** messages, enums, services, RPCs and `extend` blocks
  (`service Orders › rpc Get`).

## Scripts and builds

- **SQL:** statements. `CREATE TABLE`, `VIEW`, `MATERIALIZED VIEW`, `FUNCTION`, `TYPE`,
  `TRIGGER`, `SCHEMA`, `SEQUENCE` and `DATABASE` statements are definitions labelled by
  their first line, and the others pack together. Block comments attach.
- **Make:** rules (labelled by their targets) and `define` blocks. Variables and
  conditionals pack.
- **CMake:** `function()` and `macro()` definitions, and command blocks (`if`,
  `foreach`, `while`, `block`) with their `elseif`, `else` and `end…` commands kept
  together.
- **Bash and PowerShell:** functions (and PowerShell classes) are definitions, and the
  top-level commands between them pack into groups.

## Fields for the next chunker steps

Each table also holds data for the next chunker steps: token-based bounds, breadcrumbs
inside split definitions, grouping of small members and splitting at seams. They
change no output today.

| Field | What it holds | Filled for |
|---|---|---|
| `content` | `Code`, `Prose` (Markdown, HTML, embedded templates) or `Data` (JSON, YAML, TOML, XML, DTD, HCL), for one size bound per kind of content. | every grammar |
| `breadcrumbs` | Kinds that continue the scope inside a split definition (match arms, conditionals, loops, closures, inner functions), each with the field where its label ends, for example Rust `match_arm` up to `value` gives `ToolCall { .. } =>`. | the classic nine, CUDA, GLSL, Objective-C, Bash, C#, Kotlin, Swift, Scala, Ruby, PHP, Dart, Lua, Zig, Solidity, PowerShell, Julia, Perl, Elixir, Erlang, OCaml, R |
| `members`, `fields`, `group_in` | Members that can group when small, the field and property declarations that stay with them, and the containers where grouping applies. | `members`: the same, except Elixir. `fields` and `group_in`: most of them (`fields` is empty for Bash, Dart, Julia, OCaml, Perl and R, `group_in` for Erlang, Perl, R and Zig), and `group_in` has Markdown sections as a container |
| `seams` | Preferred split points inside a large definition (statements, match arms, Markdown blocks). | the same, plus Markdown |

## Checking

The tests in `arko_index/tests/languages.rs` run on a sample of every grammar
(`arko_index/tests/data/languages/`) and on samples of the nine original languages:

- the invariants at ten bounds from 4 to 100,000 bytes: exact coverage after the
  metadata region, the byte bound, UTF-8 boundaries, lines, IDs recomputed from path
  and trimmed content, one language per file
- determinism
- IDs that survive an edit in another chunk
- bounded work on adversarial input (a 70,000-byte line, deep nesting, runs of
  brackets, quotes and backticks)
- invariants on randomly mutated samples (deletions, duplications, inserted syntax,
  swapped lines, truncation) for every grammar
- the scope each sample's rules give, and the structure small bounds give prose,
  markup and data
- Markdown's metadata region, title root and description option
- the original languages against a golden file

`arko_index/tools/gen_language_rules.py GRAMMAR_META` regenerates `rules/tags.rs` from
the grammars' `tags.scm` and `node-types.json` (one folder per grammar key, with
`node-types.json` and `queries/`).

### Checking whole corpora

`arko_index/scripts/language_check.py` chunks whole corpora with the release CLI and
checks every file without the crate's code: coverage after the metadata region, the
bound, UTF-8 boundaries, lines, IDs and token estimates. With `--twice` it checks
determinism. With `--baseline` it runs a build from before the new languages too and
requires byte-identical output for every file that build parsed as one of the original
nine languages. It prints a table per language (files, chunks, sizes, the share of
chunks with a scope, unparsed files, files with a metadata region, failures).

```sh
cargo build --release -p arko_index --example arko_index_cli
python3 arko_index/scripts/language_check.py \
  --cli target/release/examples/arko_index_cli \
  --baseline /path/to/baseline/arko_index_cli \
  --twice --json report.json \
  ~/corpora/linux ~/corpora/django '~/src/repo:--exclude=runs/'
```

A corpus may carry walk flags after a colon, separated by commas.
