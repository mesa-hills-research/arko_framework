use crate::{
    bench_v1::{Chunk, Error, FileChunks, FileMetadata, Options, SourceSpan},
    headers::{SCOPE_SEPARATOR, clip, collapse},
    rules::{Label, Rules, resolve},
};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::HashMap, path::Path};
use tree_sitter::{Node, ParseOptions, Parser};

// An independent byte guard, also applied to every unsupported-language fallback.
// UTF-8 byte count upper-bounds byte-level tokenizer tokens; reserve path + separators.
const RENDER_LIMIT: usize = 24_000;
thread_local! { static PARSERS: RefCell<HashMap<&'static str, Parser>> = RefCell::new(HashMap::new()); }

/// The `language` of a file's chunks, from its path alone (`.pl` and `.t` count as Perl
/// here; chunking also checks their content).
pub(crate) fn language(path: &Path) -> &'static str {
    crate::languages::detect_path(path).name
}
fn parse(lang: &'static str, text: &str) -> Result<Option<tree_sitter::Tree>, Error> {
    let Some(grammar) = crate::languages::grammar(lang) else {
        return Ok(None);
    };
    PARSERS.with(|parsers| {
        let mut parsers = parsers.borrow_mut();
        if !parsers.contains_key(lang) {
            let mut parser = Parser::new();
            parser.set_language(&grammar)?;
            parsers.insert(lang, parser);
        }
        let parser = parsers.get_mut(lang).unwrap();
        parser.reset();
        // Deterministic work guard instead of a machine-speed dependent time deadline.
        let mut calls = 0usize;
        let budget = (1000 + text.len() / 32).min(50_000);
        let mut progress = |_: &tree_sitter::ParseState| {
            calls += 1;
            calls > budget
        };
        let tree = parser.parse_with_options(
            &mut |i, _| &text.as_bytes()[i..],
            None,
            Some(ParseOptions::new().progress_callback(&mut progress)),
        );
        parser.reset();
        Ok(tree)
    })
}
#[derive(Clone, Debug)]
struct Part {
    s: usize,
    e: usize,
    symbol: String,
    kind: String,
    atomic: bool,
    // Ephemeral AST parent identity: grouping must not cross lexical scopes even
    // when a wrapper has no display name (e.g. a module or anonymous namespace).
    sibling_scope: Option<usize>,
    /// The chain of enclosing definition labels (`Chunk::scope`). Metadata only:
    /// nothing below reads it to choose boundaries.
    scope: String,
    /// While the node being split packs its parts: whether the unit the part came from
    /// stayed one part (`Table::pack_whole_units`). The pieces of a split unit were
    /// packed when that unit was split.
    whole: bool,
}
/// Whether a semantic node is among `n`'s named descendants at most 12 named levels
/// down (paths through named nodes only). One cursor walks the whole search instead
/// of a new cursor (and recursion) per visited node.
fn contains_definition(n: Node, text: &str, rules: &Rules) -> bool {
    const MAX_DEPTH: usize = 12;
    let mut cursor = n.walk();
    if !cursor.goto_first_child() {
        return false;
    }
    let mut depth = 1;
    loop {
        let c = cursor.node();
        if c.is_named() {
            if rules.definition(c, text) {
                return true;
            }
            if depth < MAX_DEPTH && cursor.goto_first_child() {
                depth += 1;
                continue;
            }
        }
        while !cursor.goto_next_sibling() {
            if depth == 1 {
                return false;
            }
            cursor.goto_parent();
            depth -= 1;
        }
    }
}
/// A node's name: the first of the table's name fields, else its kind's name paths
/// (at most 200 bytes, whitespace collapsed); else, for declarator kinds, the
/// identifier at the end of C's declarator chain.
fn name(n: Node, text: &str, rules: &Rules) -> Option<String> {
    let t = rules.table;
    let kind = n.kind();
    if let Some(s) = t
        .name_fields
        .iter()
        .find_map(|f| n.child_by_field_name(f))
        .or_else(|| {
            t.names
                .iter()
                .filter(|(k, _)| *k == kind)
                .find_map(|(_, path)| resolve(n, path))
        })
    {
        let raw = &text[s.start_byte()..s.end_byte()];
        if raw.len() <= 200 {
            return Some(raw.split_whitespace().collect::<Vec<_>>().join(" "));
        }
    }
    // C/C++ names occur under nested pointer/function declarators.
    if t.declarators.contains(&kind) {
        let mut d = n.child_by_field_name("declarator");
        for _ in 0..16 {
            let Some(node) = d else {
                break;
            };
            if matches!(
                node.kind(),
                "identifier"
                    | "field_identifier"
                    | "qualified_identifier"
                    | "operator_name"
                    | "destructor_name"
            ) {
                return Some(text[node.byte_range()].to_owned());
            }
            d = node.child_by_field_name("declarator");
        }
    }
    None
}
/// The label of a node that opens a scope in `Chunk::scope` (`fn f`,
/// `impl Display for Point`, `class C`, `def m`, `namespace N`), or `None`.
fn scope_label(n: Node, text: &str, lang: &str, rules: &Rules) -> Option<String> {
    if rules.table.classic {
        classic_label(n, text, lang, rules)
    } else {
        table_label(n, text, rules)
    }
}
/// The label from the table: a wrapper's inner definition's label, a keyed
/// definition's `key name`, the kind's label rule, or a tags.scm definition's
/// `category name`.
fn table_label(n: Node, text: &str, rules: &Rules) -> Option<String> {
    let t = rules.table;
    let kind = n.kind();
    if rules.wrapper(n) {
        return children(n)
            .into_iter()
            .rev()
            .find(|c| rules.definition(*c, text))
            .and_then(|d| table_label(d, text, rules));
    }
    if let Some(keyed) = rules.keyed(n, text).filter(|k| k.key_label) {
        let key = resolve(n, keyed.key)?;
        let key = clip(collapse(&text[key.byte_range()]));
        return Some(match name(n, text, rules) {
            Some(name) => format!("{key} {}", clip(name)),
            None => key,
        });
    }
    let label = match t.labels.iter().find(|(k, _)| *k == kind) {
        // A binding's label is only for the bindings that define something.
        Some(_) if rules.conditional(n) && !rules.definition(n, text) => return None,
        Some((_, label)) => *label,
        None => match t.tag_category(kind) {
            Some(category) if rules.definition(n, text) => Label::Keyword(category),
            _ => return None,
        },
    };
    let label = match label {
        Label::Keyword(keyword) => format!("{keyword} {}", clip(name(n, text, rules)?)),
        Label::Name => clip(unquote(name(n, text, rules)?)),
        Label::Header => {
            let end = rules
                .body(n)
                .filter(|b| b.start_byte() > n.start_byte())
                .map_or(n.end_byte(), |b| b.start_byte());
            header(&text[n.start_byte()..end])
        }
        Label::Path(path) => {
            let at = resolve(n, path)?;
            header(&text[at.byte_range()])
        }
        Label::Line => {
            let node = &text[n.byte_range()];
            header(node.split('\n').next().unwrap_or(node))
        }
    };
    (!label.is_empty()).then_some(label)
}
/// A quoted name (a JSON or YAML key) without its quotes.
fn unquote(name: String) -> String {
    for quote in ['"', '\''] {
        if name.len() >= 2 && name.starts_with(quote) && name.ends_with(quote) {
            return name[1..name.len() - 1].to_owned();
        }
    }
    name
}
/// A label from a node's leading text: its first 400 bytes (to a character
/// boundary), whitespace collapsed, without a trailing `{`, `:`, `=`, `(`, `do` or
/// `where`.
fn header(text: &str) -> String {
    let mut end = text.len().min(400);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let mut label = collapse(&text[..end]);
    loop {
        let trimmed = label.trim_end_matches(['{', ':', '=', '(', ' ']).to_owned();
        let trimmed = trimmed
            .strip_suffix(" do")
            .or_else(|| trimmed.strip_suffix(" where"))
            .unwrap_or(&trimmed)
            .to_owned();
        if trimmed.len() == label.len() {
            break;
        }
        label = trimmed;
    }
    clip(label)
}
/// The labels of the classic languages. `lang` is the parser language: Python, C and
/// C++ share `function_definition`, Go and TypeScript `function_declaration`, Go and
/// Java `method_declaration`.
fn classic_label(n: Node, text: &str, lang: &str, rules: &Rules) -> Option<String> {
    let kind = n.kind();
    let keyword = match kind {
        // Wrappers take the label of the definition they wrap.
        "decorated_definition" => {
            return n
                .child_by_field_name("definition")
                .and_then(|d| classic_label(d, text, lang, rules));
        }
        "export_statement" => {
            return n
                .child_by_field_name("declaration")
                .and_then(|d| classic_label(d, text, lang, rules));
        }
        "template_declaration" => {
            return children(n)
                .into_iter()
                .rev()
                .find(|c| rules.definition(*c, text))
                .and_then(|d| classic_label(d, text, lang, rules));
        }
        "impl_item" => return impl_label(n, text),
        "method_declaration" if lang == "go" => {
            let receiver = n.child_by_field_name("receiver")?;
            let name = n.child_by_field_name("name")?;
            return Some(format!(
                "func {} {}",
                clip(collapse(&text[receiver.byte_range()])),
                clip(collapse(&text[name.byte_range()]))
            ));
        }
        "type_declaration" if lang == "go" => {
            let specs: Vec<_> = children(n)
                .into_iter()
                .filter(|c| matches!(c.kind(), "type_spec" | "type_alias"))
                .collect();
            let [spec] = specs.as_slice() else {
                return None;
            };
            let name = spec.child_by_field_name("name")?;
            return Some(format!("type {}", clip(collapse(&text[name.byte_range()]))));
        }
        // Rust, and the C, C++, Java and TypeScript kinds that take the same keyword
        "mod_item" => "mod",
        "trait_item" => "trait",
        "struct_item" | "struct_specifier" => "struct",
        "enum_item" | "enum_declaration" | "enum_specifier" => "enum",
        "union_item" => "union",
        "function_item" | "function_signature_item" => "fn",
        "macro_definition" => "macro_rules!",
        // Python, C and C++
        "function_definition" if lang == "python" => {
            if text[n.byte_range()].starts_with("async") {
                "async def"
            } else {
                "def"
            }
        }
        "function_definition" => "function",
        "class_definition"
        | "class_declaration"
        | "abstract_class_declaration"
        | "class"
        | "class_specifier" => "class",
        "namespace_definition" => {
            return Some(match name(n, text, rules) {
                Some(name) => format!("namespace {}", clip(name)),
                None => "namespace".to_owned(),
            });
        }
        // JavaScript, TypeScript, Java, Go
        "interface_declaration" => "interface",
        "record_declaration" => "record",
        "type_alias_declaration" => "type",
        "internal_module" => "namespace",
        "module" if lang == "typescript" || lang == "tsx" => "module",
        "function_declaration" if lang == "go" => "func",
        "function_declaration" | "generator_function_declaration" | "function_signature" => {
            "function"
        }
        "method_definition"
        | "method_declaration"
        | "method_signature"
        | "abstract_method_signature" => "method",
        "constructor_declaration" => "constructor",
        // A callable bound to a name (`const f = () => ...`, `{ f: function () {} }`).
        "pair" | "variable_declarator" | "assignment_expression" if rules.definition(n, text) => {
            "function"
        }
        _ => return None,
    };
    Some(format!("{keyword} {}", clip(name(n, text, rules)?)))
}
/// `impl Type`, `impl Trait for Type` or `impl !Trait for Type`.
fn impl_label(n: Node, text: &str) -> Option<String> {
    let ty = clip(collapse(&text[n.child_by_field_name("type")?.byte_range()]));
    Some(match n.child_by_field_name("trait") {
        Some(t) => {
            let negative = text[n.start_byte()..t.start_byte()].contains('!');
            format!(
                "impl {}{} for {ty}",
                if negative { "!" } else { "" },
                clip(collapse(&text[t.byte_range()]))
            )
        }
        None => format!("impl {ty}"),
    })
}
/// The scope of parts merged into one chunk (a function family, a group of tiny
/// definitions): their own scope when they share it; otherwise the enclosing scope
/// followed by the members' own labels joined with ` | ` (bounded by `cap` bytes,
/// then `…`), as `symbol_path` joins symbols; the enclosing scope alone when a member
/// has no label of its own.
fn merged_scope(parent: &str, members: &[Part], cap: usize) -> String {
    if members.iter().all(|m| m.scope == members[0].scope) {
        return members[0].scope.clone();
    }
    let mut tails = Vec::with_capacity(members.len());
    for m in members {
        let tail = if parent.is_empty() {
            Some(m.scope.as_str())
        } else {
            m.scope
                .strip_prefix(parent)
                .and_then(|t| t.strip_prefix(SCOPE_SEPARATOR))
        };
        match tail {
            Some(t) if !t.is_empty() => tails.push(t),
            _ => return parent.to_owned(),
        }
    }
    let mut joined = String::new();
    for t in tails {
        if !joined.is_empty() {
            if joined.len() + t.len() >= cap {
                joined.push_str(" | …");
                break;
            }
            joined.push_str(" | ");
        }
        joined.push_str(t);
    }
    if parent.is_empty() {
        joined
    } else {
        format!("{parent}{SCOPE_SEPARATOR}{joined}")
    }
}
fn finish_group(mut group: Part, members: &[Part], parent: &str) -> Part {
    group.scope = merged_scope(parent, members, 500);
    group
}
fn children(n: Node) -> Vec<Node> {
    let mut cursor = n.walk();
    n.named_children(&mut cursor)
        .filter(|x| x.end_byte() > x.start_byte() && !x.is_missing())
        .collect()
}
fn child_units<'t>(mut n: Node<'t>, oversized: bool, text: &str, rules: &Rules) -> Vec<Node<'t>> {
    // Iterative descent: adversarial parenthesized/declarator chains can contain
    // hundreds of thousands of single-child nodes without a long physical line.
    for _ in 0..128 {
        if let Some(body) = rules.body(n)
            && (rules.container(n, text) || oversized)
        {
            let c = children(body);
            if !c.is_empty() {
                return c;
            }
        }
        let c = children(n);
        if rules.wrapper(n)
            && let Some(inner) = c.iter().rev().copied().find(|x| rules.definition(*x, text))
        {
            n = inner;
            continue;
        }
        if oversized && c.len() == 1 && c[0].byte_range() != n.byte_range() {
            n = c[0];
            continue;
        }
        return c;
    }
    children(n)
}
fn line_boundary(text: &str, byte: usize, floor: usize) -> usize {
    let s = text[..byte].rfind('\n').map_or(0, |i| i + 1);
    if s >= floor
        && text[s..byte]
            .chars()
            .all(|c| c == ' ' || c == '\t' || c == '\r')
    {
        s
    } else {
        byte
    }
}
#[allow(clippy::too_many_arguments)]
fn emit(
    out: &mut Vec<Part>,
    s: usize,
    e: usize,
    symbol: &str,
    scope: &str,
    kind: &str,
    atomic: bool,
) {
    if e > s {
        out.push(Part {
            s,
            e,
            symbol: symbol.to_owned(),
            kind: kind.to_owned(),
            atomic,
            sibling_scope: None,
            scope: scope.to_owned(),
            whole: true,
        });
    }
}
#[allow(clippy::too_many_arguments)]
fn fallback(
    text: &str,
    s: usize,
    e: usize,
    symbol: &str,
    scope: &str,
    kind: &str,
    limit: usize,
    out: &mut Vec<Part>,
) {
    let mut p = s;
    while p < e {
        let mut q = p.saturating_add(limit).min(e);
        while !text.is_char_boundary(q) {
            q -= 1;
        }
        if q < e {
            // Prefer a complete line, then a whitespace/word boundary. Never loop on a long line.
            let min = p + ((q - p) / 3).max(1);
            if let Some(i) = text[p..q]
                .rfind('\n')
                .map(|i| p + i + 1)
                .filter(|i| *i >= min)
            {
                q = i;
            } else if let Some(i) = text[p..q]
                .rfind(char::is_whitespace)
                .map(|i| p + i)
                .filter(|i| *i >= min)
            {
                q = i;
            }
        }
        emit(out, p, q, symbol, scope, kind, false);
        p = q;
    }
}
// A family is an exact overload name, or the first word of a compound name.
// Do not merge bare `render` with `render_debug`, nor generic short prefixes such
// as get/set/is. Require an actual identifier, not a parser-recovered signature.
fn function_family_key(part: &Part) -> Option<(&str, &str, bool)> {
    part.sibling_scope?;
    let (parent, name) = part.symbol.rsplit_once("::").unwrap_or(("", &part.symbol));
    if name.is_empty()
        || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_')
        || !name.as_bytes()[0].is_ascii_alphabetic()
    {
        return None;
    }
    let stem = name
        .bytes()
        .take_while(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        .count();
    let compound = stem >= 4
        && stem < name.len()
        && (name.as_bytes()[stem].is_ascii_uppercase()
            || (name.as_bytes()[stem] == b'_' && stem + 1 < name.len()));
    Some((
        parent,
        if compound { &name[..stem] } else { name },
        compound,
    ))
}

// Coherent short sibling pairs preserve complete function bodies while sharing
// useful nearby vocabulary. The entire same-family run must be exactly two:
// never greedily phase a long accessor run into windows. Each function retains
// at least one third of the group's nonwhitespace bytes, independent of queries.
fn function_families(parts: Vec<Part>, text: &str, limit: usize, parent_scope: &str) -> Vec<Part> {
    let mut result = Vec::with_capacity(parts.len());
    let mut i = 0;
    while i < parts.len() {
        let key = function_family_key(&parts[i]);
        let mut j = i + 1;
        if key.is_some() {
            while j < parts.len()
                && parts[j].sibling_scope == parts[i].sibling_scope
                && parts[j].kind == parts[i].kind
                && function_family_key(&parts[j]) == key
            {
                j += 1;
            }
        }
        if j - i == 2
            && parts[j - 1].e - parts[i].s <= limit.min(1200)
            && text[parts[i].s..parts[i].e].trim() != text[parts[i + 1].s..parts[i + 1].e].trim()
        {
            let masses: Vec<_> = parts[i..j]
                .iter()
                .map(|p| {
                    text[p.s..p.e]
                        .bytes()
                        .filter(|b| !b.is_ascii_whitespace())
                        .count()
                })
                .collect();
            let total = masses.iter().sum::<usize>();
            if total > 0 && masses.iter().all(|m| m * 3 >= total) {
                let mut group = parts[i].clone();
                group.e = parts[j - 1].e;
                group.kind = "function_family".into();
                group.symbol.push_str(" | ");
                group.symbol.push_str(&parts[j - 1].symbol);
                group.sibling_scope = None;
                group.scope = merged_scope(parent_scope, &parts[i..j], usize::MAX);
                result.push(group);
                i = j;
                continue;
            }
        }
        result.extend_from_slice(&parts[i..j]);
        i = j;
    }
    result
}

// Coalesce sustained runs of tiny sibling definitions (generated accessors/tables),
// not ordinary short files. Content-defined cuts resynchronize after local insertions;
// unrelated distant groups keep their identity instead of moving fixed-size windows.
fn tiny_runs(parts: Vec<Part>, text: &str, limit: usize, parent_scope: &str) -> Vec<Part> {
    let mut result = Vec::new();
    let mut i = 0;
    while i < parts.len() {
        let mut j = i;
        while j < parts.len()
            && parts[j].atomic
            && parts[j].kind != "function_family"
            && parts[j].e - parts[j].s <= 256
            && parts[j].kind == parts[i].kind
        {
            j += 1;
        }
        if j - i < 8 {
            result.push(parts[i].clone());
            i += 1;
            continue;
        }
        let mut group: Option<Part> = None;
        // Members of the group being built, for its scope.
        let mut first = i;
        for (at, p) in parts.iter().enumerate().take(j).skip(i) {
            if group.as_ref().is_some_and(|g| p.e - g.s > limit.min(768)) {
                result.push(finish_group(
                    group.take().unwrap(),
                    &parts[first..at],
                    parent_scope,
                ));
            }
            if let Some(g) = &mut group {
                g.e = p.e;
                g.kind = "small_definition_group".into();
                if g.symbol.len() + p.symbol.len() < 500 {
                    g.symbol.push_str(" | ");
                    g.symbol.push_str(&p.symbol);
                }
            } else {
                group = Some(p.clone());
                first = at;
            }
            let mut key = text[p.s..p.e]
                .trim()
                .bytes()
                .fold(0xcbf29ce484222325u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(0x100000001b3)
                });
            // Avalanche all bits: raw FNV low bits correlate for repeated numeric names.
            key = (key ^ (key >> 33)).wrapping_mul(0xff51afd7ed558ccd);
            key = (key ^ (key >> 33)).wrapping_mul(0xc4ceb9fe1a85ec53);
            key ^= key >> 33;
            if key & 7 == 0 {
                result.push(finish_group(
                    group.take().unwrap(),
                    &parts[first..=at],
                    parent_scope,
                ));
            }
        }
        if let Some(g) = group {
            result.push(finish_group(g, &parts[first..j], parent_scope));
        }
        i = j;
    }
    result
}

/// Adjacent definitions of the same kind and name in the same scope merge while the
/// result fits: Haskell and Elixir write one function as several clauses.
fn merge_clauses(parts: Vec<Part>, limit: usize) -> Vec<Part> {
    let mut out: Vec<Part> = Vec::with_capacity(parts.len());
    for part in parts {
        if let Some(last) = out.last_mut()
            && last.atomic
            && part.atomic
            && last.kind == part.kind
            && !part.symbol.is_empty()
            && last.symbol == part.symbol
            && last.scope == part.scope
            && last.e == part.s
            && part.e - last.s <= limit
        {
            last.e = part.e;
            continue;
        }
        out.push(part);
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn split(
    n: Node,
    s: usize,
    e: usize,
    inherited: &str,
    inherited_scope: &str,
    lang: &str,
    rules: &Rules,
    text: &str,
    limit: usize,
    depth: usize,
    out: &mut Vec<Part>,
) {
    if e <= s {
        return;
    }
    let mut symbol = inherited.to_owned();
    let is_semantic = rules.definition(n, text);
    if let Some(name) = name(n, text, rules).filter(|_| is_semantic) {
        if !symbol.is_empty() {
            symbol.push_str("::");
        }
        symbol.push_str(&name);
    }
    // The scope chain grows like the symbol path, by labels (`impl A for B`, `fn f`),
    // and also through modules and namespaces without a symbol.
    let scope = match scope_label(n, text, lang, rules) {
        Some(label) if inherited_scope.is_empty() => label,
        Some(label) => format!("{inherited_scope}{SCOPE_SEPARATOR}{label}"),
        None => inherited_scope.to_owned(),
    };
    let oversized = e - s > limit;
    // A node that contains definitions is split into them, even when it fits (code) or
    // only when it does not (tables without `split_small`). Atomic kinds that fit
    // stay whole.
    let container = rules.container(n, text)
        || (!is_semantic
            && (rules.table.split_small || oversized)
            && !(rules.atomic(n) && !oversized)
            && contains_definition(n, text, rules));
    if (!oversized && !container) || depth >= 100 {
        if oversized {
            fallback(text, s, e, &symbol, &scope, n.kind(), limit, out);
        } else {
            emit(out, s, e, &symbol, &scope, n.kind(), is_semantic);
            if is_semantic && rules.family(n) {
                out.last_mut().unwrap().sibling_scope = n.parent().map(|p| p.id());
            }
        }
        return;
    }
    let c = child_units(n, oversized, text, rules);
    let mut units: Vec<(usize, Node)> = Vec::new();
    let mut prefix = None;
    for child in c {
        if child.start_byte() < s || child.end_byte() > e {
            continue;
        }
        if rules.attaches(child, text) {
            prefix.get_or_insert(child.start_byte());
            continue;
        }
        // A closer is covered by the range of the unit before it.
        if !units.is_empty() && rules.attaches_back(child) {
            continue;
        }
        let start = prefix.take().unwrap_or(child.start_byte());
        units.push((line_boundary(text, start, s), child));
    }
    if units.is_empty() {
        fallback(text, s, e, &symbol, &scope, n.kind(), limit, out);
        return;
    }
    // Tiny classes remain self-contained; large containers expose member boundaries.
    if !oversized && rules.compact(n) && e - s < 700 {
        emit(out, s, e, &symbol, &scope, n.kind(), true);
        return;
    }
    if units.len() == 1 && units[0].1.id() == n.id() {
        fallback(text, s, e, &symbol, &scope, n.kind(), limit, out);
        return;
    }
    let mut parts: Vec<Part> = Vec::new();
    for i in 0..units.len() {
        let first = parts.len();
        let a = if i == 0 { s } else { units[i].0 };
        let b = if i + 1 == units.len() {
            e
        } else {
            units[i + 1].0
        };
        split(
            units[i].1,
            a,
            b,
            &symbol,
            &scope,
            lang,
            rules,
            text,
            limit,
            depth + 1,
            &mut parts,
        );
        // A node with one unit passes its unit's marks on unchanged.
        if units.len() > 1 {
            let whole = parts.len() - first == 1;
            for part in &mut parts[first..] {
                part.whole = whole;
            }
        }
    }
    if rules.table.join_header
        && depth > 0
        && parts.len() >= 2
        && !parts[0].atomic
        && parts[1].atomic
        && parts[0].e == parts[1].s
        && parts[0].e - parts[0].s <= 256
        && parts[1].e - parts[0].s <= limit
    {
        let header = parts.remove(0);
        parts[0].s = header.s;
    }
    if rules.table.merge_clauses {
        parts = merge_clauses(parts, limit);
    }
    // Pack short declarations/statements within the same lexical container, not unrelated functions.
    let pack = if container {
        limit.min(rules.table.container_pack)
    } else {
        limit
    };
    for part in tiny_runs(
        function_families(parts, text, limit, &scope),
        text,
        limit,
        &scope,
    ) {
        if let Some(last) = out.last_mut()
            && !last.atomic
            && !part.atomic
            && last.s >= s
            && last.e == part.s
            && part.e - last.s <= pack
            && (!rules.table.pack_whole_units || (last.whole && part.whole))
        {
            last.e = part.e;
            if last.kind != part.kind {
                last.kind = if container { "declarations" } else { n.kind() }.to_owned();
            }
            if last.symbol != part.symbol {
                last.symbol = symbol.clone();
            }
            if last.scope != part.scope {
                last.scope = scope.clone();
            }
            continue;
        }
        out.push(part);
    }
}

fn error_mass(root: Node) -> usize {
    let mut stack = vec![root];
    let mut mass = 0;
    while let Some(n) = stack.pop() {
        if n.is_error() {
            mass += n.end_byte() - n.start_byte();
        } else if n.is_missing() {
            mass += 1;
        } else if n.has_error() {
            stack.extend(children(n));
        }
    }
    mass
}

pub(crate) fn chunk(path: &Path, text: &str, options: &Options) -> Result<FileChunks, Error> {
    if options.max_bytes < 4 {
        return Err("max_bytes must be at least 4 (one UTF-8 scalar)".into());
    }
    let path_s = path.to_str().ok_or("paths must be UTF-8")?.to_owned();
    let budget = RENDER_LIMIT
        .checked_sub(path_s.len() + 1)
        .ok_or("path exceeds rendered chunk budget")?;
    if budget < 4 {
        return Err("path leaves no UTF-8 source budget".into());
    }
    let limit = options.max_bytes.min(budget);
    let detected = crate::languages::detect(path, text);
    let language = detected.name;
    let parser_language = detected.parser;
    let markdown = crate::headers::is_markdown(path);
    // A Markdown file's front matter (and MDX's import/export lines after it) is file
    // metadata: the chunks cover the text after it.
    let region = if markdown {
        let mdx = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("mdx"));
        crate::front_matter::region(text, mdx)
    } else {
        None
    };
    let base = region.as_ref().map_or(0, |r| r.end);
    let body = &text[base..];
    let mut parts = Vec::new();
    // Very long single lines are deliberately bounded fallback units, including
    // malformed strings and generated/minified source. They never enter unbounded parsing.
    let giant_line = body.split('\n').any(|line| line.len() > 65_536);
    let tree = if giant_line {
        None
    } else {
        parse(parser_language, body)?
    };
    let tree = if matches!(parser_language, "c" | "cpp")
        && tree.as_ref().is_some_and(|t| t.root_node().has_error())
    {
        let normalized = crate::normalize::annotations(body);
        if normalized != body {
            let recovered = parse(parser_language, &normalized)?;
            if recovered.as_ref().is_some_and(|t| {
                error_mass(t.root_node()) < error_mass(tree.as_ref().unwrap().root_node())
            }) {
                recovered
            } else {
                tree
            }
        } else {
            tree
        }
    } else {
        tree
    };
    if let Some(tree) = tree
        && let Some(rules) = crate::rules::rules(parser_language)
    {
        split(
            tree.root_node(),
            0,
            body.len(),
            "",
            "",
            parser_language,
            rules,
            body,
            limit,
            0,
            &mut parts,
        );
    } else {
        fallback(body, 0, body.len(), "", "", "text", limit, &mut parts);
    }
    for part in &mut parts {
        part.s += base;
        part.e += base;
    }
    if markdown {
        let (title, description) = match &region {
            Some(r) => (
                r.title.as_deref(),
                r.description
                    .as_deref()
                    .filter(|_| options.front_matter_description),
            ),
            None => (None, None),
        };
        let scopes = crate::headers::markdown_scopes(
            text,
            parts.iter().map(|p| (p.s, p.e)),
            title,
            description,
        );
        for (part, scope) in parts.iter_mut().zip(scopes) {
            part.scope = scope;
        }
    }
    let mut starts = vec![0usize];
    starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
    let line_of = |s: usize, e: usize| {
        (
            starts.partition_point(|&b| b <= s),
            starts.partition_point(|&b| b < e),
        )
    };
    let metadata = region.map(|r| {
        let (start_line, end_line) = line_of(0, r.end);
        FileMetadata {
            file: path_s.clone(),
            metadata_range: SourceSpan {
                start_byte: 0,
                end_byte: r.end,
                start_line,
                end_line,
            },
            front_matter: r.front_matter,
            title: r.title,
            description: r.description,
        }
    });
    let mut duplicates: HashMap<String, u64> = HashMap::new();
    let mut chunks = Vec::with_capacity(parts.len());
    for p in parts {
        // Trim only for identity: indentation/newlines remain fully covered in the range.
        let source = &text[p.s..p.e];
        let mut digest = Sha256::new();
        digest.update(path_s.as_bytes());
        digest.update([0]);
        digest.update(source.trim().as_bytes());
        let content = format!("{:x}", digest.finalize());
        let occurrence = duplicates.entry(content.clone()).or_default();
        let id = format!("{}-{}", content, occurrence);
        *occurrence += 1;
        let (start_line, end_line) = line_of(p.s, p.e);
        chunks.push(Chunk {
            id,
            path: path_s.clone(),
            start_byte: p.s,
            end_byte: p.e,
            start_line,
            end_line,
            language: language.to_owned(),
            symbol_path: p.symbol,
            kind: p.kind,
            token_estimate: source.len() + path_s.len() + 1,
            scope: crate::headers::bounded(p.scope),
        });
    }
    Ok(FileChunks { metadata, chunks })
}
