use crate::bench_v1::{Chunk, Error, Options};
use sha2::{Digest, Sha256};
use std::{cell::RefCell, collections::HashMap, path::Path};
use tree_sitter::{Language, Node, ParseOptions, Parser};

// An independent byte guard, also applied to every unsupported-language fallback.
// UTF-8 byte count upper-bounds byte-level tokenizer tokens; reserve path + separators.
const RENDER_LIMIT: usize = 24_000;
thread_local! { static PARSERS: RefCell<HashMap<&'static str, Parser>> = RefCell::new(HashMap::new()); }

pub(crate) fn language(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "rs" => "rust",
        "py" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cc" | "cpp" | "hpp" => "cpp",
        _ => "text",
    }
}
fn grammar(lang: &str) -> Option<Language> {
    #[cfg(arko_index_pgo)]
    if let Some(language) = crate::pgo::grammar(lang) {
        return Some(language);
    }
    Some(match lang {
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "python" => tree_sitter_python::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "java" => tree_sitter_java::LANGUAGE.into(),
        "c" => tree_sitter_c::LANGUAGE.into(),
        "cpp" => tree_sitter_cpp::LANGUAGE.into(),
        _ => return None,
    })
}
fn parse(lang: &'static str, text: &str) -> Result<Option<tree_sitter::Tree>, Error> {
    let Some(grammar) = grammar(lang) else {
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
}
fn comment(n: Node) -> bool {
    n.kind().contains("comment") || matches!(n.kind(), "attribute_item" | "inner_attribute_item")
}
fn callable(n: Node) -> bool {
    matches!(
        n.kind(),
        "function_expression" | "arrow_function" | "generator_function" | "function"
    )
}
fn semantic(n: Node) -> bool {
    let k = n.kind();
    matches!(
        k,
        "function_item"
            | "function_definition"
            | "function_declaration"
            | "method_definition"
            | "method_declaration"
            | "constructor_declaration"
            | "destructor_declaration"
            | "generator_function_declaration"
            | "struct_item"
            | "enum_item"
            | "trait_item"
            | "type_item"
            | "type_alias_declaration"
            | "interface_declaration"
            | "type_declaration"
            | "class_declaration"
            | "class_definition"
            | "enum_declaration"
            | "record_declaration"
            | "decorated_definition"
            | "template_declaration"
            | "export_statement"
            | "impl_item"
            | "namespace_definition"
            | "class_specifier"
            | "struct_specifier"
    ) || (matches!(k, "pair" | "variable_declarator")
        && n.child_by_field_name("value").is_some_and(callable))
        || (k == "assignment_expression" && n.child_by_field_name("right").is_some_and(callable))
}
/// Whether a semantic node is among `n`'s named descendants at most 12 named levels
/// down (paths through named nodes only). One cursor walks the whole search instead
/// of a new cursor (and recursion) per visited node.
fn contains_definition(n: Node) -> bool {
    const MAX_DEPTH: usize = 12;
    let mut cursor = n.walk();
    if !cursor.goto_first_child() {
        return false;
    }
    let mut depth = 1;
    loop {
        let c = cursor.node();
        if c.is_named() {
            if semantic(c) {
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
fn name(n: Node, text: &str) -> Option<String> {
    if let Some(s) = n
        .child_by_field_name("name")
        .or_else(|| n.child_by_field_name("key"))
        .or_else(|| {
            if n.kind() == "assignment_expression" {
                n.child_by_field_name("left")
            } else {
                None
            }
        })
        .or_else(|| {
            if n.kind() == "impl_item" {
                n.child_by_field_name("type")
            } else {
                None
            }
        })
    {
        let raw = &text[s.start_byte()..s.end_byte()];
        if raw.len() <= 200 {
            return Some(raw.split_whitespace().collect::<Vec<_>>().join(" "));
        }
    }
    // C/C++ names occur under nested pointer/function declarators.
    if n.kind() == "function_definition" || n.kind() == "declaration" {
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
fn children(n: Node) -> Vec<Node> {
    let mut cursor = n.walk();
    n.named_children(&mut cursor)
        .filter(|x| x.end_byte() > x.start_byte() && !x.is_missing())
        .collect()
}
fn is_container(n: Node) -> bool {
    matches!(
        n.kind(),
        "source_file"
            | "program"
            | "translation_unit"
            | "module"
            | "impl_item"
            | "declaration_list"
            | "namespace_definition"
            | "class_body"
            | "field_declaration_list"
            | "class_declaration"
            | "class_definition"
            | "record_declaration"
            | "class_specifier"
            | "struct_specifier"
            | "enum_declaration"
            | "mod_item"
            | "trait_item"
    )
}
fn child_units(mut n: Node, oversized: bool) -> Vec<Node> {
    // Iterative descent: adversarial parenthesized/declarator chains can contain
    // hundreds of thousands of single-child nodes without a long physical line.
    for _ in 0..128 {
        if let Some(body) = n.child_by_field_name("body")
            && (is_container(n) || oversized)
        {
            let c = children(body);
            if !c.is_empty() {
                return c;
            }
        }
        let c = children(n);
        if matches!(
            n.kind(),
            "decorated_definition" | "template_declaration" | "export_statement"
        ) && let Some(inner) = c.iter().rev().copied().find(|x| semantic(*x))
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
fn emit(out: &mut Vec<Part>, s: usize, e: usize, symbol: &str, kind: &str, atomic: bool) {
    if e > s {
        out.push(Part {
            s,
            e,
            symbol: symbol.to_owned(),
            kind: kind.to_owned(),
            atomic,
            sibling_scope: None,
        });
    }
}
fn fallback(
    text: &str,
    s: usize,
    e: usize,
    symbol: &str,
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
        emit(out, p, q, symbol, kind, false);
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
fn function_families(parts: Vec<Part>, text: &str, limit: usize) -> Vec<Part> {
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
fn tiny_runs(parts: Vec<Part>, text: &str, limit: usize) -> Vec<Part> {
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
        for p in &parts[i..j] {
            if group.as_ref().is_some_and(|g| p.e - g.s > limit.min(768)) {
                result.push(group.take().unwrap());
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
                result.push(group.take().unwrap());
            }
        }
        if let Some(g) = group {
            result.push(g);
        }
        i = j;
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn split(
    n: Node,
    s: usize,
    e: usize,
    inherited: &str,
    text: &str,
    limit: usize,
    depth: usize,
    out: &mut Vec<Part>,
) {
    if e <= s {
        return;
    }
    let mut symbol = inherited.to_owned();
    let is_semantic = semantic(n);
    if let Some(name) = name(n, text).filter(|_| is_semantic) {
        if !symbol.is_empty() {
            symbol.push_str("::");
        }
        symbol.push_str(&name);
    }
    let oversized = e - s > limit;
    let container = is_container(n) || (!is_semantic && contains_definition(n));
    if (!oversized && !container) || depth >= 100 {
        if oversized {
            fallback(text, s, e, &symbol, n.kind(), limit, out);
        } else {
            emit(out, s, e, &symbol, n.kind(), is_semantic);
            if matches!(
                n.kind(),
                "function_item"
                    | "function_definition"
                    | "function_declaration"
                    | "method_definition"
                    | "method_declaration"
            ) {
                out.last_mut().unwrap().sibling_scope = n.parent().map(|p| p.id());
            }
        }
        return;
    }
    let c = child_units(n, oversized);
    let mut units: Vec<(usize, Node)> = Vec::new();
    let mut prefix = None;
    for child in c {
        if child.start_byte() < s || child.end_byte() > e {
            continue;
        }
        if comment(child) {
            prefix.get_or_insert(child.start_byte());
            continue;
        }
        let start = prefix.take().unwrap_or(child.start_byte());
        units.push((line_boundary(text, start, s), child));
    }
    if units.is_empty() {
        fallback(text, s, e, &symbol, n.kind(), limit, out);
        return;
    }
    // Tiny classes remain self-contained; large containers expose member boundaries.
    if !oversized
        && matches!(
            n.kind(),
            "class_declaration" | "class_definition" | "record_declaration"
        )
        && e - s < 700
    {
        emit(out, s, e, &symbol, n.kind(), true);
        return;
    }
    if units.len() == 1 && units[0].1.id() == n.id() {
        fallback(text, s, e, &symbol, n.kind(), limit, out);
        return;
    }
    let mut parts = Vec::new();
    for i in 0..units.len() {
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
            text,
            limit,
            depth + 1,
            &mut parts,
        );
    }
    // Pack short declarations/statements within the same lexical container, not unrelated functions.
    let pack = if container { limit.min(1000) } else { limit };
    for part in tiny_runs(function_families(parts, text, limit), text, limit) {
        if let Some(last) = out.last_mut()
            && !last.atomic
            && !part.atomic
            && last.s >= s
            && last.e == part.s
            && part.e - last.s <= pack
        {
            last.e = part.e;
            if last.kind != part.kind {
                last.kind = if container { "declarations" } else { n.kind() }.to_owned();
            }
            if last.symbol != part.symbol {
                last.symbol = symbol.clone();
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

pub(crate) fn chunk(path: &Path, text: &str, options: &Options) -> Result<Vec<Chunk>, Error> {
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
    let language = language(path);
    let parser_language = if path.extension().is_some_and(|x| x == "tsx") {
        "tsx"
    } else if path.extension().is_some_and(|x| x == "h") {
        "cpp"
    } else {
        language
    };
    let mut parts = Vec::new();
    // Very long single lines are deliberately bounded fallback units, including
    // malformed strings and generated/minified source. They never enter unbounded parsing.
    let giant_line = text.split('\n').any(|line| line.len() > 65_536);
    let tree = if giant_line {
        None
    } else {
        parse(parser_language, text)?
    };
    let tree = if matches!(parser_language, "c" | "cpp")
        && tree.as_ref().is_some_and(|t| t.root_node().has_error())
    {
        let normalized = crate::normalize::annotations(text);
        if normalized != text {
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
    if let Some(tree) = tree {
        split(
            tree.root_node(),
            0,
            text.len(),
            "",
            text,
            limit,
            0,
            &mut parts,
        );
    } else {
        fallback(text, 0, text.len(), "", "text", limit, &mut parts);
    }
    let mut starts = vec![0usize];
    starts.extend(text.match_indices('\n').map(|(i, _)| i + 1));
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
        let start_line = starts.partition_point(|&b| b <= p.s);
        let end_line = starts.partition_point(|&b| b < p.e);
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
        });
    }
    Ok(chunks)
}
