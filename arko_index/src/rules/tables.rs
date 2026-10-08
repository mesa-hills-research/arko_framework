//! The rules table of every grammar. `docs/arko_index-languages.md` explains each one.
use super::{Content, Keyed, Label, Role, Table, tags};

const D: Table = Table::DEFAULT;

/// The table of a parser key (`languages::PARSERS`); `text` and unknown keys get an
/// empty table.
pub(crate) fn table(parser: &str) -> &'static Table {
    match parser {
        "rust" | "python" | "typescript" | "tsx" | "javascript" | "go" | "java" | "c" | "cpp" => {
            &CLASSIC_TABLES[classic_index(parser)]
        }
        "cuda" => &CUDA,
        "glsl" => &GLSL,
        "bash" => &BASH,
        "c_sharp" => &C_SHARP,
        "cmake" => &CMAKE,
        "css" => &CSS,
        "dart" => &DART,
        "dtd" => &DTD,
        "elixir" => &ELIXIR,
        "elm" => &ELM,
        "embedded_template" => &EMBEDDED_TEMPLATE,
        "erlang" => &ERLANG,
        "fortran" => &FORTRAN,
        "fsharp" => &FSHARP,
        "haskell" => &HASKELL,
        "hcl" => &HCL,
        "html" => &HTML,
        "json" => &JSON,
        "julia" => &JULIA,
        "kotlin" => &KOTLIN,
        "lua" => &LUA,
        "make" => &MAKE,
        "markdown" => &MARKDOWN,
        "objc" => &OBJC,
        "ocaml" => &OCAML,
        "ocaml_interface" => &OCAML_INTERFACE,
        "odin" => &ODIN,
        "perl" => &PERL,
        "php" => &PHP,
        "powershell" => &POWERSHELL,
        "proto" => &PROTO,
        "r" => &R,
        "ruby" => &RUBY,
        "scala" => &SCALA,
        "solidity" => &SOLIDITY,
        "sql" => &SQL,
        "svelte" => &SVELTE,
        "swift" => &SWIFT,
        "toml" => &TOML,
        "verilog" => &VERILOG,
        "xml" => &XML,
        "yaml" => &YAML,
        "zig" => &ZIG,
        _ => &D,
    }
}

const CLASSIC_KEYS: [&str; 9] = [
    "rust",
    "python",
    "typescript",
    "tsx",
    "javascript",
    "go",
    "java",
    "c",
    "cpp",
];

fn classic_index(parser: &str) -> usize {
    CLASSIC_KEYS.iter().position(|k| *k == parser).unwrap_or(0)
}

/// The classic tables: the shared rules, plus each language's fields for the next
/// chunker steps (breadcrumbs, member groups, seams), which the engine does not read
/// yet and so leave the output unchanged.
const CLASSIC_TABLES: [Table; 9] = [
    Table {
        key: "rust",
        breadcrumbs: &[
            ("match_expression", "body"),
            ("match_arm", "value"),
            ("if_expression", "consequence"),
            ("while_expression", "body"),
            ("loop_expression", "body"),
            ("for_expression", "body"),
            ("closure_expression", "body"),
            ("function_item", "body"),
        ],
        members: &[
            "function_item",
            "function_signature_item",
            "const_item",
            "type_item",
            "associated_type",
        ],
        fields: &["field_declaration"],
        group_in: &[
            "impl_item",
            "trait_item",
            "mod_item",
            "source_file",
            "field_declaration_list",
        ],
        seams: &[
            "expression_statement",
            "let_declaration",
            "match_arm",
            "macro_invocation",
        ],
        ..CLASSIC
    },
    Table {
        key: "python",
        breadcrumbs: &[
            ("if_statement", "consequence"),
            ("elif_clause", "consequence"),
            ("else_clause", "body"),
            ("for_statement", "body"),
            ("while_statement", "body"),
            ("with_statement", "body"),
            ("try_statement", "body"),
            ("match_statement", "body"),
            ("case_clause", "consequence"),
            ("lambda", "body"),
            ("function_definition", "body"),
            ("class_definition", "body"),
        ],
        members: &["function_definition", "decorated_definition"],
        fields: &["expression_statement"],
        group_in: &["class_definition", "module"],
        seams: &[
            "expression_statement",
            "if_statement",
            "for_statement",
            "while_statement",
            "with_statement",
            "try_statement",
            "return_statement",
            "case_clause",
        ],
        ..CLASSIC
    },
    Table {
        key: "typescript",
        breadcrumbs: &[
            ("if_statement", "consequence"),
            ("for_statement", "body"),
            ("for_in_statement", "body"),
            ("while_statement", "body"),
            ("switch_statement", "body"),
            ("switch_case", ""),
            ("arrow_function", "body"),
            ("function_expression", "body"),
            ("function_declaration", "body"),
            ("method_definition", "body"),
            ("try_statement", "body"),
            ("catch_clause", "body"),
        ],
        members: &[
            "method_definition",
            "method_signature",
            "abstract_method_signature",
        ],
        fields: &["public_field_definition", "property_signature"],
        group_in: &["class_body", "interface_body", "object_type", "program"],
        seams: &[
            "expression_statement",
            "lexical_declaration",
            "variable_declaration",
            "if_statement",
            "return_statement",
            "switch_case",
            "for_statement",
        ],
        ..CLASSIC
    },
    Table {
        key: "tsx",
        breadcrumbs: &[
            ("if_statement", "consequence"),
            ("for_statement", "body"),
            ("for_in_statement", "body"),
            ("while_statement", "body"),
            ("switch_statement", "body"),
            ("switch_case", ""),
            ("arrow_function", "body"),
            ("function_expression", "body"),
            ("function_declaration", "body"),
            ("method_definition", "body"),
            ("try_statement", "body"),
            ("catch_clause", "body"),
        ],
        members: &[
            "method_definition",
            "method_signature",
            "abstract_method_signature",
        ],
        fields: &["public_field_definition", "property_signature"],
        group_in: &["class_body", "interface_body", "object_type", "program"],
        seams: &[
            "expression_statement",
            "lexical_declaration",
            "variable_declaration",
            "if_statement",
            "return_statement",
            "switch_case",
            "for_statement",
        ],
        ..CLASSIC
    },
    Table {
        key: "javascript",
        breadcrumbs: &[
            ("if_statement", "consequence"),
            ("for_statement", "body"),
            ("for_in_statement", "body"),
            ("while_statement", "body"),
            ("switch_statement", "body"),
            ("switch_case", ""),
            ("arrow_function", "body"),
            ("function_expression", "body"),
            ("function_declaration", "body"),
            ("method_definition", "body"),
            ("try_statement", "body"),
            ("catch_clause", "body"),
        ],
        members: &["method_definition"],
        fields: &["field_definition"],
        group_in: &["class_body", "program"],
        seams: &[
            "expression_statement",
            "lexical_declaration",
            "variable_declaration",
            "if_statement",
            "return_statement",
            "switch_case",
            "for_statement",
        ],
        ..CLASSIC
    },
    Table {
        key: "go",
        breadcrumbs: &[
            ("if_statement", "consequence"),
            ("for_statement", "body"),
            ("expression_case", ""),
            ("type_case", ""),
            ("communication_case", ""),
            ("select_statement", ""),
            ("func_literal", "body"),
            ("function_declaration", "body"),
            ("method_declaration", "body"),
        ],
        members: &["function_declaration", "method_declaration"],
        fields: &["field_declaration"],
        group_in: &["source_file", "field_declaration_list"],
        seams: &[
            "short_var_declaration",
            "assignment_statement",
            "expression_statement",
            "if_statement",
            "return_statement",
            "for_statement",
            "expression_case",
        ],
        ..CLASSIC
    },
    Table {
        key: "java",
        breadcrumbs: &[
            ("if_statement", "consequence"),
            ("for_statement", "body"),
            ("enhanced_for_statement", "body"),
            ("while_statement", "body"),
            ("switch_block_statement_group", ""),
            ("switch_rule", ""),
            ("lambda_expression", "body"),
            ("try_statement", "body"),
            ("method_declaration", "body"),
        ],
        members: &["method_declaration", "constructor_declaration"],
        fields: &["field_declaration"],
        group_in: &["class_body", "interface_body", "enum_body_declarations"],
        seams: &[
            "expression_statement",
            "local_variable_declaration",
            "if_statement",
            "return_statement",
            "for_statement",
            "switch_block_statement_group",
        ],
        ..CLASSIC
    },
    Table {
        key: "c",
        breadcrumbs: &[
            ("if_statement", "consequence"),
            ("for_statement", "body"),
            ("while_statement", "body"),
            ("do_statement", "body"),
            ("case_statement", ""),
            ("function_definition", "body"),
        ],
        members: &["function_definition", "declaration"],
        fields: &["field_declaration"],
        group_in: &["translation_unit", "field_declaration_list"],
        seams: &[
            "expression_statement",
            "declaration",
            "if_statement",
            "return_statement",
            "case_statement",
        ],
        ..CLASSIC
    },
    Table {
        key: "cpp",
        breadcrumbs: &[
            ("if_statement", "consequence"),
            ("for_statement", "body"),
            ("while_statement", "body"),
            ("do_statement", "body"),
            ("case_statement", ""),
            ("function_definition", "body"),
            ("lambda_expression", "body"),
            ("for_range_loop", "body"),
            ("class_specifier", "body"),
            ("namespace_definition", "body"),
        ],
        members: &["function_definition", "declaration", "template_declaration"],
        fields: &["field_declaration"],
        group_in: &[
            "translation_unit",
            "field_declaration_list",
            "declaration_list",
        ],
        seams: &[
            "expression_statement",
            "declaration",
            "if_statement",
            "return_statement",
            "case_statement",
        ],
        ..CLASSIC
    },
];

/// The original rules of the nine classic languages, one list for all of them (the
/// engine matched kind names before it had tables). Frozen: their chunk output must
/// not change.
pub(crate) const CLASSIC: Table = Table {
    key: "",
    definitions: &[
        "function_item",
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declaration",
        "constructor_declaration",
        "destructor_declaration",
        "generator_function_declaration",
        "struct_item",
        "enum_item",
        "trait_item",
        "type_item",
        "type_alias_declaration",
        "interface_declaration",
        "type_declaration",
        "class_declaration",
        "class_definition",
        "enum_declaration",
        "record_declaration",
        "decorated_definition",
        "template_declaration",
        "export_statement",
        "impl_item",
        "namespace_definition",
        "class_specifier",
        "struct_specifier",
    ],
    containers: &[
        "source_file",
        "program",
        "translation_unit",
        "module",
        "impl_item",
        "declaration_list",
        "namespace_definition",
        "class_body",
        "field_declaration_list",
        "class_declaration",
        "class_definition",
        "record_declaration",
        "class_specifier",
        "struct_specifier",
        "enum_declaration",
        "mod_item",
        "trait_item",
    ],
    attach: &["attribute_item", "inner_attribute_item"],
    wrappers: &[
        "decorated_definition",
        "template_declaration",
        "export_statement",
    ],
    bindings: &[
        ("pair", "value"),
        ("variable_declarator", "value"),
        ("assignment_expression", "right"),
    ],
    callables: &[
        "function_expression",
        "arrow_function",
        "generator_function",
        "function",
    ],
    name_fields: &["name", "key"],
    names: &[("assignment_expression", "left"), ("impl_item", "type")],
    declarators: &["function_definition", "declaration"],
    families: &[
        "function_item",
        "function_definition",
        "function_declaration",
        "method_definition",
        "method_declaration",
    ],
    compact: &[
        "class_declaration",
        "class_definition",
        "record_declaration",
    ],
    attach_blank: false,
    join_header: false,
    pack_whole_units: false,
    classic: true,
    ..D
};

/// CUDA and GLSL extend C and C++: they use the classic C++ rules and labels.
const CUDA: Table = Table {
    key: "cuda",
    breadcrumbs: &[
        ("if_statement", "consequence"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("do_statement", "body"),
        ("case_statement", ""),
        ("function_definition", "body"),
        ("lambda_expression", "body"),
        ("for_range_loop", "body"),
        ("class_specifier", "body"),
        ("namespace_definition", "body"),
    ],
    members: &["function_definition", "declaration", "template_declaration"],
    fields: &["field_declaration"],
    group_in: &[
        "translation_unit",
        "field_declaration_list",
        "declaration_list",
    ],
    seams: &[
        "expression_statement",
        "declaration",
        "if_statement",
        "return_statement",
        "case_statement",
    ],
    ..CLASSIC
};
const GLSL: Table = Table {
    key: "glsl",
    breadcrumbs: &[
        ("if_statement", "consequence"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("do_statement", "body"),
        ("case_statement", ""),
        ("function_definition", "body"),
    ],
    members: &["function_definition"],
    fields: &["field_declaration"],
    group_in: &["translation_unit"],
    seams: &[
        "expression_statement",
        "declaration",
        "if_statement",
        "return_statement",
        "case_statement",
    ],
    ..CLASSIC
};

const BASH: Table = Table {
    key: "bash",
    breadcrumbs: &[
        ("if_statement", ""),
        ("elif_clause", ""),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("case_item", ""),
        ("function_definition", "body"),
    ],
    members: &["function_definition"],
    group_in: &["program"],
    seams: &[
        "command",
        "pipeline",
        "if_statement",
        "for_statement",
        "while_statement",
        "case_item",
    ],
    definitions: &["function_definition"],
    labels: &[("function_definition", Label::Keyword("function"))],
    families: &["function_definition"],
    ..D
};

const C_SHARP: Table = Table {
    key: "c_sharp",
    breadcrumbs: &[
        ("if_statement", "consequence"),
        ("for_statement", "body"),
        ("foreach_statement", "body"),
        ("while_statement", "body"),
        ("switch_section", ""),
        ("lambda_expression", "body"),
        ("try_statement", "body"),
        ("method_declaration", "body"),
        ("local_function_statement", "body"),
    ],
    members: &[
        "method_declaration",
        "constructor_declaration",
        "property_declaration",
        "local_function_statement",
    ],
    fields: &["field_declaration", "property_declaration"],
    group_in: &["declaration_list"],
    seams: &[
        "expression_statement",
        "local_declaration_statement",
        "if_statement",
        "return_statement",
        "switch_section",
    ],
    tags: tags::C_SHARP,
    definitions: &[
        "struct_declaration",
        "record_declaration",
        "enum_declaration",
        "constructor_declaration",
        "destructor_declaration",
        "property_declaration",
        "delegate_declaration",
        "operator_declaration",
        "conversion_operator_declaration",
        "event_declaration",
        "indexer_declaration",
        "local_function_statement",
    ],
    containers: &[
        "compilation_unit",
        "namespace_declaration",
        "declaration_list",
        "class_declaration",
        "struct_declaration",
        "record_declaration",
        "interface_declaration",
    ],
    compact: &[
        "class_declaration",
        "struct_declaration",
        "record_declaration",
        "interface_declaration",
    ],
    attach: &["attribute_list"],
    labels: &[
        ("class_declaration", Label::Keyword("class")),
        ("struct_declaration", Label::Keyword("struct")),
        ("record_declaration", Label::Keyword("record")),
        ("interface_declaration", Label::Keyword("interface")),
        ("enum_declaration", Label::Keyword("enum")),
        ("namespace_declaration", Label::Keyword("namespace")),
        ("method_declaration", Label::Keyword("method")),
        ("constructor_declaration", Label::Keyword("constructor")),
        ("destructor_declaration", Label::Keyword("destructor")),
        ("property_declaration", Label::Keyword("property")),
        ("delegate_declaration", Label::Keyword("delegate")),
        ("event_declaration", Label::Keyword("event")),
        ("local_function_statement", Label::Keyword("function")),
        ("operator_declaration", Label::Header),
        ("conversion_operator_declaration", Label::Header),
        ("indexer_declaration", Label::Header),
    ],
    families: &[
        "method_declaration",
        "constructor_declaration",
        "local_function_statement",
    ],
    ..D
};

const CMAKE: Table = Table {
    key: "cmake",
    name_fields: &[],
    definitions: &["function_def", "macro_def"],
    attach: &["elseif_command", "else_command"],
    attach_back: &[
        "endfunction_command",
        "endmacro_command",
        "endif_command",
        "endforeach_command",
        "endwhile_command",
        "endblock_command",
    ],
    bodies: &[("function_def", "#body"), ("macro_def", "#body")],
    names: &[
        ("function_def", "#function_command/#argument_list/*"),
        ("macro_def", "#macro_command/#argument_list/*"),
    ],
    labels: &[
        ("function_def", Label::Keyword("function")),
        ("macro_def", Label::Keyword("macro")),
    ],
    split_small: false,
    ..D
};

const CSS: Table = Table {
    key: "css",
    definitions: &[
        "rule_set",
        "media_statement",
        "supports_statement",
        "keyframes_statement",
        "scope_statement",
        "at_rule",
    ],
    bodies: &[
        ("rule_set", "#block"),
        ("media_statement", "#block"),
        ("supports_statement", "#block"),
        ("scope_statement", "#block"),
        ("at_rule", "#block"),
        ("keyframes_statement", "#keyframe_block_list"),
    ],
    name_fields: &[],
    names: &[("rule_set", "#selectors")],
    labels: &[
        ("rule_set", Label::Header),
        ("media_statement", Label::Header),
        ("supports_statement", Label::Header),
        ("scope_statement", Label::Header),
        ("at_rule", Label::Header),
        ("keyframes_statement", Label::Header),
    ],
    split_small: false,
    ..D
};

const DART: Table = Table {
    key: "dart",
    breadcrumbs: &[
        ("if_statement", "consequence"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("switch_statement_case", ""),
        ("function_expression", "body"),
    ],
    members: &["class_member"],
    group_in: &["class_body"],
    seams: &[
        "expression_statement",
        "local_variable_declaration",
        "return_statement",
        "if_statement",
    ],
    tags: tags::DART,
    skip_tags: &[
        "constant_constructor_signature",
        "constructor_signature",
        "enum_constant",
        "external_function_declaration",
        "external_getter_declaration",
        "external_setter_declaration",
        "external_variable_declaration",
        "factory_constructor_signature",
        "method_signature",
        "redirecting_factory_constructor_signature",
        "top_level_variable_declaration",
    ],
    definitions: &["class_member"],
    containers: &[
        "class_declaration",
        "mixin_declaration",
        "extension_declaration",
        "extension_type_declaration",
    ],
    compact: &[
        "class_declaration",
        "mixin_declaration",
        "extension_declaration",
        "extension_type_declaration",
    ],
    names: &[
        ("class_member", "~name"),
        ("function_declaration", "~name"),
        ("getter_declaration", "~name"),
        ("setter_declaration", "~name"),
        ("type_alias", "#type_identifier"),
    ],
    labels: &[
        ("class_declaration", Label::Keyword("class")),
        ("mixin_declaration", Label::Keyword("mixin")),
        ("enum_declaration", Label::Keyword("enum")),
        ("extension_declaration", Label::Keyword("extension")),
        (
            "extension_type_declaration",
            Label::Keyword("extension type"),
        ),
        ("function_declaration", Label::Keyword("function")),
        ("getter_declaration", Label::Keyword("get")),
        ("setter_declaration", Label::Keyword("set")),
        ("type_alias", Label::Keyword("typedef")),
        ("class_member", Label::Name),
    ],
    families: &["function_declaration"],
    ..D
};

const DTD: Table = Table {
    key: "dtd",
    content: Content::Data,
    name_fields: &[],
    split_small: false,
    ..D
};

/// Elixir's definitions are calls (`def`, `defmodule`): keyed on the call's target.
const ELIXIR_DEFINITIONS: &[&str] = &[
    "def",
    "defp",
    "defmacro",
    "defmacrop",
    "defguard",
    "defguardp",
    "defdelegate",
    "defn",
    "defnp",
    "test",
];
const ELIXIR_MODULES: &[&str] = &["defmodule", "defprotocol", "defimpl", "describe"];
const ELIXIR: Table = Table {
    key: "elixir",
    breadcrumbs: &[("stab_clause", ""), ("anonymous_function", "")],
    seams: &["stab_clause"],
    name_fields: &[],
    keyed: &[
        Keyed {
            kind: "call",
            key: "target",
            values: ELIXIR_MODULES,
            role: Role::Container,
            key_label: true,
        },
        Keyed {
            kind: "call",
            key: "target",
            values: ELIXIR_DEFINITIONS,
            role: Role::Definition,
            key_label: true,
        },
        // Module attributes (`@doc`, `@spec`) attach to what follows.
        Keyed {
            kind: "unary_operator",
            key: "operator",
            values: &["@"],
            role: Role::Attach,
            key_label: false,
        },
    ],
    bodies: &[("call", "#do_block")],
    names: &[
        ("call", "#arguments/#call/target"),
        ("call", "#arguments/#binary_operator/left/target"),
        ("call", "#arguments/#alias"),
        ("call", "#arguments/#identifier"),
        ("call", "#arguments/#string"),
    ],
    compact: &["call"],
    merge_clauses: true,
    ..D
};

const ELM: Table = Table {
    key: "elm",
    tags: tags::ELM,
    skip_tags: &["module_declaration"],
    definitions: &["type_alias_declaration", "port_annotation"],
    attach: &["type_annotation"],
    names: &[("value_declaration", "functionDeclarationLeft/*")],
    labels: &[
        ("value_declaration", Label::Name),
        ("type_declaration", Label::Keyword("type")),
        ("type_alias_declaration", Label::Keyword("type alias")),
        ("port_annotation", Label::Keyword("port")),
    ],
    families: &["value_declaration"],
    ..D
};

const EMBEDDED_TEMPLATE: Table = Table {
    key: "embedded_template",
    content: Content::Prose,
    name_fields: &[],
    split_small: false,
    ..D
};

const ERLANG: Table = Table {
    key: "erlang",
    breadcrumbs: &[
        ("case_expr", ""),
        ("cr_clause", ""),
        ("if_expr", ""),
        ("anonymous_fun", ""),
    ],
    members: &["fun_decl"],
    fields: &["record_decl"],
    seams: &["cr_clause", "function_clause"],
    definitions: &[
        "fun_decl",
        "record_decl",
        "type_alias",
        "opaque",
        "pp_define",
    ],
    attach: &["spec", "wild_attribute"],
    names: &[("fun_decl", "clause/name"), ("pp_define", "lhs")],
    labels: &[
        ("fun_decl", Label::Name),
        ("record_decl", Label::Keyword("record")),
        ("type_alias", Label::Keyword("type")),
        ("opaque", Label::Keyword("opaque")),
        ("pp_define", Label::Keyword("define")),
    ],
    families: &["fun_decl"],
    merge_clauses: true,
    ..D
};

const FORTRAN: Table = Table {
    key: "fortran",
    definitions: &[
        "module",
        "submodule",
        "program",
        "function",
        "subroutine",
        "derived_type_definition",
        "interface",
    ],
    containers: &["module", "submodule", "internal_procedures"],
    attach: &["contains_statement"],
    attach_back: &[
        "end_module_statement",
        "end_submodule_statement",
        "end_program_statement",
        "end_function_statement",
        "end_subroutine_statement",
        "end_type_statement",
        "end_interface_statement",
    ],
    names: &[
        ("module", "#module_statement/#name"),
        ("submodule", "#submodule_statement/#name"),
        ("program", "#program_statement/#name"),
        ("function", "#function_statement/name"),
        ("subroutine", "#subroutine_statement/name"),
        (
            "derived_type_definition",
            "#derived_type_statement/#type_name",
        ),
        ("interface", "#interface_statement/#name"),
    ],
    labels: &[
        ("module", Label::Keyword("module")),
        ("submodule", Label::Keyword("submodule")),
        ("program", Label::Keyword("program")),
        ("function", Label::Keyword("function")),
        ("subroutine", Label::Keyword("subroutine")),
        ("derived_type_definition", Label::Keyword("type")),
        ("interface", Label::Keyword("interface")),
    ],
    families: &["function", "subroutine"],
    ..D
};

const FSHARP: Table = Table {
    key: "fsharp",
    definitions: &[
        "module_defn",
        "type_definition",
        "function_or_value_defn",
        "member_defn",
        "exception_definition",
    ],
    containers: &["namespace", "named_module", "module_defn"],
    attach: &["xml_doc", "attributes"],
    names: &[
        ("module_defn", "#identifier"),
        ("type_definition", "~type_name"),
        ("type_definition", "*/#type_name"),
        ("function_or_value_defn", "#function_declaration_left/*"),
        ("function_or_value_defn", "#value_declaration_left/*"),
        ("member_defn", "#method_or_prop_defn/name"),
        ("exception_definition", "exception_name"),
    ],
    labels: &[
        ("namespace", Label::Keyword("namespace")),
        ("named_module", Label::Keyword("module")),
        ("module_defn", Label::Keyword("module")),
        ("type_definition", Label::Keyword("type")),
        ("function_or_value_defn", Label::Keyword("let")),
        ("member_defn", Label::Keyword("member")),
        ("exception_definition", Label::Keyword("exception")),
    ],
    families: &["function_or_value_defn", "member_defn"],
    ..D
};

const HASKELL: Table = Table {
    key: "haskell",
    definitions: &[
        "function",
        "bind",
        "data_type",
        "newtype",
        "type_synonym",
        "type_family",
        "data_family",
        "class",
        "instance",
        "foreign_import",
        "pattern_synonym",
    ],
    containers: &["declarations", "class", "instance"],
    compact: &["class", "instance"],
    attach: &["signature", "haddock", "pragma"],
    bodies: &[("class", "declarations"), ("instance", "declarations")],
    labels: &[
        ("function", Label::Name),
        ("bind", Label::Name),
        ("data_type", Label::Keyword("data")),
        ("newtype", Label::Keyword("newtype")),
        ("type_synonym", Label::Keyword("type")),
        ("type_family", Label::Keyword("type family")),
        ("data_family", Label::Keyword("data family")),
        ("class", Label::Keyword("class")),
        ("instance", Label::Header),
    ],
    families: &["function"],
    merge_clauses: true,
    ..D
};

const HCL: Table = Table {
    key: "hcl",
    content: Content::Data,
    name_fields: &[],
    definitions: &["block"],
    bodies: &[("block", "#body")],
    names: &[("block", "#identifier")],
    labels: &[("block", Label::Header)],
    ..D
};

const HTML: Table = Table {
    key: "html",
    content: Content::Prose,
    name_fields: &[],
    atomic: &["script_element", "style_element"],
    attach_back: &["end_tag"],
    labels: &[
        ("element", Label::Path("#start_tag")),
        ("script_element", Label::Path("#start_tag")),
        ("style_element", Label::Path("#start_tag")),
    ],
    split_small: false,
    ..D
};

const JSON: Table = Table {
    key: "json",
    content: Content::Data,
    bodies: &[("pair", "value")],
    name_fields: &["key"],
    labels: &[("pair", Label::Name)],
    split_small: false,
    ..D
};

const JULIA: Table = Table {
    key: "julia",
    breadcrumbs: &[
        ("if_statement", ""),
        ("for_statement", ""),
        ("while_statement", ""),
        ("do_clause", ""),
        ("let_statement", ""),
        ("try_statement", ""),
    ],
    members: &["function_definition", "assignment"],
    group_in: &["module_definition"],
    seams: &["if_statement", "for_statement"],
    definitions: &[
        "function_definition",
        "macro_definition",
        "struct_definition",
        "abstract_definition",
        "primitive_definition",
        "module_definition",
    ],
    containers: &["module_definition"],
    attach: &["string_literal"],
    // `f(x) = x + 1` defines a function: an assignment whose left side is a call.
    bindings: &[("assignment", "*")],
    callables: &["call_expression"],
    names: &[
        ("function_definition", "#signature/#call_expression/*"),
        (
            "function_definition",
            "#signature/#typed_expression/#call_expression/*",
        ),
        ("function_definition", "#signature/#identifier"),
        ("macro_definition", "#signature/#call_expression/*"),
        ("struct_definition", "#type_head"),
        ("abstract_definition", "#type_head"),
        ("primitive_definition", "#type_head"),
        ("assignment", "#call_expression/*"),
    ],
    labels: &[
        ("function_definition", Label::Keyword("function")),
        ("macro_definition", Label::Keyword("macro")),
        ("struct_definition", Label::Keyword("struct")),
        ("abstract_definition", Label::Keyword("abstract type")),
        ("primitive_definition", Label::Keyword("primitive type")),
        ("module_definition", Label::Keyword("module")),
        ("assignment", Label::Keyword("function")),
    ],
    families: &["function_definition", "assignment"],
    ..D
};

const KOTLIN: Table = Table {
    key: "kotlin",
    breadcrumbs: &[
        ("if_expression", ""),
        ("when_entry", ""),
        ("for_statement", ""),
        ("while_statement", ""),
        ("lambda_literal", ""),
        ("function_declaration", ""),
    ],
    members: &["function_declaration", "property_declaration"],
    fields: &["property_declaration"],
    group_in: &["class_body"],
    seams: &["statement", "when_entry"],
    definitions: &[
        "class_declaration",
        "object_declaration",
        "companion_object",
        "function_declaration",
        "secondary_constructor",
        "type_alias",
    ],
    containers: &[
        "class_declaration",
        "object_declaration",
        "companion_object",
    ],
    compact: &[
        "class_declaration",
        "object_declaration",
        "companion_object",
    ],
    attach: &["annotated_expression", "file_annotation"],
    names: &[("type_alias", "type")],
    labels: &[
        ("class_declaration", Label::Header),
        ("object_declaration", Label::Keyword("object")),
        ("companion_object", Label::Header),
        ("function_declaration", Label::Keyword("fun")),
        ("secondary_constructor", Label::Header),
        ("type_alias", Label::Keyword("typealias")),
    ],
    bodies: &[
        ("class_declaration", "#class_body|enum_class_body"),
        ("object_declaration", "#class_body"),
        ("companion_object", "#class_body"),
    ],
    families: &["function_declaration"],
    ..D
};

const LUA: Table = Table {
    key: "lua",
    breadcrumbs: &[
        ("if_statement", "consequence"),
        ("elseif_statement", "consequence"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("repeat_statement", "body"),
        ("function_definition", "body"),
        ("function_declaration", "body"),
    ],
    members: &["function_declaration"],
    fields: &["field"],
    group_in: &["table_constructor", "chunk"],
    seams: &[
        "function_call",
        "assignment_statement",
        "variable_declaration",
        "return_statement",
        "if_statement",
    ],
    definitions: &["function_declaration"],
    // `local f = function () end`, `M.f = function () end`, `{ f = function () end }`
    bindings: &[
        (
            "variable_declaration",
            "#assignment_statement/#expression_list/*",
        ),
        ("assignment_statement", "#expression_list/*"),
        ("field", "value"),
    ],
    callables: &["function_definition"],
    names: &[
        (
            "variable_declaration",
            "#assignment_statement/#variable_list/*",
        ),
        ("assignment_statement", "#variable_list/*"),
    ],
    labels: &[
        ("function_declaration", Label::Keyword("function")),
        ("variable_declaration", Label::Keyword("function")),
        ("assignment_statement", Label::Keyword("function")),
        ("field", Label::Keyword("function")),
    ],
    families: &["function_declaration"],
    ..D
};

const MAKE: Table = Table {
    key: "make",
    definitions: &["rule", "define_directive"],
    names: &[("rule", "#targets")],
    labels: &[
        ("rule", Label::Name),
        ("define_directive", Label::Keyword("define")),
    ],
    split_small: false,
    ..D
};

const MARKDOWN: Table = Table {
    key: "markdown",
    group_in: &["section"],
    seams: &[
        "paragraph",
        "list",
        "list_item",
        "fenced_code_block",
        "pipe_table",
        "block_quote",
        "html_block",
    ],
    content: Content::Prose,
    definitions: &["section"],
    // A heading stays with the content after it; the front matter (when it is not
    // the file's metadata region) with the first section.
    attach: &[
        "atx_heading",
        "setext_heading",
        "minus_metadata",
        "plus_metadata",
    ],
    name_fields: &[],
    names: &[("section", "#atx_heading/heading_content")],
    labels: &[("section", Label::Name)],
    split_small: false,
    ..D
};

const OBJC: Table = Table {
    key: "objc",
    breadcrumbs: &[
        ("if_statement", "consequence"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("do_statement", "body"),
        ("case_statement", ""),
        ("function_definition", "body"),
    ],
    members: &[
        "method_definition",
        "method_declaration",
        "function_definition",
    ],
    fields: &["property_declaration"],
    group_in: &[
        "class_interface",
        "class_implementation",
        "protocol_declaration",
        "translation_unit",
    ],
    seams: &[
        "expression_statement",
        "declaration",
        "if_statement",
        "return_statement",
        "case_statement",
    ],
    definitions: &[
        "class_interface",
        "class_implementation",
        "protocol_declaration",
        "implementation_definition",
        "method_definition",
        "method_declaration",
        "function_definition",
        "struct_specifier",
        "enum_specifier",
    ],
    containers: &[
        "translation_unit",
        "class_interface",
        "class_implementation",
        "protocol_declaration",
    ],
    compact: &[
        "class_interface",
        "class_implementation",
        "protocol_declaration",
    ],
    wrappers: &["implementation_definition"],
    names: &[
        ("class_interface", "#identifier"),
        ("class_implementation", "#identifier"),
        ("protocol_declaration", "#identifier"),
        ("method_definition", "#identifier"),
        ("method_definition", "#keyword_declarator/#identifier"),
        ("method_declaration", "#identifier"),
        ("method_declaration", "#keyword_declarator/#identifier"),
    ],
    declarators: &["function_definition", "declaration"],
    labels: &[
        ("class_interface", Label::Keyword("@interface")),
        ("class_implementation", Label::Keyword("@implementation")),
        ("protocol_declaration", Label::Keyword("@protocol")),
        ("method_definition", Label::Header),
        ("method_declaration", Label::Header),
        ("function_definition", Label::Keyword("function")),
        ("struct_specifier", Label::Keyword("struct")),
        ("enum_specifier", Label::Keyword("enum")),
    ],
    families: &["method_definition", "function_definition"],
    ..D
};

const OCAML_DEFINITIONS: &[&str] = &[
    "value_definition",
    "type_definition",
    "module_definition",
    "module_type_definition",
    "class_definition",
    "class_type_definition",
    "exception_definition",
    "external",
    "value_specification",
];
const OCAML_NAMES: &[(&str, &str)] = &[
    ("value_definition", "#let_binding/pattern"),
    ("type_definition", "#type_binding/name"),
    ("module_definition", "#module_binding/#module_name"),
    ("module_type_definition", "#module_type_name"),
    ("class_definition", "#class_binding/#class_name"),
    (
        "class_type_definition",
        "#class_type_binding/#class_type_name",
    ),
    ("exception_definition", "#constructor_declaration"),
    ("external", "#value_name"),
    ("value_specification", "#value_name"),
];
const OCAML_LABELS: &[(&str, Label)] = &[
    ("value_definition", Label::Keyword("let")),
    ("type_definition", Label::Keyword("type")),
    ("module_definition", Label::Keyword("module")),
    ("module_type_definition", Label::Keyword("module type")),
    ("class_definition", Label::Keyword("class")),
    ("class_type_definition", Label::Keyword("class type")),
    ("exception_definition", Label::Keyword("exception")),
    ("external", Label::Keyword("external")),
    ("value_specification", Label::Keyword("val")),
];
const OCAML: Table = Table {
    key: "ocaml",
    breadcrumbs: &[
        ("match_expression", ""),
        ("match_case", ""),
        ("let_expression", ""),
        ("fun_expression", ""),
        ("if_expression", ""),
    ],
    members: &["value_definition"],
    group_in: &["structure"],
    seams: &["match_case"],
    definitions: OCAML_DEFINITIONS,
    containers: &["module_definition", "module_type_definition"],
    compact: &["module_definition", "module_type_definition"],
    bodies: &[
        ("module_definition", "#module_binding/body"),
        ("class_definition", "#class_binding/body"),
    ],
    names: OCAML_NAMES,
    labels: OCAML_LABELS,
    families: &["value_definition"],
    ..D
};
const OCAML_INTERFACE: Table = Table {
    key: "ocaml_interface",
    ..OCAML
};

const ODIN: Table = Table {
    key: "odin",
    name_fields: &[],
    definitions: &[
        "procedure_declaration",
        "overloaded_procedure_declaration",
        "struct_declaration",
        "union_declaration",
        "enum_declaration",
        "bit_field_declaration",
    ],
    names: &[
        ("procedure_declaration", "#identifier"),
        ("overloaded_procedure_declaration", "#identifier"),
        ("struct_declaration", "#identifier"),
        ("union_declaration", "#identifier"),
        ("enum_declaration", "#identifier"),
        ("bit_field_declaration", "#identifier"),
    ],
    labels: &[
        ("procedure_declaration", Label::Keyword("proc")),
        ("overloaded_procedure_declaration", Label::Keyword("proc")),
        ("struct_declaration", Label::Keyword("struct")),
        ("union_declaration", Label::Keyword("union")),
        ("enum_declaration", Label::Keyword("enum")),
        ("bit_field_declaration", Label::Keyword("bit_field")),
    ],
    bodies: &[("procedure_declaration", "#procedure/#block")],
    families: &["procedure_declaration"],
    ..D
};

const PERL: Table = Table {
    key: "perl",
    breadcrumbs: &[
        ("conditional_statement", "block"),
        ("for_statement", "block"),
        ("cstyle_for_statement", "block"),
        ("anonymous_subroutine_expression", "body"),
        ("subroutine_declaration_statement", "body"),
    ],
    members: &["subroutine_declaration_statement"],
    seams: &["expression_statement", "conditional_statement"],
    definitions: &["subroutine_declaration_statement"],
    attach: &["pod"],
    labels: &[("subroutine_declaration_statement", Label::Keyword("sub"))],
    families: &["subroutine_declaration_statement"],
    ..D
};

const PHP: Table = Table {
    key: "php",
    breadcrumbs: &[
        ("if_statement", "body"),
        ("foreach_statement", "body"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("case_statement", ""),
        ("anonymous_function", "body"),
        ("arrow_function", "body"),
        ("method_declaration", "body"),
    ],
    members: &["method_declaration"],
    fields: &["property_declaration"],
    group_in: &["declaration_list"],
    seams: &[
        "expression_statement",
        "return_statement",
        "if_statement",
        "case_statement",
    ],
    tags: tags::PHP,
    skip_tags: &["property_declaration"],
    definitions: &["enum_declaration"],
    containers: &[
        "program",
        "namespace_definition",
        "declaration_list",
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
        "enum_declaration",
    ],
    compact: &[
        "class_declaration",
        "interface_declaration",
        "trait_declaration",
        "enum_declaration",
    ],
    attach: &["attribute_list"],
    labels: &[
        ("class_declaration", Label::Keyword("class")),
        ("interface_declaration", Label::Keyword("interface")),
        ("trait_declaration", Label::Keyword("trait")),
        ("enum_declaration", Label::Keyword("enum")),
        ("namespace_definition", Label::Keyword("namespace")),
        ("function_definition", Label::Keyword("function")),
        ("method_declaration", Label::Keyword("method")),
    ],
    families: &["function_definition", "method_declaration"],
    ..D
};

const POWERSHELL: Table = Table {
    key: "powershell",
    breadcrumbs: &[
        ("if_statement", ""),
        ("foreach_statement", ""),
        ("while_statement", ""),
        ("switch_clause", ""),
        ("function_statement", ""),
    ],
    members: &["function_statement", "class_method_definition"],
    fields: &["class_property_definition"],
    group_in: &["class_statement"],
    seams: &["pipeline", "if_statement", "foreach_statement"],
    name_fields: &[],
    definitions: &[
        "function_statement",
        "class_statement",
        "class_method_definition",
        "enum_statement",
    ],
    containers: &["class_statement"],
    compact: &["class_statement"],
    names: &[
        ("function_statement", "#function_name"),
        ("class_statement", "#simple_name"),
        ("class_method_definition", "#simple_name"),
        ("enum_statement", "#simple_name"),
    ],
    labels: &[
        ("function_statement", Label::Keyword("function")),
        ("class_statement", Label::Keyword("class")),
        ("class_method_definition", Label::Keyword("method")),
        ("enum_statement", Label::Keyword("enum")),
    ],
    families: &["function_statement", "class_method_definition"],
    ..D
};

const PROTO: Table = Table {
    key: "proto",
    definitions: &["message", "enum", "service", "rpc", "extend"],
    bodies: &[("message", "#message_body"), ("enum", "#enum_body")],
    names: &[
        ("message", "#message_name"),
        ("enum", "#enum_name"),
        ("service", "#service_name"),
        ("rpc", "#rpc_name"),
    ],
    labels: &[
        ("message", Label::Keyword("message")),
        ("enum", Label::Keyword("enum")),
        ("service", Label::Keyword("service")),
        ("rpc", Label::Keyword("rpc")),
        ("extend", Label::Header),
    ],
    ..D
};

const R: Table = Table {
    key: "r",
    breadcrumbs: &[
        ("if_statement", "consequence"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("function_definition", "body"),
    ],
    members: &["binary_operator"],
    seams: &["braced_expression"],
    // `f <- function(x) ...`
    bindings: &[("binary_operator", "rhs")],
    callables: &["function_definition"],
    names: &[("binary_operator", "lhs")],
    labels: &[("binary_operator", Label::Keyword("function"))],
    families: &["binary_operator"],
    ..D
};

const RUBY: Table = Table {
    key: "ruby",
    breadcrumbs: &[
        ("if", "consequence"),
        ("elsif", "consequence"),
        ("unless", "consequence"),
        ("while", "body"),
        ("for", "body"),
        ("when", ""),
        ("block", "body"),
        ("do_block", "body"),
        ("method", "body"),
        ("singleton_method", "body"),
    ],
    members: &["method", "singleton_method"],
    fields: &["call", "assignment"],
    group_in: &["body_statement"],
    seams: &["call", "assignment", "if", "when"],
    tags: tags::RUBY,
    skip_tags: &["alias"],
    containers: &["class", "module", "singleton_class"],
    compact: &["class", "singleton_class"],
    labels: &[
        ("class", Label::Keyword("class")),
        ("module", Label::Keyword("module")),
        ("singleton_class", Label::Header),
        ("method", Label::Keyword("def")),
        ("singleton_method", Label::Header),
    ],
    families: &["method", "singleton_method"],
    ..D
};

const SCALA: Table = Table {
    key: "scala",
    breadcrumbs: &[
        ("if_expression", "consequence"),
        ("case_clause", "body"),
        ("for_expression", "body"),
        ("while_expression", "body"),
        ("lambda_expression", ""),
        ("function_definition", "body"),
    ],
    members: &[
        "function_definition",
        "function_declaration",
        "val_definition",
        "var_definition",
    ],
    fields: &["val_definition", "var_definition"],
    group_in: &["template_body"],
    seams: &["case_clause"],
    tags: tags::SCALA,
    skip_tags: &[
        "class_parameter",
        "val_declaration",
        "val_definition",
        "var_declaration",
        "var_definition",
        "simple_enum_case",
        "full_enum_case",
        "package_clause",
    ],
    definitions: &["function_declaration", "extension_definition"],
    containers: &[
        "class_definition",
        "object_definition",
        "trait_definition",
        "enum_definition",
    ],
    compact: &[
        "class_definition",
        "object_definition",
        "trait_definition",
        "enum_definition",
    ],
    labels: &[
        ("class_definition", Label::Keyword("class")),
        ("object_definition", Label::Keyword("object")),
        ("trait_definition", Label::Keyword("trait")),
        ("enum_definition", Label::Keyword("enum")),
        ("function_definition", Label::Keyword("def")),
        ("function_declaration", Label::Keyword("def")),
        ("given_definition", Label::Header),
        ("type_definition", Label::Keyword("type")),
        ("extension_definition", Label::Header),
    ],
    families: &["function_definition"],
    ..D
};

const SOLIDITY: Table = Table {
    key: "solidity",
    breadcrumbs: &[
        ("if_statement", "body"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("function_definition", "body"),
        ("modifier_definition", "body"),
    ],
    members: &["function_definition", "modifier_definition"],
    fields: &["state_variable_declaration"],
    group_in: &["contract_body"],
    seams: &[
        "expression_statement",
        "variable_declaration_statement",
        "return_statement",
        "if_statement",
    ],
    tags: tags::SOLIDITY,
    skip_tags: &["event_definition"],
    definitions: &[
        "modifier_definition",
        "constructor_definition",
        "fallback_receive_definition",
    ],
    containers: &[
        "contract_declaration",
        "interface_declaration",
        "library_declaration",
    ],
    compact: &[
        "contract_declaration",
        "interface_declaration",
        "library_declaration",
    ],
    labels: &[
        ("contract_declaration", Label::Keyword("contract")),
        ("interface_declaration", Label::Keyword("interface")),
        ("library_declaration", Label::Keyword("library")),
        ("function_definition", Label::Keyword("function")),
        ("modifier_definition", Label::Keyword("modifier")),
        ("constructor_definition", Label::Header),
        ("fallback_receive_definition", Label::Header),
        ("struct_declaration", Label::Keyword("struct")),
        ("enum_declaration", Label::Keyword("enum")),
    ],
    families: &["function_definition"],
    ..D
};

const SQL: Table = Table {
    key: "sql",
    // A statement that creates something (`CREATE TABLE`, `CREATE FUNCTION`) is a
    // definition; the others pack together.
    bindings: &[("statement", "*")],
    callables: &[
        "create_table",
        "create_view",
        "create_materialized_view",
        "create_function",
        "create_type",
        "create_trigger",
        "create_schema",
        "create_sequence",
        "create_database",
    ],
    attach: &["marginalia"],
    name_fields: &[],
    names: &[("statement", "*/#object_reference")],
    labels: &[("statement", Label::Line)],
    split_small: false,
    ..D
};

const SVELTE: Table = Table {
    key: "svelte",
    name_fields: &[],
    atomic: &["script_element", "style_element"],
    attach_back: &[
        "end_tag",
        "if_end",
        "each_end",
        "await_end",
        "key_end",
        "snippet_end",
    ],
    labels: &[
        ("element", Label::Path("#start_tag")),
        ("script_element", Label::Path("#start_tag")),
        ("style_element", Label::Path("#start_tag")),
    ],
    split_small: false,
    ..D
};

const SWIFT: Table = Table {
    key: "swift",
    breadcrumbs: &[
        ("if_statement", ""),
        ("guard_statement", ""),
        ("for_statement", ""),
        ("while_statement", ""),
        ("switch_entry", ""),
        ("lambda_literal", ""),
        ("function_declaration", "body"),
    ],
    members: &[
        "function_declaration",
        "init_declaration",
        "property_declaration",
    ],
    fields: &["property_declaration"],
    group_in: &["class_body", "protocol_body"],
    seams: &["statements", "switch_entry"],
    tags: tags::SWIFT,
    skip_tags: &["property_declaration"],
    definitions: &[
        "init_declaration",
        "deinit_declaration",
        "subscript_declaration",
        "typealias_declaration",
    ],
    containers: &["class_declaration", "protocol_declaration"],
    compact: &["class_declaration", "protocol_declaration"],
    labels: &[
        ("class_declaration", Label::Header),
        ("protocol_declaration", Label::Header),
        ("function_declaration", Label::Keyword("func")),
        ("init_declaration", Label::Header),
        ("deinit_declaration", Label::Header),
        ("subscript_declaration", Label::Header),
        ("typealias_declaration", Label::Keyword("typealias")),
    ],
    families: &["function_declaration", "init_declaration"],
    ..D
};

const TOML: Table = Table {
    key: "toml",
    content: Content::Data,
    definitions: &["table", "table_array_element"],
    name_fields: &[],
    names: &[
        ("table", "#bare_key|dotted_key|quoted_key"),
        ("table_array_element", "#bare_key|dotted_key|quoted_key"),
        ("pair", "#bare_key|dotted_key|quoted_key"),
    ],
    labels: &[
        ("table", Label::Name),
        ("table_array_element", Label::Name),
        ("pair", Label::Name),
    ],
    split_small: false,
    ..D
};

const VERILOG: Table = Table {
    key: "verilog",
    name_fields: &[],
    definitions: &[
        "module_declaration",
        "interface_declaration",
        "package_declaration",
        "class_declaration",
        "function_declaration",
        "task_declaration",
    ],
    containers: &[
        "module_declaration",
        "interface_declaration",
        "package_declaration",
    ],
    compact: &[
        "module_declaration",
        "interface_declaration",
        "package_declaration",
    ],
    names: &[
        ("module_declaration", "#module_header/#simple_identifier"),
        ("class_declaration", "#class_identifier"),
        (
            "function_declaration",
            "#function_body_declaration/#function_identifier",
        ),
        (
            "task_declaration",
            "#task_body_declaration/#task_identifier",
        ),
    ],
    labels: &[
        ("module_declaration", Label::Keyword("module")),
        ("interface_declaration", Label::Line),
        ("package_declaration", Label::Line),
        ("class_declaration", Label::Keyword("class")),
        ("function_declaration", Label::Keyword("function")),
        ("task_declaration", Label::Keyword("task")),
    ],
    families: &["function_declaration", "task_declaration"],
    ..D
};

const XML: Table = Table {
    key: "xml",
    content: Content::Data,
    name_fields: &[],
    attach: &["Comment"],
    attach_back: &["ETag"],
    bodies: &[("element", "#content")],
    labels: &[("element", Label::Path("#STag"))],
    split_small: false,
    ..D
};

const YAML: Table = Table {
    key: "yaml",
    content: Content::Data,
    bodies: &[("block_mapping_pair", "value"), ("flow_pair", "value")],
    name_fields: &["key"],
    labels: &[
        ("block_mapping_pair", Label::Name),
        ("flow_pair", Label::Name),
    ],
    split_small: false,
    ..D
};

const ZIG_TYPES: &[&str] = &[
    "struct_declaration",
    "union_declaration",
    "enum_declaration",
    "opaque_declaration",
];
const ZIG: Table = Table {
    key: "zig",
    breadcrumbs: &[
        ("if_statement", "body"),
        ("for_statement", "body"),
        ("while_statement", "body"),
        ("switch_case", ""),
        ("function_declaration", "body"),
    ],
    members: &["function_declaration"],
    fields: &["container_field"],
    seams: &["expression_statement", "variable_declaration"],
    definitions: &["function_declaration", "test_declaration"],
    // `const Point = struct { ... }`
    bindings: &[(
        "variable_declaration",
        "#struct_declaration|union_declaration|enum_declaration|opaque_declaration",
    )],
    callables: ZIG_TYPES,
    bodies: &[(
        "variable_declaration",
        "#struct_declaration|union_declaration|enum_declaration|opaque_declaration",
    )],
    names: &[("variable_declaration", "#identifier")],
    labels: &[
        ("function_declaration", Label::Keyword("fn")),
        ("test_declaration", Label::Line),
        ("variable_declaration", Label::Keyword("const")),
    ],
    families: &["function_declaration"],
    ..D
};
