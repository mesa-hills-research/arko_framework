//! What each grammar's `queries/tags.scm` marks as a definition: `(kind, category)` for
//! every `@definition.<category>` capture. Written by `tools/gen_language_rules.py`
//! from the grammars' `tags.scm` and `node-types.json`: do not edit by hand. The
//! curated rules in `mod.rs` start from these lists.

/// `c_sharp`: tags.scm definitions.
pub(super) const C_SHARP: &[(&str, &str)] = &[
    ("class_declaration", "class"),
    ("interface_declaration", "interface"),
    ("method_declaration", "method"),
    ("namespace_declaration", "module"),
];
/// `dart`: tags.scm definitions.
pub(super) const DART: &[(&str, &str)] = &[
    ("class_declaration", "class"),
    ("constant_constructor_signature", "method"),
    ("constructor_signature", "method"),
    ("enum_constant", "constant"),
    ("enum_declaration", "class"),
    ("extension_declaration", "class"),
    ("extension_type_declaration", "class"),
    ("external_function_declaration", "function"),
    ("external_getter_declaration", "function"),
    ("external_setter_declaration", "function"),
    ("external_variable_declaration", "variable"),
    ("factory_constructor_signature", "method"),
    ("function_declaration", "function"),
    ("getter_declaration", "function"),
    ("method_signature", "method"),
    ("mixin_declaration", "class"),
    ("redirecting_factory_constructor_signature", "method"),
    ("setter_declaration", "function"),
    ("top_level_variable_declaration", "variable"),
    ("type_alias", "type"),
];
/// `elixir`: tags.scm definitions.
pub(super) const ELIXIR: &[(&str, &str)] = &[("call", "function"), ("call", "module")];
/// `elm`: tags.scm definitions.
pub(super) const ELM: &[(&str, &str)] = &[
    ("module_declaration", "module"),
    ("type_declaration", "type"),
    ("type_declaration", "union"),
    ("value_declaration", "function"),
];
/// `fortran`: tags.scm definitions.
pub(super) const FORTRAN: &[(&str, &str)] = &[
    ("derived_type_statement", "class"),
    ("function_statement", "function"),
    ("function_statement", "interface"),
    ("module_statement", "module"),
    ("program_statement", "module"),
    ("submodule_statement", "module"),
    ("subroutine_statement", "function"),
    ("subroutine_statement", "interface"),
];
/// `fsharp`: tags.scm definitions.
pub(super) const FSHARP: &[(&str, &str)] = &[
    ("exception_definition", "class"),
    ("file", "constant"),
    ("file", "function"),
    ("member_defn", "method"),
    ("module_defn", "constant"),
    ("module_defn", "function"),
    ("module_defn", "module"),
    ("named_module", "constant"),
    ("named_module", "function"),
    ("named_module", "module"),
    ("namespace", "constant"),
    ("namespace", "function"),
    ("namespace", "module"),
    ("type_definition", "class"),
];
/// `lua`: tags.scm definitions.
pub(super) const LUA: &[(&str, &str)] = &[
    ("assignment_statement", "function"),
    ("function_declaration", "function"),
    ("function_declaration", "method"),
    ("table_constructor", "function"),
];
/// `ocaml`: tags.scm definitions.
pub(super) const OCAML: &[(&str, &str)] = &[
    ("class_binding", "class"),
    ("class_type_binding", "class"),
    ("constructor_declaration", "enum_variant"),
    ("external", "function"),
    ("field_declaration", "field"),
    ("let_binding", "function"),
    ("let_binding", "operator"),
    ("method_definition", "method"),
    ("module_binding", "module"),
    ("module_type_definition", "interface"),
    ("tag_specification", "enum_variant"),
    ("type_binding", "type"),
];
/// `ocaml_interface`: tags.scm definitions.
pub(super) const OCAML_INTERFACE: &[(&str, &str)] = &[
    ("class_binding", "class"),
    ("class_type_binding", "class"),
    ("constructor_declaration", "enum_variant"),
    ("external", "function"),
    ("field_declaration", "field"),
    ("let_binding", "function"),
    ("let_binding", "operator"),
    ("method_definition", "method"),
    ("module_binding", "module"),
    ("module_type_definition", "interface"),
    ("tag_specification", "enum_variant"),
    ("type_binding", "type"),
];
/// `php`: tags.scm definitions.
pub(super) const PHP: &[(&str, &str)] = &[
    ("class_declaration", "class"),
    ("function_definition", "function"),
    ("interface_declaration", "interface"),
    ("method_declaration", "function"),
    ("namespace_definition", "module"),
    ("property_declaration", "field"),
    ("trait_declaration", "interface"),
];
/// `r`: tags.scm definitions.
pub(super) const R: &[(&str, &str)] = &[("binary_operator", "function")];
/// `ruby`: tags.scm definitions.
pub(super) const RUBY: &[(&str, &str)] = &[
    ("alias", "method"),
    ("class", "class"),
    ("method", "method"),
    ("module", "module"),
    ("singleton_class", "class"),
    ("singleton_method", "method"),
];
/// `scala`: tags.scm definitions.
pub(super) const SCALA: &[(&str, &str)] = &[
    ("class_definition", "class"),
    ("class_parameter", "property"),
    ("enum_definition", "enum"),
    ("full_enum_case", "class"),
    ("function_definition", "function"),
    ("given_definition", "variable"),
    ("object_definition", "object"),
    ("package_clause", "module"),
    ("simple_enum_case", "class"),
    ("trait_definition", "interface"),
    ("type_definition", "type"),
    ("val_declaration", "variable"),
    ("val_definition", "variable"),
    ("var_declaration", "variable"),
    ("var_definition", "variable"),
];
/// `solidity`: tags.scm definitions.
pub(super) const SOLIDITY: &[(&str, &str)] = &[
    ("contract_declaration", "class"),
    ("enum_declaration", "class"),
    ("event_definition", "class"),
    ("function_definition", "function"),
    ("function_definition", "method"),
    ("interface_declaration", "interface"),
    ("library_declaration", "interface"),
    ("struct_declaration", "class"),
];
/// `swift`: tags.scm definitions.
pub(super) const SWIFT: &[(&str, &str)] = &[
    ("class_declaration", "class"),
    ("class_declaration", "method"),
    ("class_declaration", "property"),
    ("function_declaration", "function"),
    ("property_declaration", "property"),
    ("protocol_declaration", "interface"),
    ("protocol_declaration", "method"),
];
