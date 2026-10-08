//! Which grammar parses a file: by extension and well-known file names, with a content
//! check for the ambiguous `.pl` and `.t`. See `docs/arko_index-languages.md`.
use std::path::Path;
use tree_sitter::Language;

/// A file's language: `name` is the chunks' `language` field, `parser` the grammar key
/// (`tsx` for TypeScript's `.tsx`, `cpp` for C's `.h`, `ocaml_interface` for `.mli`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Detected {
    pub(crate) name: &'static str,
    pub(crate) parser: &'static str,
}

const TEXT: Detected = Detected {
    name: "text",
    parser: "text",
};

const fn same(name: &'static str) -> Detected {
    Detected { name, parser: name }
}

/// The languages arko_index chunked before the other grammars were added, with their
/// original extensions. Their rules and output are frozen (`rules::CLASSIC`).
fn classic(extension: &str) -> Option<&'static str> {
    Some(match extension {
        "rs" => "rust",
        "py" => "python",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "go" => "go",
        "java" => "java",
        "c" | "h" => "c",
        "cc" | "cpp" | "hpp" => "cpp",
        _ => return None,
    })
}

/// Lowercased extensions of the other grammars, and more extensions for the classic
/// languages' grammars.
fn by_extension(extension: &str) -> Option<Detected> {
    Some(match extension {
        // More extensions for the classic grammars.
        "pyi" | "pyw" => same("python"),
        "mjs" | "cjs" => same("javascript"),
        "mts" | "cts" => same("typescript"),
        "cxx" | "hh" | "hxx" | "ipp" | "tpp" => same("cpp"),
        // Prose, markup and styles.
        "md" | "markdown" | "mdx" => same("markdown"),
        "html" | "htm" => same("html"),
        "xml" | "svg" | "xsd" | "xsl" | "xslt" | "rng" | "plist" | "xaml" | "csproj" | "fsproj"
        | "vbproj" | "vcxproj" | "props" | "targets" | "nuspec" | "resx" => same("xml"),
        "dtd" => same("dtd"),
        "svelte" => same("svelte"),
        "erb" | "ejs" => same("embedded_template"),
        "css" => same("css"),
        // Data and configuration.
        "json" | "jsonc" | "geojson" | "webmanifest" => same("json"),
        "yaml" | "yml" => same("yaml"),
        "toml" => same("toml"),
        "hcl" | "tf" | "tfvars" => same("hcl"),
        "proto" => same("proto"),
        // Scripts and builds.
        "sh" | "bash" | "zsh" => same("bash"),
        "ps1" | "psm1" | "psd1" => same("powershell"),
        "sql" => same("sql"),
        "mk" | "mak" | "make" => same("make"),
        "cmake" => same("cmake"),
        // Code.
        "cs" | "csx" => same("c_sharp"),
        "cu" | "cuh" => same("cuda"),
        "dart" => same("dart"),
        "ex" | "exs" => same("elixir"),
        "elm" => same("elm"),
        "erl" | "hrl" | "escript" => same("erlang"),
        "f90" | "f95" | "f03" | "f08" => same("fortran"),
        "fs" | "fsx" => same("fsharp"),
        "glsl" | "vert" | "frag" | "geom" | "comp" | "tesc" | "tese" | "vsh" | "fsh" => {
            same("glsl")
        }
        "hs" => same("haskell"),
        "jl" => same("julia"),
        "kt" | "kts" => same("kotlin"),
        "lua" => same("lua"),
        "m" => same("objc"),
        "ml" => same("ocaml"),
        "mli" => Detected {
            name: "ocaml",
            parser: "ocaml_interface",
        },
        "odin" => same("odin"),
        "pm" => same("perl"),
        "php" | "phtml" => same("php"),
        "r" => same("r"),
        "rb" | "rake" | "gemspec" | "ru" => same("ruby"),
        "scala" | "sc" | "sbt" => same("scala"),
        "sol" => same("solidity"),
        "swift" => same("swift"),
        "v" | "vh" | "sv" | "svh" => same("verilog"),
        "zig" | "zon" => same("zig"),
        _ => return None,
    })
}

/// Well-known file names (exact, case-sensitive).
fn by_name(name: &str) -> Option<Detected> {
    Some(match name {
        "Makefile" | "makefile" | "GNUmakefile" => same("make"),
        "CMakeLists.txt" => same("cmake"),
        ".bashrc" | ".bash_profile" | ".bash_login" | ".bash_logout" | ".profile" | ".zshrc"
        | ".zshenv" | ".zprofile" | "PKGBUILD" | "APKBUILD" => same("bash"),
        "Gemfile" | "Rakefile" | "Podfile" | "Vagrantfile" | "Brewfile" => same("ruby"),
        "Cargo.lock" | "Pipfile" => same("toml"),
        _ => return None,
    })
}

/// The language of `path` from its name alone: the classic extensions first (so their
/// detection is unchanged), then the other grammars' file names and extensions. `.pl`
/// and `.t` are Perl here; [`detect`] also checks their content.
pub(crate) fn detect_path(path: &Path) -> Detected {
    let extension = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if let Some(name) = classic(&extension) {
        // Unchanged: `.tsx` (exact case) parses as TSX and `.h` (exact case) as C++,
        // and both keep their language's name.
        let parser = if path.extension().is_some_and(|x| x == "tsx") {
            "tsx"
        } else if path.extension().is_some_and(|x| x == "h") {
            "cpp"
        } else {
            name
        };
        return Detected { name, parser };
    }
    if let Some(found) = path.file_name().and_then(|n| n.to_str()).and_then(by_name) {
        return found;
    }
    if matches!(extension.as_str(), "pl" | "t") {
        return same("perl");
    }
    by_extension(&extension).unwrap_or(TEXT)
}

/// [`detect_path`], except that `.pl` and `.t` files are Perl only when their start
/// looks like Perl: Prolog also uses `.pl`, and other test frameworks use `.t`.
pub(crate) fn detect(path: &Path, text: &str) -> Detected {
    let found = detect_path(path);
    let extension = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    if found.parser == "perl"
        && (extension.eq_ignore_ascii_case("pl") || extension.eq_ignore_ascii_case("t"))
        && !looks_like_perl(text)
    {
        return TEXT;
    }
    found
}

/// A Perl shebang, or a line among the first 50 (in at most 4 KB) that starts with
/// `use strict`, `use warnings`, `use v5`, `use 5.`, `use Test::`, `package X;`,
/// `sub name` or `my $`/`my @`/`my %`/`my (`.
fn looks_like_perl(text: &str) -> bool {
    let mut end = text.len().min(4096);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let head = &text[..end];
    if head.starts_with("#!") && head.lines().next().is_some_and(|l| l.contains("perl")) {
        return true;
    }
    head.lines().take(50).any(|line| {
        let line = line.trim_start();
        [
            "use strict",
            "use warnings",
            "use v5",
            "use 5.",
            "use Test::",
            "my $",
            "my @",
            "my %",
            "my (",
        ]
        .iter()
        .any(|p| line.starts_with(p))
            || line.strip_prefix("package ").is_some_and(|rest| {
                rest.trim_end().ends_with(';') || rest.trim_end().ends_with('{')
            })
            || line
                .strip_prefix("sub ")
                .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_'))
    })
}

/// The grammar for a parser key, or `None` for `text`.
pub(crate) fn grammar(parser: &str) -> Option<Language> {
    Some(match parser {
        "rust" => tree_sitter_rust::LANGUAGE.into(),
        "python" => tree_sitter_python::LANGUAGE.into(),
        "typescript" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        "javascript" => tree_sitter_javascript::LANGUAGE.into(),
        "go" => tree_sitter_go::LANGUAGE.into(),
        "java" => tree_sitter_java::LANGUAGE.into(),
        "c" => tree_sitter_c::LANGUAGE.into(),
        "cpp" => tree_sitter_cpp::LANGUAGE.into(),
        "bash" => tree_sitter_bash::LANGUAGE.into(),
        "c_sharp" => tree_sitter_c_sharp::LANGUAGE.into(),
        "cmake" => tree_sitter_cmake::LANGUAGE.into(),
        "css" => tree_sitter_css::LANGUAGE.into(),
        "cuda" => tree_sitter_cuda::LANGUAGE.into(),
        "dart" => tree_sitter_dart::LANGUAGE.into(),
        "dtd" => tree_sitter_xml::LANGUAGE_DTD.into(),
        "elixir" => tree_sitter_elixir::LANGUAGE.into(),
        "elm" => tree_sitter_elm::LANGUAGE.into(),
        "embedded_template" => tree_sitter_embedded_template::LANGUAGE.into(),
        "erlang" => tree_sitter_erlang::LANGUAGE.into(),
        "fortran" => tree_sitter_fortran::LANGUAGE.into(),
        "fsharp" => tree_sitter_fsharp::LANGUAGE_FSHARP.into(),
        "glsl" => tree_sitter_glsl::LANGUAGE_GLSL.into(),
        "haskell" => tree_sitter_haskell::LANGUAGE.into(),
        "hcl" => tree_sitter_hcl::LANGUAGE.into(),
        "html" => tree_sitter_html::LANGUAGE.into(),
        "json" => tree_sitter_json::LANGUAGE.into(),
        "julia" => tree_sitter_julia::LANGUAGE.into(),
        "kotlin" => tree_sitter_kotlin_ng::LANGUAGE.into(),
        "lua" => tree_sitter_lua::LANGUAGE.into(),
        "make" => tree_sitter_make::LANGUAGE.into(),
        "markdown" => tree_sitter_md::LANGUAGE.into(),
        "objc" => tree_sitter_objc::LANGUAGE.into(),
        "ocaml" => tree_sitter_ocaml::LANGUAGE_OCAML.into(),
        "ocaml_interface" => tree_sitter_ocaml::LANGUAGE_OCAML_INTERFACE.into(),
        "odin" => tree_sitter_odin::LANGUAGE.into(),
        "perl" => ts_parser_perl::LANGUAGE.into(),
        "php" => tree_sitter_php::LANGUAGE_PHP.into(),
        "powershell" => tree_sitter_powershell::LANGUAGE.into(),
        "proto" => tree_sitter_proto::LANGUAGE.into(),
        "r" => tree_sitter_r::LANGUAGE.into(),
        "ruby" => tree_sitter_ruby::LANGUAGE.into(),
        "scala" => tree_sitter_scala::LANGUAGE.into(),
        "solidity" => tree_sitter_solidity::LANGUAGE.into(),
        "sql" => tree_sitter_sequel::LANGUAGE.into(),
        "svelte" => tree_sitter_svelte_ng::LANGUAGE.into(),
        "swift" => tree_sitter_swift::LANGUAGE.into(),
        "toml" => tree_sitter_toml_ng::LANGUAGE.into(),
        "verilog" => tree_sitter_verilog::LANGUAGE.into(),
        "xml" => tree_sitter_xml::LANGUAGE_XML.into(),
        "yaml" => tree_sitter_yaml::LANGUAGE.into(),
        "zig" => tree_sitter_zig::LANGUAGE.into(),
        _ => return None,
    })
}

/// Every parser key [`grammar`] knows, for tests and the rules' consistency checks.
pub(crate) const PARSERS: &[&str] = &[
    "rust",
    "python",
    "typescript",
    "tsx",
    "javascript",
    "go",
    "java",
    "c",
    "cpp",
    "bash",
    "c_sharp",
    "cmake",
    "css",
    "cuda",
    "dart",
    "dtd",
    "elixir",
    "elm",
    "embedded_template",
    "erlang",
    "fortran",
    "fsharp",
    "glsl",
    "haskell",
    "hcl",
    "html",
    "json",
    "julia",
    "kotlin",
    "lua",
    "make",
    "markdown",
    "objc",
    "ocaml",
    "ocaml_interface",
    "odin",
    "perl",
    "php",
    "powershell",
    "proto",
    "r",
    "ruby",
    "scala",
    "solidity",
    "sql",
    "svelte",
    "swift",
    "toml",
    "verilog",
    "xml",
    "yaml",
    "zig",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn of(name: &str) -> Detected {
        detect_path(Path::new(name))
    }

    #[test]
    fn classic_detection_is_unchanged() {
        for (file, name, parser) in [
            ("a.rs", "rust", "rust"),
            ("a.PY", "python", "python"),
            ("a.ts", "typescript", "typescript"),
            ("a.tsx", "typescript", "tsx"),
            ("a.TSX", "typescript", "typescript"),
            ("a.jsx", "javascript", "javascript"),
            ("a.h", "c", "cpp"),
            ("a.H", "c", "c"),
            ("a.hpp", "cpp", "cpp"),
            ("a.cc", "cpp", "cpp"),
        ] {
            assert_eq!(of(file), Detected { name, parser }, "{file}");
        }
    }

    #[test]
    fn names_extensions_and_ambiguities() {
        for (file, parser) in [
            ("Makefile", "make"),
            ("rules.mk", "make"),
            ("CMakeLists.txt", "cmake"),
            ("notes.txt", "text"),
            ("PKGBUILD", "bash"),
            (".bashrc", "bash"),
            ("x.m", "objc"),
            ("x.v", "verilog"),
            ("x.mdx", "markdown"),
            ("x.svelte", "svelte"),
            ("x.html.erb", "embedded_template"),
            ("x.ejs", "embedded_template"),
            ("x.cu", "cuda"),
            ("x.cuh", "cuda"),
            ("x.frag", "glsl"),
            ("x.fs", "fsharp"),
            ("x.php", "php"),
            ("x.mli", "ocaml_interface"),
            ("x.pm", "perl"),
            ("x.R", "r"),
            ("x.F90", "fortran"),
        ] {
            assert_eq!(of(file).parser, parser, "{file}");
        }
        assert_eq!(of("x.mli").name, "ocaml");
        for parser in PARSERS {
            assert!(grammar(parser).is_some(), "{parser}");
        }
    }

    #[test]
    fn perl_only_when_it_looks_like_perl() {
        let perl = detect(Path::new("x.pl"), "#!/usr/bin/perl\nprint 1;\n");
        assert_eq!(perl.parser, "perl");
        let perl = detect(Path::new("x.t"), "use strict;\nuse Test::More;\nok(1);\n");
        assert_eq!(perl.parser, "perl");
        let prolog = detect(Path::new("x.pl"), ":- module(x, []).\nfoo(X) :- bar(X).\n");
        assert_eq!(prolog.parser, "text");
        let module = detect(Path::new("x.pm"), "1;\n");
        assert_eq!(module.parser, "perl");
    }
}
