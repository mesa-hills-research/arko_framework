use super::types::{SkillFrontmatter, SkillLoadError};

/// Parses YAML frontmatter from a markdown string.
///
/// Expects the format:
/// ```text
/// ---
/// name: my-skill
/// description: Does things
/// ---
/// # Body content here
/// ```
///
/// Returns (frontmatter, body_content) on success.
pub fn parse_frontmatter(
    content: &str,
    source_path: &str,
) -> Result<(SkillFrontmatter, String), SkillLoadError> {
    let trimmed = content.trim_start();
    let lines: Vec<&str> = trimmed.lines().collect();

    if lines.is_empty() || lines[0].trim() != "---" {
        return Err(SkillLoadError::MissingFrontmatter {
            path: source_path.to_string(),
        });
    }

    let mut closing_line_idx = None;
    for (i, line) in lines.iter().enumerate().skip(1) {
        if line.trim() == "---" {
            closing_line_idx = Some(i);
            break;
        }
    }

    let closing_idx = closing_line_idx.ok_or_else(|| SkillLoadError::MalformedFrontmatter {
        path: source_path.to_string(),
    })?;

    let yaml_str = lines[1..closing_idx].join("\n");

    let frontmatter: SkillFrontmatter =
        serde_yaml::from_str(&yaml_str).map_err(|e| SkillLoadError::FrontmatterParseError {
            path: source_path.to_string(),
            source: e,
        })?;

    if frontmatter.name.trim().is_empty() {
        return Err(SkillLoadError::MissingRequiredField {
            field: "name".to_string(),
            path: source_path.to_string(),
        });
    }

    if frontmatter.description.trim().is_empty() {
        return Err(SkillLoadError::MissingRequiredField {
            field: "description".to_string(),
            path: source_path.to_string(),
        });
    }

    if frontmatter
        .name
        .chars()
        .any(|c| c.is_uppercase() || c == '_' || c == ' ')
    {
        log::warn!(
            "Skill '{}': name is not kebab-case. Recommended format: lowercase-with-hyphens.",
            frontmatter.name
        );
    }

    if frontmatter.description.len() > 1024 {
        log::warn!(
            "Skill '{}': description exceeds 1024 characters ({} chars). Consider shortening.",
            frontmatter.name,
            frontmatter.description.len()
        );
    }

    let body = if closing_idx + 1 < lines.len() {
        lines[closing_idx + 1..].join("\n")
    } else {
        String::new()
    };

    Ok((frontmatter, body))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_frontmatter() {
        let content = "---\nname: test-skill\ndescription: A test skill\n---\n# Body\nHello";
        let (fm, body) = parse_frontmatter(content, "test.md").unwrap();
        assert_eq!(fm.name, "test-skill");
        assert_eq!(fm.description, "A test skill");
        assert!(!fm.auto_load);
        assert_eq!(body, "# Body\nHello");
    }

    #[test]
    fn test_parse_auto_load() {
        let content = "---\nname: auto\ndescription: Auto loaded\nauto_load: true\n---\nContent";
        let (fm, _body) = parse_frontmatter(content, "test.md").unwrap();
        assert!(fm.auto_load);
    }

    #[test]
    fn test_missing_frontmatter() {
        let content = "# Just a markdown file";
        let result = parse_frontmatter(content, "test.md");
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_closing_delimiter() {
        let content = "---\nname: broken\ndescription: No closing\n# Body";
        let result = parse_frontmatter(content, "test.md");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_name() {
        let content = "---\nname: \"\"\ndescription: Has description\n---\nBody";
        let result = parse_frontmatter(content, "test.md");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_description() {
        let content = "---\nname: valid-name\ndescription: \"\"\n---\nBody";
        let result = parse_frontmatter(content, "test.md");
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_fields_preserved() {
        let content = "---\nname: extra\ndescription: Has extras\nlicense: MIT\n---\nBody";
        let (fm, _) = parse_frontmatter(content, "test.md").unwrap();
        assert!(fm.extra.contains_key("license"));
    }

    #[test]
    fn test_empty_body() {
        let content = "---\nname: no-body\ndescription: No body content\n---";
        let (fm, body) = parse_frontmatter(content, "test.md").unwrap();
        assert_eq!(fm.name, "no-body");
        assert!(body.is_empty());
    }
}
