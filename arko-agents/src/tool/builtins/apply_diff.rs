use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use crate::tool::{
    TaskContext, Tool, ToolDefinition, ToolError, ToolParameterDefinition, ToolParameterType,
    ToolTypeDetails,
};

pub const APPLY_DIFF_TOOL_NAME: &str = "apply_diff";
const PATH_PARAM_NAME: &str = "path";
const DIFF_PARAM_NAME: &str = "diff";

const AD_DIFF_MARKER_ESCAPE_PREFIX: &str = "AD_ESC::";
const DEFAULT_FUZZY_THRESHOLD: f64 = 1.0;
const BUFFER_LINES: usize = 20;

#[derive(Debug, Clone)]
struct ParsedDiffBlock {
    original_index: usize,
    start_line: usize,
    search_lines: Vec<String>,
    replace_lines: Vec<String>,
}

#[derive(Debug)]
struct DiffApplicationResult {
    original_block_index: usize,
    original_target_start_line: usize,
    applied_successfully: bool,
    message: String,
    matched_at_line: Option<usize>,
    similarity_score: Option<f64>,
}

fn diff_parser_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        let esc = regex::escape(AD_DIFF_MARKER_ESCAPE_PREFIX);
        let pattern = format!(
            r"(?ms)(?:^|\n)(?P<s_esc>{esc})?<<<<<<< SEARCH\s*\n\s*:start_line:\s*(?P<start_line>\d+)\s*\n(?P<h_esc>{esc})?-------\s*\n(?P<search_content>.*?)\n(?P<eq_esc>{esc})?=======\s*\n(?P<replace_content>.*?)\n(?P<r_esc>{esc})?>>>>>>> REPLACE(?:(?:\r?\n)|$)"
        );
        Regex::new(&pattern).expect("Invalid diff parser regex")
    })
}

static LINE_SPLIT_RE: OnceLock<Regex> = OnceLock::new();
fn get_line_split_re() -> &'static Regex {
    LINE_SPLIT_RE.get_or_init(|| Regex::new(r"\r?\n").expect("Invalid line split regex"))
}

static LINE_NUM_PREFIX_RE: OnceLock<Regex> = OnceLock::new();
fn get_line_num_prefix_re() -> &'static Regex {
    LINE_NUM_PREFIX_RE
        .get_or_init(|| Regex::new(r"^\s*\d+\s+\|\s").expect("Invalid line number prefix regex"))
}

#[derive(Debug, Clone)]
pub struct ApplyDiffTool;

impl ApplyDiffTool {
    pub fn new() -> Self {
        Self {}
    }

    fn unescape_content(mut content: String) -> String {
        let markers_to_unescape = [
            ("<<<<<<< SEARCH", "<<<<<<< SEARCH"),
            ("=======", "======="),
            (">>>>>>> REPLACE", ">>>>>>> REPLACE"),
            ("-------", "-------"),
            (":start_line:", ":start_line:"),
        ];

        for (escaped_marker_suffix, unescaped_marker) in markers_to_unescape {
            let full_escaped_marker =
                format!("{}{}", AD_DIFF_MARKER_ESCAPE_PREFIX, escaped_marker_suffix);
            content = content.replace(&full_escaped_marker, unescaped_marker);
        }
        content
    }

    fn parse_diff_str(&self, diff_str: &str) -> Result<Vec<ParsedDiffBlock>, ToolError> {
        if let Err(e) = self.validate_marker_sequencing(diff_str) {
            return Err(ToolError::ParameterValidation {
                param_name: DIFF_PARAM_NAME.to_string(),
                message: format!("Diff content validation failed: {}", e),
            });
        }

        let mut blocks = Vec::new();
        for cap in diff_parser_regex().captures_iter(diff_str) {
            if cap.name("s_esc").is_some()
                || cap.name("h_esc").is_some()
                || cap.name("eq_esc").is_some()
                || cap.name("r_esc").is_some()
            {
                log::trace!("Skipping capture as a diff block because a delimiter was escaped.");
                continue;
            }

            let start_line_str = cap.name("start_line").unwrap().as_str();
            let current_block_index = blocks.len();

            let start_line =
                start_line_str
                    .parse::<usize>()
                    .map_err(|_| ToolError::ParameterValidation {
                        param_name: DIFF_PARAM_NAME.to_string(),
                        message: format!(
                            "Invalid start_line number '{}' in diff block {}.",
                            start_line_str,
                            current_block_index + 1
                        ),
                    })?;
            if start_line == 0 {
                return Err(ToolError::ParameterValidation {
                    param_name: DIFF_PARAM_NAME.to_string(),
                    message: format!(
                        "start_line must be 1-based, found 0 in diff block {}.",
                        current_block_index + 1
                    ),
                });
            }

            let raw_search_content = cap.name("search_content").unwrap().as_str().to_string();
            let raw_replace_content = cap.name("replace_content").unwrap().as_str().to_string();

            let search_content = Self::unescape_content(raw_search_content);
            let replace_content = Self::unescape_content(raw_replace_content);

            let line_splitter = get_line_split_re();
            let search_lines = line_splitter
                .split(&search_content)
                .map(String::from)
                .collect();
            let replace_lines = line_splitter
                .split(&replace_content)
                .map(String::from)
                .collect();

            blocks.push(ParsedDiffBlock {
                original_index: current_block_index,
                start_line,
                search_lines,
                replace_lines,
            });
        }

        if blocks.is_empty() && !diff_str.trim().is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: DIFF_PARAM_NAME.to_string(),
                message: "No valid diff blocks found in 'diff' parameter. Ensure format is correct and delimiters are not escaped.".to_string(),
            });
        }
        Ok(blocks)
    }

    fn validate_marker_sequencing(&self, diff_content: &str) -> Result<(), String> {
        enum State {
            Start,
            AfterSearch,
            AfterSeparator,
        }
        let mut current_state = State::Start;
        let mut line_num = 0;

        let search_marker_prefix = format!("{}<<<<<<< SEARCH", AD_DIFF_MARKER_ESCAPE_PREFIX);
        let sep_marker_prefix = format!("{}=======", AD_DIFF_MARKER_ESCAPE_PREFIX);
        let replace_marker_prefix = format!("{}>>>>>>> REPLACE", AD_DIFF_MARKER_ESCAPE_PREFIX);

        let unescaped_search_marker = "<<<<<<< SEARCH";
        let unescaped_sep_marker = "=======";
        let unescaped_replace_marker = ">>>>>>> REPLACE";

        for line_content in diff_content.lines() {
            line_num += 1;
            let line = line_content.trim_start();

            let is_unescaped_search = line.starts_with(unescaped_search_marker)
                && !line.starts_with(&search_marker_prefix);
            let is_unescaped_sep =
                line.starts_with(unescaped_sep_marker) && !line.starts_with(&sep_marker_prefix);
            let is_unescaped_replace = line.starts_with(unescaped_replace_marker)
                && !line.starts_with(&replace_marker_prefix);

            match current_state {
                State::Start => {
                    if is_unescaped_sep {
                        return Err(format!(
                            "Unexpected '{}' at line {}. Expected '{}'.",
                            unescaped_sep_marker, line_num, unescaped_search_marker
                        ));
                    }
                    if is_unescaped_replace {
                        return Err(format!(
                            "Unexpected '{}' at line {}. Expected '{}'.",
                            unescaped_replace_marker, line_num, unescaped_search_marker
                        ));
                    }
                    if is_unescaped_search {
                        current_state = State::AfterSearch;
                    }
                }
                State::AfterSearch => {
                    if is_unescaped_search {
                        return Err(format!(
                            "Unexpected '{}' at line {}. Expected '{}' or content.",
                            unescaped_search_marker, line_num, unescaped_sep_marker
                        ));
                    }
                    if is_unescaped_replace {
                        return Err(format!(
                            "Unexpected '{}' at line {}. Expected '{}' or content.",
                            unescaped_replace_marker, line_num, unescaped_sep_marker
                        ));
                    }
                    if is_unescaped_sep {
                        current_state = State::AfterSeparator;
                    }
                }
                State::AfterSeparator => {
                    if is_unescaped_search {
                        return Err(format!(
                            "Unexpected '{}' at line {}. Expected '{}' or content.",
                            unescaped_search_marker, line_num, unescaped_replace_marker
                        ));
                    }
                    if is_unescaped_sep {
                        return Err(format!(
                            "Unexpected '{}' at line {}. Expected '{}' or content.",
                            unescaped_sep_marker, line_num, unescaped_replace_marker
                        ));
                    }
                    if is_unescaped_replace {
                        current_state = State::Start;
                    }
                }
            }
        }

        if !matches!(current_state, State::Start) {
            let expected = match current_state {
                State::AfterSearch => unescaped_sep_marker,
                State::AfterSeparator => unescaped_replace_marker,
                State::Start => unreachable!(),
            };
            return Err(format!(
                "Unexpected end of diff. Expected marker '{}' was not found.",
                expected
            ));
        }
        Ok(())
    }

    fn determine_line_ending<'a>(&self, content: &'a str) -> &'a str {
        if content.contains("\r\n") {
            "\r\n"
        } else {
            "\n"
        }
    }

    fn normalize_text_for_comparison(&self, text: &str) -> String {
        text.replace("\r\n", "\n")
    }

    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let s1_chars: Vec<char> = s1.chars().collect();
        let s2_chars: Vec<char> = s2.chars().collect();
        let len1 = s1_chars.len();
        let len2 = s2_chars.len();
        let mut d: Vec<Vec<usize>> = vec![vec![0; len2 + 1]; len1 + 1];
        for (i, row) in d.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }
        for (j, cell) in d[0].iter_mut().enumerate().take(len2 + 1) {
            *cell = j;
        }
        for j in 1..=len2 {
            for i in 1..=len1 {
                let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                    0
                } else {
                    1
                };
                d[i][j] = (d[i - 1][j] + 1)
                    .min(d[i][j - 1] + 1)
                    .min(d[i - 1][j - 1] + cost);
            }
        }
        d[len1][len2]
    }

    fn get_similarity(&self, original: &str, search: &str) -> f64 {
        if search.is_empty() && original.is_empty() {
            return 1.0;
        }
        if search.is_empty() || original.is_empty() {
            return 0.0;
        }
        let normalized_original = self.normalize_text_for_comparison(original);
        let normalized_search = self.normalize_text_for_comparison(search);
        if normalized_original == normalized_search {
            return 1.0;
        }
        let dist = self.levenshtein_distance(&normalized_original, &normalized_search);
        let max_len = normalized_original.len().max(normalized_search.len()) as f64;
        if max_len == 0.0 {
            return 1.0;
        }
        (1.0 - (dist as f64 / max_len)).max(0.0)
    }

    fn strip_line_numbers_from_text(&self, text: &str) -> String {
        text.lines()
            .map(|line| {
                let line_prefix_re = get_line_num_prefix_re();
                if line_prefix_re.is_match(line)
                    && let Some(idx) = line.find('|')
                {
                    return line[idx + 1..].trim_start().to_string();
                }
                line.to_string()
            })
            .collect::<Vec<String>>()
            .join("\n")
    }

    fn find_match_location(
        &self,
        search_lines_vec: &[String],
        current_file_lines: &[String],
        target_start_line: usize,
        fuzzy_threshold: f64,
    ) -> Option<(usize, f64, String)> {
        if search_lines_vec.is_empty() {
            return None;
        }
        let search_chunk = search_lines_vec.join("\n");
        let exact_match_start_idx = target_start_line.saturating_sub(1);

        if exact_match_start_idx + search_lines_vec.len() <= current_file_lines.len() {
            let file_chunk_slice = &current_file_lines
                [exact_match_start_idx..exact_match_start_idx + search_lines_vec.len()];
            let file_chunk = file_chunk_slice.join("\n");
            let similarity = self.get_similarity(&file_chunk, &search_chunk);
            if similarity == 1.0 {
                return Some((exact_match_start_idx, similarity, file_chunk));
            }
        }

        let search_len = search_lines_vec.len();
        let window_start = exact_match_start_idx.saturating_sub(BUFFER_LINES);
        let window_end =
            (exact_match_start_idx + search_len + BUFFER_LINES).min(current_file_lines.len());
        let mut best_score = 0.0;
        let mut best_match_index = None;
        let mut best_match_content = String::new();
        let mid_point_abs = exact_match_start_idx;
        let mut left_idx_abs = mid_point_abs;
        let mut right_idx_abs = mid_point_abs + 1;

        while (left_idx_abs >= window_start && left_idx_abs + search_len <= window_end)
            || (right_idx_abs + search_len <= window_end && right_idx_abs >= window_start)
        {
            if left_idx_abs >= window_start && left_idx_abs + search_len <= window_end {
                if let Some(slice) = current_file_lines.get(left_idx_abs..left_idx_abs + search_len)
                {
                    let original_chunk = slice.join("\n");
                    let similarity = self.get_similarity(&original_chunk, &search_chunk);
                    if similarity > best_score {
                        best_score = similarity;
                        best_match_index = Some(left_idx_abs);
                        best_match_content = original_chunk;
                    }
                }
                if left_idx_abs == 0 {
                    if !(right_idx_abs + search_len <= window_end && right_idx_abs >= window_start)
                    {
                        break;
                    }
                } else {
                    left_idx_abs = left_idx_abs.saturating_sub(1);
                }
            }
            if right_idx_abs + search_len <= window_end && right_idx_abs >= window_start {
                if let Some(slice) =
                    current_file_lines.get(right_idx_abs..right_idx_abs + search_len)
                {
                    let original_chunk = slice.join("\n");
                    let similarity = self.get_similarity(&original_chunk, &search_chunk);
                    if similarity > best_score {
                        best_score = similarity;
                        best_match_index = Some(right_idx_abs);
                        best_match_content = original_chunk;
                    }
                }
                right_idx_abs += 1;
            }
            if left_idx_abs < window_start && right_idx_abs + search_len > window_end {
                break;
            }
        }

        if best_score >= fuzzy_threshold
            && let Some(idx) = best_match_index
        {
            return Some((idx, best_score, best_match_content));
        }

        let prefixed_count = search_lines_vec
            .iter()
            .filter(|l| get_line_num_prefix_re().is_match(l))
            .count();
        let should_attempt_stripping =
            prefixed_count >= 3 || (prefixed_count * 100) >= (search_len.saturating_mul(70));
        if should_attempt_stripping {
            let stripped_search_chunk = self.strip_line_numbers_from_text(&search_chunk);
            if stripped_search_chunk != search_chunk {
                let mut agg_best_score = 0.0;
                let mut agg_best_match_index = None;
                let mut agg_best_match_content = String::new();
                left_idx_abs = mid_point_abs;
                right_idx_abs = mid_point_abs + 1;
                while (left_idx_abs >= window_start && left_idx_abs + search_len <= window_end)
                    || (right_idx_abs + search_len <= window_end && right_idx_abs >= window_start)
                {
                    if left_idx_abs >= window_start && left_idx_abs + search_len <= window_end {
                        if let Some(slice) =
                            current_file_lines.get(left_idx_abs..left_idx_abs + search_len)
                        {
                            let original_chunk_text = slice.join("\n");
                            let stripped_original_chunk =
                                self.strip_line_numbers_from_text(&original_chunk_text);
                            let similarity = self
                                .get_similarity(&stripped_original_chunk, &stripped_search_chunk);
                            if similarity > agg_best_score {
                                agg_best_score = similarity;
                                agg_best_match_index = Some(left_idx_abs);
                                agg_best_match_content = original_chunk_text;
                            }
                        }
                        if left_idx_abs == 0 {
                            if !(right_idx_abs + search_len <= window_end
                                && right_idx_abs >= window_start)
                            {
                                break;
                            }
                        } else {
                            left_idx_abs = left_idx_abs.saturating_sub(1);
                        }
                    }
                    if right_idx_abs + search_len <= window_end && right_idx_abs >= window_start {
                        if let Some(slice) =
                            current_file_lines.get(right_idx_abs..right_idx_abs + search_len)
                        {
                            let original_chunk_text = slice.join("\n");
                            let stripped_original_chunk =
                                self.strip_line_numbers_from_text(&original_chunk_text);
                            let similarity = self
                                .get_similarity(&stripped_original_chunk, &stripped_search_chunk);
                            if similarity > agg_best_score {
                                agg_best_score = similarity;
                                agg_best_match_index = Some(right_idx_abs);
                                agg_best_match_content = original_chunk_text;
                            }
                        }
                        right_idx_abs += 1;
                    }
                    if left_idx_abs < window_start && right_idx_abs + search_len > window_end {
                        break;
                    }
                }
                if agg_best_score >= fuzzy_threshold
                    && let Some(idx) = agg_best_match_index
                    && agg_best_score > best_score
                {
                    return Some((idx, agg_best_score, agg_best_match_content));
                }
            }
        }

        if best_score >= fuzzy_threshold
            && let Some(idx) = best_match_index
        {
            return Some((idx, best_score, best_match_content));
        }
        None
    }

    fn perform_replacement(
        &self,
        match_start_index: usize,
        search_lines_from_block: &[String],
        replace_lines_from_block: &[String],
        current_file_lines: &mut Vec<String>,
    ) {
        let matched_file_lines = current_file_lines
            .iter()
            .skip(match_start_index)
            .take(search_lines_from_block.len())
            .map(String::from)
            .collect::<Vec<String>>();
        let base_original_indent = matched_file_lines
            .first()
            .map(|line| {
                line.chars()
                    .take_while(|c| c.is_whitespace())
                    .collect::<String>()
            })
            .unwrap_or_default();
        let base_search_block_indent = search_lines_from_block
            .first()
            .map(|line| {
                line.chars()
                    .take_while(|c| c.is_whitespace())
                    .collect::<String>()
            })
            .unwrap_or_default();
        let mut final_replace_lines = Vec::new();

        for replace_line_content in replace_lines_from_block {
            let current_replace_line_indent = replace_line_content
                .chars()
                .take_while(|c| c.is_whitespace())
                .collect::<String>();
            let content_without_indent =
                replace_line_content[current_replace_line_indent.len()..].to_string();
            let relative_indent =
                if current_replace_line_indent.starts_with(&base_search_block_indent) {
                    current_replace_line_indent[base_search_block_indent.len()..].to_string()
                } else {
                    current_replace_line_indent.clone()
                };
            let new_line = format!(
                "{}{}{}",
                base_original_indent, relative_indent, content_without_indent
            );
            final_replace_lines.push(new_line);
        }
        current_file_lines.splice(
            match_start_index..match_start_index + matched_file_lines.len(),
            final_replace_lines,
        );
    }
}

impl Default for ApplyDiffTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ApplyDiffTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: APPLY_DIFF_TOOL_NAME.to_string(),
            description: format!(
                "Apply surgical edits to an existing file using a special diff format. \
More efficient than `write_file` for modifying existing files as it only requires the changed portions. \
Supports multiple diff blocks in a single call for the same file. \
Diff blocks are applied in reverse order of their `start_line` to ensure line number accuracy.

Diff Format:
Each diff block must follow this structure:
<<<<<<< SEARCH
:start_line: (Required) The 1-based line number where the SEARCH block starts.
-------
[Exact content to find, including all whitespace, newlines, and indentation.]
=======
[New content to replace the SEARCH block with.]
[To delete content, leave this section empty.]
>>>>>>> REPLACE

Escaping Diff Markers:
If any diff markers (`<<<<<<< SEARCH`, `-------`, `=======`, `>>>>>>> REPLACE`, or `:start_line:`) \
need to appear as literal text within your SEARCH or REPLACE content, prefix them with `{esc}`.

Example diff value (simple replacement):
<<<<<<< SEARCH
:start_line:10
-------
def old_function(x):
    return x + 1
=======
def new_function(x):
    return x * 2
>>>>>>> REPLACE

Example diff value (multiple changes, applied in reverse start_line order):
<<<<<<< SEARCH
:start_line:20
-------
const OLD_VALUE = 100;
=======
const NEW_VALUE = 200;
>>>>>>> REPLACE

<<<<<<< SEARCH
:start_line:5
-------
import {{ old }} from 'module';
=======
import {{ new }} from 'module';
>>>>>>> REPLACE

Best Practices:
- Always use `read_file` first to get exact current content and line numbers.
- Ensure `start_line` is accurate.
- Include precise indentation and surrounding blank lines.
- For insertions or deletions, include surrounding code in the SEARCH block to anchor the change.",
                esc = AD_DIFF_MARKER_ESCAPE_PREFIX,
            ),
            output_based_example: Some(
                "<apply_diff>\n\
                <path>src/main.rs</path>\n\
                <diff>\n\
                <<<<<<< SEARCH\n\
                :start_line:10\n\
                -------\n\
                def old_function(x):\n\
                    return x + 1\n\
                =======\n\
                def new_function(x):\n\
                    return x * 2\n\
                >>>>>>> REPLACE\n\
                </diff>\n\
                </apply_diff>".to_string()
            ),
            parameters: vec![
                ToolParameterDefinition {
                    name: PATH_PARAM_NAME.to_string(),
                    description: "The path to the file to modify (relative to the current working directory, e.g., 'src/main.rs').".to_string(),
                    param_type: ToolParameterType::String, required: true, is_path: true, flag: None, default_value: None,
                },
                ToolParameterDefinition {
                    name: DIFF_PARAM_NAME.to_string(),
                    description: "A string containing one or more diff blocks specifying the changes.".to_string(),
                    param_type: ToolParameterType::String, required: true, is_path: false, flag: None, default_value: None,
                },
            ],
            supports_batched_calls: true,
            tool_type_details: ToolTypeDetails::BuiltIn,
            raw_json_schema_override: None,
        }
    }

    async fn execute(
        &self,
        context: &TaskContext<'_>,
        params: HashMap<String, Value>,
    ) -> Result<String, ToolError> {
        let path_str = params
            .get(PATH_PARAM_NAME)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: PATH_PARAM_NAME.to_string(),
            })?
            .as_str()
            .ok_or_else(|| ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: "Parameter 'path' must be a string.".to_string(),
            })?
            .trim()
            .to_string();
        let diff_str = params
            .get(DIFF_PARAM_NAME)
            .ok_or_else(|| ToolError::MissingParameter {
                param_name: DIFF_PARAM_NAME.to_string(),
            })?
            .as_str()
            .ok_or_else(|| ToolError::ParameterValidation {
                param_name: DIFF_PARAM_NAME.to_string(),
                message: "Parameter 'diff' must be a string.".to_string(),
            })?
            .to_string();

        if path_str.is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: PATH_PARAM_NAME.to_string(),
                message: "File path cannot be empty.".to_string(),
            });
        }
        if diff_str.trim().is_empty() {
            return Err(ToolError::ParameterValidation {
                param_name: DIFF_PARAM_NAME.to_string(),
                message: "Diff content cannot be empty.".to_string(),
            });
        }

        let absolute_path = context.current_working_directory.join(&path_str);
        if !absolute_path.exists() {
            return Err(ToolError::Io {
                tool_name: self.name(),
                source: std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("File not found at path: {}", absolute_path.display()),
                ),
            });
        }

        let original_content = tokio::fs::read_to_string(&absolute_path)
            .await
            .map_err(|e| ToolError::Io {
                tool_name: self.name(),
                source: e,
            })?;
        let original_line_ending = self.determine_line_ending(&original_content);

        let mut current_file_lines: Vec<String> = get_line_split_re()
            .split(&original_content)
            .map(String::from)
            .collect();

        let mut diff_blocks = self.parse_diff_str(&diff_str)?;
        if diff_blocks.is_empty() {
            return Ok(format!(
                "No valid diff operations to perform for file '{}'.",
                path_str
            ));
        }

        diff_blocks.sort_by_key(|b| std::cmp::Reverse(b.start_line));
        let mut application_results = Vec::new();
        let total_blocks = diff_blocks.len();

        for block in diff_blocks {
            if block.search_lines.is_empty() && block.replace_lines.is_empty() {
                application_results.push(DiffApplicationResult {
                    original_block_index: block.original_index,
                    original_target_start_line: block.start_line,
                    applied_successfully: false,
                    message: "Search and Replace content are both empty.".to_string(),
                    matched_at_line: None,
                    similarity_score: None,
                });
                continue;
            }
            if block.search_lines.is_empty() && !block.replace_lines.is_empty() {
                application_results.push(DiffApplicationResult {
                    original_block_index: block.original_index, original_target_start_line: block.start_line,
                    applied_successfully: false, message: "Search content is empty. For pure insertion, search for a line to insert near.".to_string(),
                    matched_at_line: None, similarity_score: None,
                });
                continue;
            }

            match self.find_match_location(
                &block.search_lines,
                &current_file_lines,
                block.start_line,
                DEFAULT_FUZZY_THRESHOLD,
            ) {
                Some((match_start_idx, score, _matched_text)) => {
                    self.perform_replacement(
                        match_start_idx,
                        &block.search_lines,
                        &block.replace_lines,
                        &mut current_file_lines,
                    );
                    application_results.push(DiffApplicationResult {
                        original_block_index: block.original_index,
                        original_target_start_line: block.start_line,
                        applied_successfully: true,
                        message: format!(
                            "Successfully applied diff block (originally targeting line {}).",
                            block.start_line
                        ),
                        matched_at_line: Some(match_start_idx + 1),
                        similarity_score: Some(score),
                    });
                }
                None => {
                    application_results.push(DiffApplicationResult {
                        original_block_index: block.original_index, original_target_start_line: block.start_line,
                        applied_successfully: false,
                        message: format!("Failed to find a suitable match for diff block (originally targeting line {}). Search content:\n{}\n", block.start_line, block.search_lines.join("\n")),
                        matched_at_line: None, similarity_score: None,
                    });
                }
            }
        }

        application_results.sort_by_key(|r| r.original_block_index);
        let successful_applications = application_results
            .iter()
            .filter(|r| r.applied_successfully)
            .count();

        let final_content = current_file_lines.join(original_line_ending);

        if successful_applications > 0 && final_content != original_content {
            tokio::fs::write(&absolute_path, &final_content)
                .await
                .map_err(|e| ToolError::Io {
                    tool_name: self.name(),
                    source: e,
                })?;
        }

        let mut summary_messages = Vec::new();
        for result in application_results {
            let status = if result.applied_successfully {
                "SUCCESS"
            } else {
                "FAILURE"
            };
            let match_info = if let Some(line) = result.matched_at_line {
                format!(
                    " (matched at line {}, score: {:.2})",
                    line,
                    result.similarity_score.unwrap_or(0.0)
                )
            } else {
                "".to_string()
            };
            summary_messages.push(format!(
                "  Block {} (target line {}): {} - {}{}",
                result.original_block_index + 1,
                result.original_target_start_line,
                status,
                result.message,
                match_info
            ));
        }

        let overall_summary = if successful_applications == total_blocks {
            format!(
                "Successfully applied all {} diff block(s) to '{}'.",
                total_blocks, path_str
            )
        } else if successful_applications > 0 {
            format!(
                "Applied {} of {} diff block(s) to '{}'.\nDetails:\n{}",
                successful_applications,
                total_blocks,
                path_str,
                summary_messages.join("\n")
            )
        } else {
            return Err(ToolError::ExecutionFailed(format!(
                "All {} diff block(s) failed to apply to file '{}'.\nDetails:\n{}",
                total_blocks,
                path_str,
                summary_messages.join("\n")
            )));
        };

        Ok(overall_summary)
    }
}
