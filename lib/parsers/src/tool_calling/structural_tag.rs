// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! xgrammar structural tags for model-native tool-call syntax.
//!
//! A `response_format` JSON schema and `tools` in the same request are not
//! composable as two constraints — the backend accepts exactly one. Sending the
//! schema alone is not neutral either: its grammar accepts nothing but the
//! schema object, so a model that opens a native tool call is coerced into the
//! JSON string and never terminates, burning the whole token budget and
//! returning a malformed object with no tool call.
//!
//! The fix is a single grammar accepting every legitimate shape:
//!
//! ```text
//! [ reasoning_prefix ] ( content_schema [ tool_calls ] | tool_calls )
//! ```
//!
//! Gemma 4 writes tool arguments as `key:<|"|>value<|"|>`, which is not JSON, so
//! the tag body is left unconstrained: the wrapper and the tool *name* are
//! constrained, the arguments are not. (xgrammar ships a `gemma_4` template but
//! leaves it unregistered for exactly this reason.)
//!
//! One exception: a REQUIRED argument must not be omitted or emitted empty (production's
//! `EMPTY_REQUIRED_ARG` -- the model calls `lookup_info{}` or `lookup_info{query:<|"|><|"|>}`
//! instead of filling `query` in). See `gemma4_tool_args_content`.

use serde_json::{Value, json};

use super::ToolDefinition;

/// Gemma 4 channel markers, as they appear in decoded text.
pub const GEMMA4_TOOL_CALL_BEGIN: &str = "<|tool_call>call:";
pub const GEMMA4_TOOL_CALL_END: &str = "<tool_call|>";
pub const GEMMA4_TOOL_CALL_TRIGGER: &str = "<|tool_call>";
pub const GEMMA4_REASONING_END: &str = "<channel|>";
pub const GEMMA4_THOUGHT_BEGIN: &str = "<|channel>thought\n";
/// Gemma 4's string-argument delimiter: `key:<|"|>value<|"|>`.
const GEMMA4_STRING_DELIM: &str = "<|\"|>";

/// Names of this tool's top-level REQUIRED properties typed `string`.
fn required_nonempty_string_keys(parameters: Option<&Value>) -> Vec<String> {
    let Some(params) = parameters else {
        return Vec::new();
    };
    let Some(required) = params.get("required").and_then(|r| r.as_array()) else {
        return Vec::new();
    };
    let Some(properties) = params.get("properties").and_then(|p| p.as_object()) else {
        return Vec::new();
    };
    required
        .iter()
        .filter_map(|r| r.as_str())
        .filter(|key| {
            properties
                .get(*key)
                .and_then(|p| p.get("type"))
                .and_then(|t| t.as_str())
                .is_some_and(|t| t.eq_ignore_ascii_case("string"))
        })
        .map(str::to_string)
        .collect()
}

/// Above this many top-level properties, permuting required×optional key combinations is not
/// worth the branch-count blow-up -- fall back to the looser dispatch patch instead. Real tools
/// here top out at 2 (`transfer_call`), so this is generous headroom, not a tight fit.
const MAX_PROPS_FOR_EXACT_ARGS_GRAMMAR: usize = 4;

/// A `key:value` slot format for a `string`/`integer`/`number`/`boolean` property, or `None` for
/// anything else (object, array, unspecified) -- those are out of scope for exact enforcement;
/// see `exact_args_grammar`.
fn simple_value_format(prop_schema: &Value, required: bool) -> Option<Value> {
    let ty = prop_schema.get("type").and_then(|t| t.as_str())?;
    Some(if ty.eq_ignore_ascii_case("string") {
        json!({
            "type": "sequence",
            "elements": [
                {"type": "const_string", "value": GEMMA4_STRING_DELIM},
                // `+` (required, non-empty) vs `*` (optional, may be empty) -- an OPTIONAL
                // string argument emitted empty is not the reported failure and not this fix's
                // job; only a REQUIRED one must never be empty.
                {"type": "regex", "pattern": if required { "[\\s\\S]+" } else { "[\\s\\S]*" }},
                {"type": "const_string", "value": GEMMA4_STRING_DELIM},
            ],
        })
    } else if ty.eq_ignore_ascii_case("integer") || ty.eq_ignore_ascii_case("number") {
        json!({"type": "regex", "pattern": "-?[0-9]+(\\.[0-9]+)?"})
    } else if ty.eq_ignore_ascii_case("boolean") {
        json!({"type": "or", "elements": [
            {"type": "const_string", "value": "true"},
            {"type": "const_string", "value": "false"},
        ]})
    } else {
        return None;
    })
}

/// All permutations of `items`, as index-order vectors. `items.len()` is capped by the caller
/// (`MAX_PROPS_FOR_EXACT_ARGS_GRAMMAR`), so this never runs on more than a handful of elements.
fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items.to_vec()];
    }
    let mut out = Vec::new();
    for (i, &head) in items.iter().enumerate() {
        let mut rest = items.to_vec();
        rest.remove(i);
        for mut perm in permutations(&rest) {
            perm.insert(0, head);
            out.push(perm);
        }
    }
    out
}

/// Exact grammar for a tool's `{key:value,...}` body: every REQUIRED key is guaranteed to
/// appear (with a well-formed, non-empty value if it's a string), every optional key may or may
/// not appear, keys may arrive in any order the model picks (permuted, not just one fixed
/// order) -- so both observed shapes of `EMPTY_REQUIRED_ARG` (`lookup_info{}` with the key
/// omitted entirely, and `lookup_info{query:<|"|><|"|>}` with it present but empty) become
/// unreachable strings, not just less likely ones.
///
/// `None` when the schema doesn't fit: too many properties, or any property typed something
/// other than string/integer/number/boolean (object/array nesting is a real shape here --
/// `transfer_call.fields` -- and not attempted). The caller falls back to a looser guarantee
/// for those; see `gemma4_tool_args_content`.
fn exact_args_grammar(properties: &serde_json::Map<String, Value>, required: &[String]) -> Option<Value> {
    if properties.is_empty() || properties.len() > MAX_PROPS_FOR_EXACT_ARGS_GRAMMAR {
        return None;
    }
    let keys: Vec<&String> = properties.keys().collect();
    let mut slots = Vec::with_capacity(keys.len());
    for key in &keys {
        let is_required = required.iter().any(|r| r == *key);
        let value_format = simple_value_format(&properties[*key], is_required)?;
        slots.push((key.as_str(), value_format, is_required));
    }

    let required_idx: Vec<usize> = (0..slots.len()).filter(|&i| slots[i].2).collect();
    let optional_idx: Vec<usize> = (0..slots.len()).filter(|&i| !slots[i].2).collect();

    let mut branches = Vec::new();
    for mask in 0u32..(1 << optional_idx.len()) {
        let mut chosen = required_idx.clone();
        for (bit, &oi) in optional_idx.iter().enumerate() {
            if mask & (1 << bit) != 0 {
                chosen.push(oi);
            }
        }
        for perm in permutations(&chosen) {
            if perm.is_empty() {
                branches.push(json!({"type": "const_string", "value": ""}));
                continue;
            }
            let mut elements = Vec::with_capacity(perm.len() * 2 - 1);
            for (i, &idx) in perm.iter().enumerate() {
                if i > 0 {
                    elements.push(json!({"type": "const_string", "value": ","}));
                }
                let (key, value_format, _) = &slots[idx];
                elements.push(json!({
                    "type": "sequence",
                    "elements": [
                        {"type": "const_string", "value": format!("{key}:")},
                        value_format,
                    ],
                }));
            }
            branches.push(json!({"type": "sequence", "elements": elements}));
        }
    }

    // The parser's regex requires a literal `{...}` wrapper right after the function name
    // (`call:name{args}`) -- with the free-form `any_text` body this came along for free
    // since the model always writes it as ordinary text; this exact grammar replaces that
    // body outright, so the braces must be put back explicitly or the call becomes invisible
    // to the parser (regex miss -> silently dropped as "markup present, suppressing") even
    // though the grammar itself was satisfied.
    let body = if branches.len() == 1 {
        branches.into_iter().next().unwrap()
    } else {
        json!({"type": "or", "elements": branches})
    };

    Some(json!({
        "type": "sequence",
        "elements": [
            {"type": "const_string", "value": "{"},
            body,
            {"type": "const_string", "value": "}"},
        ],
    }))
}

/// Content grammar for one tool call's `{key:value,...}` body.
///
/// Free-form as always (arguments are not JSON; see the module docs) UNLESS the tool has a
/// REQUIRED argument to protect. Two tiers, from strongest guarantee to weakest:
///
/// 1. `exact_args_grammar` — when every property is simple (string/integer/number/boolean) and
///    there are few enough to permute, the body grammar exactly matches the schema: required
///    keys MUST appear (non-empty if string), optional keys may or may not, any order. Both
///    observed shapes of `EMPTY_REQUIRED_ARG` become unreachable strings.
/// 2. Otherwise, if there is still a required STRING property (e.g. `transfer_call.summary`
///    alongside the `fields` object this fix does not attempt to constrain): a `dispatch` rule
///    that reacts once the model writes `key:<|"|>` for that key and requires at least one
///    character before the closing delimiter. This does NOT force the key to appear at all —
///    only that model's known real failure (an empty value) becomes unreachable once it does.
/// 3. No required string property at all: unchanged `any_text`, as before this fix.
fn gemma4_tool_args_content(parameters: Option<&Value>) -> Value {
    let any_text_fallback = || json!({"type": "any_text", "excludes": []});
    let Some(params) = parameters else {
        return any_text_fallback();
    };

    if let Some(properties) = params.get("properties").and_then(|p| p.as_object()) {
        let required: Vec<String> = params
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).map(str::to_string).collect())
            .unwrap_or_default();
        if let Some(exact) = exact_args_grammar(properties, &required) {
            return exact;
        }
    }

    let required_strings = required_nonempty_string_keys(parameters);
    if required_strings.is_empty() {
        return any_text_fallback();
    }

    let nonempty_value = json!({
        "type": "sequence",
        "elements": [
            {"type": "regex", "pattern": "[\\s\\S]+"},
            {"type": "const_string", "value": GEMMA4_STRING_DELIM},
        ],
    });

    // Two trigger spellings per key (no space / one space after `:`) since both are legal
    // per the parser's `skip_whitespace` and observed in practice; anything wider (multiple
    // spaces, a newline) falls through to the free-form default, same as before this change.
    let rules: Vec<Value> = required_strings
        .iter()
        .flat_map(|key| {
            [
                json!([format!("{key}:{GEMMA4_STRING_DELIM}"), nonempty_value]),
                json!([format!("{key}: {GEMMA4_STRING_DELIM}"), nonempty_value]),
            ]
        })
        .collect();

    json!({
        "type": "dispatch",
        "rules": rules,
        "loop": true,
        "excludes": [],
    })
}

fn gemma4_tool_tag(tool: &ToolDefinition) -> Value {
    json!({
        "type": "tag",
        "begin": format!("{GEMMA4_TOOL_CALL_BEGIN}{}", tool.name),
        "content": gemma4_tool_args_content(tool.parameters.as_ref()),
        "end": GEMMA4_TOOL_CALL_END,
    })
}

/// Tool-call branch: one **optional slot per tool**, so every tool may appear at most once.
///
/// The repetition loop and a multi-tool turn are different shapes and must not be conflated:
///
/// ```text
/// loop       = the SAME tool 26-62x until finish_reason=length   -> must be impossible
/// multi-tool = DISTINCT tools, once each (transfer + hangup)     -> must be possible
/// ```
///
/// `tags_with_separator` cannot express that. Without `stop_after_first` it means "one or
/// more", so 50 identical calls are an accepting string and a second `<|tool_call>` stays a
/// reachable prefix — the production loop. With `stop_after_first` the second call is banned
/// outright, which also makes `transfer_to_agent` + `end_call` in one turn unsamplable, and
/// that is a required behaviour. A sequence of optionals is exactly the right language: no
/// tool can repeat, any subset of distinct tools is legal.
///
/// `at_least_one == false` also matches the empty string, which is correct *after* an envelope
/// (the envelope alone is a complete turn). As a standalone branch that would make empty
/// output — dead air — legal, so that case is built as one alternative per tool with that
/// tool's slot mandatory.
///
/// Caveat: the slots are ordered, so calls arrive in the order the tools were declared in the
/// request. The caller controls that order; permuting would cost one branch per ordering.
fn gemma4_tool_calls(tools: &[ToolDefinition], at_least_one: bool) -> Value {
    let optional = |tool: &ToolDefinition| json!({"type": "optional", "content": gemma4_tool_tag(tool)});

    if !at_least_one {
        return json!({
            "type": "sequence",
            "elements": tools.iter().map(optional).collect::<Vec<_>>(),
        });
    }

    let branches: Vec<Value> = (0..tools.len())
        .map(|i| {
            let mut elements: Vec<Value> = tools[..i].iter().map(optional).collect();
            elements.push(gemma4_tool_tag(&tools[i])); // required => branch is non-empty
            elements.extend(tools[i + 1..].iter().map(optional));
            json!({"type": "sequence", "elements": elements})
        })
        .collect();

    if branches.len() == 1 {
        branches.into_iter().next().unwrap()
    } else {
        json!({"type": "or", "elements": branches})
    }
}

/// Exactly one tool call, chosen from the offered tools. `stop_after_first` is what makes the
/// second `<|tool_call>` unsamplable, so neither a repetition loop nor an unrequested extra tool
/// can follow the first call.
fn gemma4_single_tool_call(tools: &[ToolDefinition]) -> Value {
    json!({
        "type": "tags_with_separator",
        "tags": tools.iter().map(gemma4_tool_tag).collect::<Vec<_>>(),
        "separator": "",
        "at_least_one": true,
        "stop_after_first": true,
    })
}

/// Build the Gemma 4 tool-call structural tag.
///
/// * `tools` — tools the model may call, name + JSON-schema parameters. Empty returns `None`.
/// * `content_schema` — the caller's `response_format` JSON schema, if any. When
///   present it becomes a branch of the union so the schema guarantee survives.
/// * `tools_mandatory` — `true` for `tool_choice: "required"` or a named choice:
///   a message may precede a call but must not stand alone.
/// * `allow_reasoning` — permit an optional leading thinking block.
/// * `allow_tool_only_turn` — permit a turn that is tool calls with **no** envelope. Off by
///   default: for a `json_schema` caller the envelope is what gets spoken, so a tool-only turn
///   is silence on the wire. Repetition is impossible either way (see `gemma4_tool_calls`), so
///   this is no longer about how many calls a turn may carry.
/// * `allow_parallel_calls` — permit more than one DISTINCT tool call in the same turn (e.g.
///   `transfer_to_agent` + `end_call`). Off by default and pinned to a single call, because
///   offering every tool as an independently-fillable slot measurably increases how often the
///   model calls one it did not need — including a call-ending tool mid-conversation. Turning
///   this on is an explicit request from the caller (mirrors OpenAI's `parallel_tool_calls`);
///   it is not inferred from the tool list. Still no repetition of the SAME tool either way.
/// * `prompt_opened_thought` — the chat template left the prompt inside an open thought
///   channel, so the completion emits only the closer. Off unless known; see below.
pub fn gemma4_structural_tag(
    tools: &[ToolDefinition],
    content_schema: Option<&Value>,
    tools_mandatory: bool,
    allow_reasoning: bool,
    allow_tool_only_turn: bool,
    allow_parallel_calls: bool,
    prompt_opened_thought: bool,
) -> Option<Value> {
    if tools.is_empty() {
        return None;
    }

    // A forced tool choice must not offer a content branch. The schema object is a legal
    // *prefix* of "content then tool call", so the model writes it and then ends the turn
    // with a special token — which the grammar cannot mask — leaving the demanded call
    // unmade. OpenAI semantics agree: response_format constrains content, and a forced
    // choice produces a tool call rather than content.
    let content_schema = if tools_mandatory { None } else { content_schema };

    // AT MOST one tool call per turn, unless the caller explicitly asked for parallel calls.
    //
    // Both halves of the single-call form matter and each was got wrong once. Without
    // `stop_after_first` the model fills every available slot (it appended end_call to ordinary
    // turns 32 times in 4 conversations). Without the `optional` wrapper the tool call becomes
    // MANDATORY -- `at_least_one` inside a bare `sequence` means the grammar cannot finish the
    // turn until a tool has been called, so the model called one on 100% of turns, including
    // `end_call` on a silence check. A speech turn must be able to end at the envelope.
    //
    // The per-tool-slots form (`gemma4_tool_calls(names, at_least_one)`) is what makes a
    // multi-tool turn (transfer_to_agent + end_call) expressible, but it reintroduces the same
    // over-calling measured above -- confirmed again with a terminal-tool policy in the chat
    // template active, which was not enough to make slots safe by default. So it is reachable
    // only via `allow_parallel_calls`, an explicit signal from the caller, not inferred from the
    // tool list or the schema.
    let one_tool_call = |at_least_one: bool| {
        if allow_parallel_calls {
            gemma4_tool_calls(tools, at_least_one)
        } else if at_least_one {
            gemma4_single_tool_call(tools)
        } else {
            json!({"type": "optional", "content": gemma4_single_tool_call(tools)})
        }
    };

    let body = match content_schema {
        Some(schema) => {
            let content = json!({
                "type": "json_schema",
                "json_schema": schema,
                "style": "json",
            });
            // `one_tool_call(false)` already covers "envelope alone" (all its slots are
            // optional / it is itself optional), so this one branch is enough unless a silent
            // tool-only turn must also be offered.
            let mut elements = vec![content, one_tool_call(false)];

            // A tool-only turn carries no envelope, and for a json_schema caller the envelope
            // *is* what gets spoken — so that shape is silence. Measured on the stress matrix:
            // with the tool-only branch available the model took it on every co-emission turn
            // (holding line, end_call, transfer), losing the spoken line each time even though
            // envelope+tool was permitted. Offering it makes the envelope a preference the
            // model can decline; withholding it makes the envelope structural.
            //
            // Set `allow_tool_only_turn` only for a caller that genuinely wants a silent
            // tool turn.
            if allow_tool_only_turn {
                let with_tools = json!({"type": "sequence", "elements": elements});
                json!({
                    "type": "or",
                    "elements": [with_tools, one_tool_call(true)],
                })
            } else {
                elements[1] = one_tool_call(false);
                json!({"type": "sequence", "elements": elements})
            }
        }
        // No content constraint to preserve: the native tag alone is enough, and
        // it is still needed so a forced choice is not pushed into the generic
        // JSON tool-call shape that Gemma 4 does not speak. A forced choice is always pinned to
        // one call regardless of `allow_parallel_calls` -- OpenAI's `parallel_tool_calls` is
        // about batching independent calls, not about what a single demanded call may do.
        None => {
            if tools_mandatory {
                gemma4_single_tool_call(tools)
            } else {
                one_tool_call(true)
            }
        }
    };

    if !allow_reasoning {
        return Some(json!({"type": "structural_tag", "format": body}));
    }

    // Generation may begin inside the thought channel two ways: the model opens the block
    // itself, or the chat template left the prompt inside an already-open block so only the
    // closer is emitted (the post-tool-response continuation).
    //
    // An unconstrained opener ("begin": "") covers both, but it accepts arbitrary text up to
    // the closer — so it is not a prefix on the grammar, it is a hole through it. With that
    // branch present, *any* leading text is a legal prefix and the content schema stops
    // constraining the start of the turn; because special tokens escape the grammar mask, the
    // model can emit that text and end its turn. Measured on gemma-4-31B-it: under a forced
    // choice the model wrote the schema object and skipped the demanded call, and on the
    // unforced multi-turn tool follow-up it returned a ```json fence, plain prose, or a
    // truncated object — none of which a strict json_schema consumer can parse.
    //
    // Both failures are the same hole, so both paths now constrain the opener, which leaves
    // `<|tool_call>call:` and the schema object as the only legal starts. The continuation
    // case is real but must be *known*, not assumed: it is opt-in via `prompt_opened_thought`,
    // because assuming it on every request is what broke the follow-up turn.
    let thought_begin = if prompt_opened_thought {
        ""
    } else {
        GEMMA4_THOUGHT_BEGIN
    };
    let reasoning_prefix = json!({
        "type": "optional",
        "content": {
            "type": "tag",
            "begin": thought_begin,
            "content": {"type": "any_text", "excludes": []},
            "end": GEMMA4_REASONING_END,
        },
    });

    Some(json!({
        "type": "structural_tag",
        "format": {"type": "sequence", "elements": [reasoning_prefix, body]},
    }))
}

/// Whether this tool-call parser has a native structural tag.
pub fn parser_has_structural_tag(parser: Option<&str>) -> bool {
    matches!(parser, Some("gemma4") | Some("gemma-4"))
}

/// Build the tag for `parser`, or `None` when it has no native format.
#[allow(clippy::too_many_arguments)]
pub fn structural_tag_for_parser(
    parser: Option<&str>,
    tools: &[ToolDefinition],
    content_schema: Option<&Value>,
    tools_mandatory: bool,
    allow_reasoning: bool,
    allow_tool_only_turn: bool,
    allow_parallel_calls: bool,
    prompt_opened_thought: bool,
) -> Option<Value> {
    if !parser_has_structural_tag(parser) {
        return None;
    }
    gemma4_structural_tag(
        tools,
        content_schema,
        tools_mandatory,
        allow_reasoning,
        allow_tool_only_turn,
        allow_parallel_calls,
        prompt_opened_thought,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<ToolDefinition> {
        vec![
            ToolDefinition { name: "fetch_seller_details".to_string(), parameters: None },
            ToolDefinition { name: "hangup_call".to_string(), parameters: None },
        ]
    }

    fn names() -> Vec<String> {
        tools().into_iter().map(|t| t.name).collect()
    }

    fn schema() -> Value {
        json!({
            "type": "object",
            "properties": {"assistant_reply": {"type": "string"}},
            "required": ["assistant_reply"],
        })
    }

    #[test]
    fn no_tools_yields_no_tag() {
        assert!(gemma4_structural_tag(&[], Some(&schema()), false, true, false, false, false).is_none());
    }

    /// Every tool slot is an `optional` wrapping a single `tag`, and `tags_with_separator` is
    /// gone — that combination is what makes repetition unrepresentable rather than merely
    /// discouraged, so assert it structurally instead of trusting a flag.
    fn assert_no_repeatable_branch(tag: &Value) {
        // Every tool branch must be pinned to a single call, so no branch can repeat a tool.
        let dumped = serde_json::to_string(tag).unwrap();
        for (i, _) in dumped.match_indices("tags_with_separator") {
            assert!(
                dumped[i..].contains("\"stop_after_first\":true"),
                "a repeatable tool branch survived: {dumped}"
            );
        }
    }

    #[test]
    fn schema_and_tools_require_the_envelope_by_default() {
        // Envelope first, tool slots after: a tool-only turn is silence for a json_schema
        // caller, so it is not offered unless the caller asks for it.
        let tag =
            gemma4_structural_tag(&tools(), Some(&schema()), false, false, false, false, false).unwrap();
        assert_eq!(tag["type"], "structural_tag");
        assert_eq!(tag["format"]["type"], "sequence");
        assert_eq!(tag["format"]["elements"][0]["type"], "json_schema");
        // at most one call: optional wrapper (so a speech turn can stop) around a
        // stop_after_first branch (so a second call is unsamplable)
        assert_eq!(tag["format"]["elements"][1]["type"], "optional");
        assert_eq!(tag["format"]["elements"][1]["content"]["type"], "tags_with_separator");
        assert_eq!(tag["format"]["elements"][1]["content"]["stop_after_first"], true);
    }

    #[test]
    fn a_tool_only_turn_is_opt_in() {
        let tag =
            gemma4_structural_tag(&tools(), Some(&schema()), false, false, true, false, false).unwrap();
        assert_eq!(tag["format"]["type"], "or");
        assert_no_repeatable_branch(&tag);
    }

    #[test]
    fn every_offered_tool_is_reachable_as_the_one_call() {
        // Any offered tool may be THE call for the turn, and only one call is possible.
        let tag = gemma4_structural_tag(&tools(), Some(&schema()), false, false, false, false, false).unwrap();
        let branch = &tag["format"]["elements"][1]["content"];
        assert_eq!(branch["stop_after_first"], true);
        let tags = branch["tags"].as_array().unwrap();
        assert_eq!(tags.len(), names().len());
        for (t, name) in tags.iter().zip(names()) {
            assert_eq!(t["begin"], format!("{GEMMA4_TOOL_CALL_BEGIN}{name}"));
        }
    }

    #[test]
    fn forced_choice_drops_the_content_branch() {
        // The schema object is a legal prefix of "content then tool call", so offering it
        // lets the model write content and end the turn with a special token the grammar
        // cannot mask, leaving the demanded call unmade. Forced choice therefore admits
        // tool calls only — which is also what response_format means in OpenAI's API.
        let tag = gemma4_structural_tag(&tools(), Some(&schema()), true, false, false, false, false).unwrap();
        // tool calls only, pinned to exactly one
        assert_eq!(tag["format"]["type"], "tags_with_separator");
        assert_eq!(tag["format"]["stop_after_first"], true);
    }

    #[test]
    fn without_schema_the_native_tag_is_used_alone() {
        let tag = gemma4_structural_tag(&tools(), None, true, false, false, false, false).unwrap();
        assert_eq!(tag["format"]["type"], "tags_with_separator");
    }

    #[test]
    fn reasoning_prefix_is_optional_and_wraps_the_body() {
        let tag = gemma4_structural_tag(&tools(), None, false, true, false, false, false).unwrap();
        assert_eq!(tag["format"]["type"], "sequence");
        assert_eq!(tag["format"]["elements"][0]["type"], "optional");
        assert_eq!(
            tag["format"]["elements"][0]["content"]["end"],
            GEMMA4_REASONING_END
        );
    }

    #[test]
    fn tool_names_are_constrained_but_arguments_are_not() {
        let tag = gemma4_structural_tag(&tools(), None, false, false, false, false, false).unwrap();
        // no schema -> tool calls only, as an `or` of per-tool branches; branch 0 starts with
        // the first tool required.
        let first = &tag["format"]["tags"][0];
        assert_eq!(first["begin"], "<|tool_call>call:fetch_seller_details");
        assert_eq!(first["content"]["type"], "any_text");
        assert_eq!(first["end"], GEMMA4_TOOL_CALL_END);
    }

    #[test]
    fn the_reasoning_opener_is_always_constrained() {
        // An unconstrained opener ("begin": "") accepts arbitrary text before the closer, so
        // it dissolves the rest of the grammar: the model can emit prose or a ```json fence
        // and end the turn. Forced and unforced turns both constrain it.
        for mandatory in [true, false] {
            let tag =
                gemma4_structural_tag(&tools(), None, mandatory, true, false, false, false).unwrap();
            assert_eq!(
                tag["format"]["elements"][0]["content"]["begin"],
                GEMMA4_THOUGHT_BEGIN,
                "tools_mandatory={mandatory} must not leave the opener unconstrained"
            );
        }
    }

    #[test]
    fn prompt_opened_thought_is_opt_in() {
        // The continuation case is real, but only correct when the prompt genuinely left the
        // channel open — assuming it on every request is what broke the tool follow-up.
        let tag = gemma4_structural_tag(&tools(), None, false, true, false, false, true).unwrap();
        assert_eq!(tag["format"]["elements"][0]["content"]["begin"], "");
    }

    #[test]
    fn repetition_is_unrepresentable_on_every_path() {
        // The production loop is the SAME tool 26-62x until finish_reason=length. With one
        // optional slot per tool there is no branch that can emit a tool twice, forced or not,
        // with or without a schema, tool-only offered or not.
        for mandatory in [true, false] {
            for tool_only in [true, false] {
                for sch in [Some(schema()), None] {
                    let tag = gemma4_structural_tag(
                        &tools(), sch.as_ref(), mandatory, false, tool_only, false, false,
                    )
                    .unwrap();
                    assert_no_repeatable_branch(&tag);
                }
            }
        }
    }

    #[test]
    fn allow_parallel_calls_is_opt_in_and_still_forbids_repetition() {
        // Default: pinned to one call, so a multi-tool turn is unreachable.
        let single = gemma4_structural_tag(
            &tools(), Some(&schema()), false, false, false, false, false,
        )
        .unwrap();
        let dumped = serde_json::to_string(&single).unwrap();
        assert!(dumped.contains("\"stop_after_first\":true"));

        // Opted in: distinct tools become reachable together...
        let parallel = gemma4_structural_tag(
            &tools(), Some(&schema()), false, false, false, true, false,
        )
        .unwrap();
        let slots = parallel["format"]["elements"][1]["elements"].as_array().unwrap();
        assert_eq!(slots.len(), names().len());
        // ...but the SAME tool still cannot repeat: no tags_with_separator without
        // stop_after_first, and each tool appears in exactly one optional slot.
        assert_no_repeatable_branch(&parallel);
        for (slot, name) in slots.iter().zip(names()) {
            assert_eq!(slot["type"], "optional");
            assert_eq!(slot["content"]["begin"], format!("{GEMMA4_TOOL_CALL_BEGIN}{name}"));
        }

        // A forced choice ignores allow_parallel_calls -- it is still pinned to one call.
        let forced = gemma4_structural_tag(
            &tools(), Some(&schema()), true, false, false, true, false,
        )
        .unwrap();
        assert_eq!(forced["format"]["type"], "tags_with_separator");
        assert_eq!(forced["format"]["stop_after_first"], true);
    }

    fn lookup_tool_with_required_query() -> ToolDefinition {
        ToolDefinition {
            name: "lookup_info".to_string(),
            parameters: Some(json!({
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            })),
        }
    }

    /// Every string in `strings` must be a reachable prefix-consistent match against `format`'s
    /// dumped shape: since these tests don't have an xgrammar matcher available, they instead
    /// walk the JSON tree checking that at least one branch's literal path spells out `s`. This
    /// is the same "does this shape survive in the tree" style as `assert_no_repeatable_branch`.
    fn any_branch_contains(format: &Value, needle: &str) -> bool {
        serde_json::to_string(format).unwrap().contains(needle)
    }

    #[test]
    fn simple_single_required_string_arg_gets_the_exact_grammar() {
        // `lookup_info`'s real shape (one required string property, nothing else) hits the
        // strongest tier: the grammar for the whole body is just `query:<|"|>` + 1+ chars +
        // `<|"|>`, so BOTH observed EMPTY_REQUIRED_ARG shapes are unreachable strings --
        // `lookup_info{}` (key omitted) and `lookup_info{query:<|"|><|"|>}` (key present, empty).
        let content = gemma4_tool_args_content(lookup_tool_with_required_query().parameters.as_ref());
        assert_eq!(content["type"], "sequence");
        let dumped = serde_json::to_string(&content).unwrap();
        assert!(dumped.contains("\"query:\""), "required key must be a literal, not optional: {dumped}");
        // GEMMA4_STRING_DELIM's `"` comes back JSON-escaped (`\"`) in the dumped text.
        assert!(dumped.contains("<|\\\"|>"), "string delimiter must appear in the value format: {dumped}");
        // one_or_more, never zero_or_more -- the empty string must not be a legal value.
        assert!(any_branch_contains(&content, "[\\\\s\\\\S]+"));
        assert!(!dumped.contains("any_text"), "must not fall back to the unconstrained body: {dumped}");
    }

    #[test]
    fn required_and_optional_mix_permutes_but_keeps_required_mandatory() {
        // Two required (string, integer) + one optional (string): every accepted branch must
        // contain both required keys; branches differ only in order and whether `note` appears.
        let mixed = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
                "note": {"type": "string"},
            },
            "required": ["name", "count"],
        });
        let content = gemma4_tool_args_content(Some(&mixed));
        // Whole body is wrapped `{ <or-of-branches> }` so the call still parses as `name{args}`.
        assert_eq!(content["type"], "sequence");
        assert_eq!(content["elements"][0]["value"], "{");
        assert_eq!(content["elements"][2]["value"], "}");
        let or = &content["elements"][1];
        assert_eq!(or["type"], "or");
        let branches = or["elements"].as_array().unwrap();
        assert!(branches.len() > 1, "order + optional-subset should produce more than one branch");
        for branch in branches {
            let dumped = serde_json::to_string(branch).unwrap();
            assert!(dumped.contains("\"name:\""), "every branch must require `name`: {dumped}");
            assert!(dumped.contains("\"count:\""), "every branch must require `count`: {dumped}");
        }
        // `note` appears in at least one branch (optional = sometimes present)...
        assert!(branches.iter().any(|b| serde_json::to_string(b).unwrap().contains("\"note:\"")));
        // ...but not in every branch (optional = not mandatory).
        assert!(branches.iter().any(|b| !serde_json::to_string(b).unwrap().contains("\"note:\"")));
    }

    #[test]
    fn complex_schema_falls_back_to_the_dispatch_patch() {
        // `transfer_call`'s real shape: a required STRING (`summary`) alongside a required
        // OBJECT (`fields`) the exact-grammar tier does not attempt. Falls back to tier 2 --
        // weaker (does not force `summary` to appear) but still closes the empty-value case
        // once it does.
        let transfer_like = json!({
            "type": "object",
            "properties": {
                "fields": {"type": "object", "properties": {"a": {"type": "string"}}},
                "summary": {"type": "string"},
            },
            "required": ["fields", "summary"],
        });
        let content = gemma4_tool_args_content(Some(&transfer_like));
        assert_eq!(content["type"], "dispatch");
        let rules = content["rules"].as_array().unwrap();
        assert!(rules.iter().any(|r| r[0] == "summary:<|\"|>"));
        assert!(!rules.iter().any(|r| r[0].as_str().unwrap_or_default().starts_with("fields:")));
    }

    #[test]
    fn too_many_properties_falls_back_to_the_dispatch_patch() {
        let many = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"}, "b": {"type": "string"}, "c": {"type": "string"},
                "d": {"type": "string"}, "e": {"type": "string"},
            },
            "required": ["a"],
        });
        assert_eq!(gemma4_tool_args_content(Some(&many))["type"], "dispatch");
    }

    #[test]
    fn tool_without_required_string_args_keeps_any_text() {
        // No schema at all, or a schema with no required string property: unchanged from
        // before this fix, so nothing about existing (schema-less) tools regresses.
        assert_eq!(gemma4_tool_args_content(None), json!({"type": "any_text", "excludes": []}));

        // Zero-arg tools (fetch_details, transfer_to_agent, fetch_seller_details in the real
        // test surface) have an empty `properties` object -- must stay any_text too, since
        // whether the model calls a zero-arg tool at all is a judgment question this fix does
        // not touch, only what's inside a call's arguments once made.
        let zero_arg = json!({"type": "object", "properties": {}, "required": []});
        assert_eq!(
            gemma4_tool_args_content(Some(&zero_arg)),
            json!({"type": "any_text", "excludes": []})
        );

        let optional_only = json!({
            "type": "object",
            "properties": {"note": {"type": "string"}},
            "required": [],
        });
        // All-optional still gets the exact tier (it must permit, but not require, `note`) --
        // confirm the empty body is still one of the reachable branches.
        let content = gemma4_tool_args_content(Some(&optional_only));
        assert!(
            any_branch_contains(&content, "\"\""),
            "an all-optional schema must still accept an empty body: {content}"
        );

        let required_non_string = json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        });
        // Required but non-string: exact tier still applies (integers get a numeric regex, not
        // an emptiness concern), so this is no longer `any_text` either -- only a genuinely
        // schema-less tool, or one this fix's tiers both decline, keeps the old fallback.
        assert_ne!(
            gemma4_tool_args_content(Some(&required_non_string))["type"],
            "any_text"
        );
    }

    #[test]
    fn tool_tag_threads_the_schema_into_its_content() {
        // End to end from `ToolDefinition` through `gemma4_tool_tag`, not just the helper.
        let tag = gemma4_tool_tag(&lookup_tool_with_required_query());
        assert_eq!(tag["begin"], "<|tool_call>call:lookup_info");
        assert_eq!(tag["content"]["type"], "sequence");
    }

    #[test]
    fn exact_grammar_wraps_the_body_in_literal_braces() {
        // The parser's regex requires `call:name{args}` -- literal braces right after the
        // name and right before the end tag. The exact-grammar tier fully replaces the
        // free-form body (unlike the dispatch fallback, where braces ride along as ordinary
        // text), so it must put them back explicitly or a grammar-valid completion becomes
        // invisible to the parser (regex miss -> silently dropped).
        let content = gemma4_tool_args_content(lookup_tool_with_required_query().parameters.as_ref());
        assert_eq!(content["type"], "sequence");
        assert_eq!(content["elements"][0], json!({"type": "const_string", "value": "{"}));
        assert_eq!(content["elements"][2], json!({"type": "const_string", "value": "}"}));
    }

    #[test]
    fn only_gemma4_has_a_tag() {
        assert!(parser_has_structural_tag(Some("gemma4")));
        assert!(parser_has_structural_tag(Some("gemma-4")));
        assert!(!parser_has_structural_tag(Some("hermes")));
        assert!(!parser_has_structural_tag(None));
        assert!(
            structural_tag_for_parser(Some("hermes"), &tools(), None, false, false, false, false, false).is_none()
        );
    }
}
