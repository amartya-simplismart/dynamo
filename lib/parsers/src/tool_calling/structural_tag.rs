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

use serde_json::{Value, json};

/// Gemma 4 channel markers, as they appear in decoded text.
pub const GEMMA4_TOOL_CALL_BEGIN: &str = "<|tool_call>call:";
pub const GEMMA4_TOOL_CALL_END: &str = "<tool_call|>";
pub const GEMMA4_TOOL_CALL_TRIGGER: &str = "<|tool_call>";
pub const GEMMA4_REASONING_END: &str = "<channel|>";
pub const GEMMA4_THOUGHT_BEGIN: &str = "<|channel>thought\n";

fn gemma4_tool_tag(name: &str) -> Value {
    json!({
        "type": "tag",
        "begin": format!("{GEMMA4_TOOL_CALL_BEGIN}{name}"),
        "content": {"type": "any_text", "excludes": []},
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
fn gemma4_tool_calls(tool_names: &[String], at_least_one: bool) -> Value {
    let optional = |name: &String| json!({"type": "optional", "content": gemma4_tool_tag(name)});

    if !at_least_one {
        return json!({
            "type": "sequence",
            "elements": tool_names.iter().map(optional).collect::<Vec<_>>(),
        });
    }

    let branches: Vec<Value> = (0..tool_names.len())
        .map(|i| {
            let mut elements: Vec<Value> = tool_names[..i].iter().map(optional).collect();
            elements.push(gemma4_tool_tag(&tool_names[i])); // required => branch is non-empty
            elements.extend(tool_names[i + 1..].iter().map(optional));
            json!({"type": "sequence", "elements": elements})
        })
        .collect();

    if branches.len() == 1 {
        branches.into_iter().next().unwrap()
    } else {
        json!({"type": "or", "elements": branches})
    }
}

/// Build the Gemma 4 tool-call structural tag.
///
/// * `tool_names` — tools the model may call. Empty returns `None`.
/// * `content_schema` — the caller's `response_format` JSON schema, if any. When
///   present it becomes a branch of the union so the schema guarantee survives.
/// * `tools_mandatory` — `true` for `tool_choice: "required"` or a named choice:
///   a message may precede a call but must not stand alone.
/// * `allow_reasoning` — permit an optional leading thinking block.
/// * `allow_tool_only_turn` — permit a turn that is tool calls with **no** envelope. Off by
///   default: for a `json_schema` caller the envelope is what gets spoken, so a tool-only turn
///   is silence on the wire. Repetition is impossible either way (see `gemma4_tool_calls`), so
///   this is no longer about how many calls a turn may carry.
/// * `prompt_opened_thought` — the chat template left the prompt inside an open thought
///   channel, so the completion emits only the closer. Off unless known; see below.
pub fn gemma4_structural_tag(
    tool_names: &[String],
    content_schema: Option<&Value>,
    tools_mandatory: bool,
    allow_reasoning: bool,
    allow_tool_only_turn: bool,
    prompt_opened_thought: bool,
) -> Option<Value> {
    if tool_names.is_empty() {
        return None;
    }

    // A forced tool choice must not offer a content branch. The schema object is a legal
    // *prefix* of "content then tool call", so the model writes it and then ends the turn
    // with a special token — which the grammar cannot mask — leaving the demanded call
    // unmade. OpenAI semantics agree: response_format constrains content, and a forced
    // choice produces a tool call rather than content.
    let content_schema = if tools_mandatory { None } else { content_schema };

    let body = match content_schema {
        Some(schema) => {
            let content = json!({
                "type": "json_schema",
                "json_schema": schema,
                "style": "json",
            });
            // Optional slots, so this one branch already covers "envelope alone".
            let mut elements = vec![content, gemma4_tool_calls(tool_names, false)];

            // A tool-only turn carries no envelope, and for a json_schema caller the envelope
            // *is* what gets spoken — so that shape is silence. Measured on the stress matrix:
            // with the tool-only branch available the model took it on every co-emission turn
            // (holding line, end_call, transfer), losing the spoken line each time even though
            // envelope+tool was permitted. Offering it makes the envelope a preference the
            // model can decline; withholding it makes the envelope structural.
            //
            // Set `allow_tool_only_turn` only for a caller that genuinely wants a silent
            // tool turn.
            // EXCLUSIVE: the envelope OR tool calls, never both in one turn.
            //
            // Co-emission (envelope followed by optional tool slots) looks harmless but it
            // builds a funnel: once the envelope closes, the only continuations the grammar
            // offers are tool calls or EOS, so the model's "keep going" probability lands on a
            // tool call. Measured on gemma-4-31B-it with a json_schema present:
            //
            //   tools [lookup, end_call]  -> BOTH called, 3/3, on a turn needing neither
            //   tools [end_call] alone    -> end_call called 3/3 on "things are going okay"
            //   same tools, NO schema     -> no tool calls at all, 3/3
            //
            // i.e. the schema itself was driving tool emission, and a spurious `end_call`
            // hangs up a live call. With the branches exclusive, nothing follows the envelope,
            // so a spurious call after it is unsamplable. A tool turn is then tool-only and the
            // envelope arrives on the follow-up request, which the runtime already handles.
            let _ = &mut elements;
            json!({
                "type": "or",
                "elements": [elements[0].clone(), gemma4_tool_calls(tool_names, true)],
            })
        }
        // No content constraint to preserve: the native tag alone is enough, and
        // it is still needed so a forced choice is not pushed into the generic
        // JSON tool-call shape that Gemma 4 does not speak.
        None => gemma4_tool_calls(tool_names, true),
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
    tool_names: &[String],
    content_schema: Option<&Value>,
    tools_mandatory: bool,
    allow_reasoning: bool,
    allow_tool_only_turn: bool,
    prompt_opened_thought: bool,
) -> Option<Value> {
    if !parser_has_structural_tag(parser) {
        return None;
    }
    gemma4_structural_tag(
        tool_names,
        content_schema,
        tools_mandatory,
        allow_reasoning,
        allow_tool_only_turn,
        prompt_opened_thought,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names() -> Vec<String> {
        vec!["fetch_seller_details".to_string(), "hangup_call".to_string()]
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
        assert!(gemma4_structural_tag(&[], Some(&schema()), false, true, false, false).is_none());
    }

    /// Every tool slot is an `optional` wrapping a single `tag`, and `tags_with_separator` is
    /// gone — that combination is what makes repetition unrepresentable rather than merely
    /// discouraged, so assert it structurally instead of trusting a flag.
    fn assert_no_repeatable_branch(tag: &Value) {
        let dumped = serde_json::to_string(tag).unwrap();
        assert!(
            !dumped.contains("tags_with_separator"),
            "tags_with_separator can express 'one or more' and must not appear: {dumped}"
        );
        assert!(!dumped.contains("stop_after_first"), "stale flag: {dumped}");
    }

    #[test]
    fn schema_and_tools_are_mutually_exclusive() {
        // The envelope OR tool calls, never both. Co-emission funnels the model into a tool
        // call after the envelope closes (measured: a spurious end_call on a benign turn), so
        // nothing may follow the envelope.
        let tag =
            gemma4_structural_tag(&names(), Some(&schema()), false, false, false, false).unwrap();
        assert_eq!(tag["type"], "structural_tag");
        assert_eq!(tag["format"]["type"], "or");
        assert_eq!(tag["format"]["elements"][0]["type"], "json_schema");
        assert_eq!(tag["format"]["elements"][1]["type"], "or"); // per-tool branches
        assert_no_repeatable_branch(&tag);
    }

    #[test]
    fn a_tool_only_turn_is_opt_in() {
        let tag =
            gemma4_structural_tag(&names(), Some(&schema()), false, false, true, false).unwrap();
        assert_eq!(tag["format"]["type"], "or");
        assert_no_repeatable_branch(&tag);
    }

    #[test]
    fn distinct_tools_each_get_their_own_slot() {
        // transfer + hangup in one turn must stay expressible: within the tool branch there is
        // one slot per tool, so N distinct calls are legal while a repeat of any one is not.
        let tag = gemma4_structural_tag(&names(), Some(&schema()), false, false, false, false).unwrap();
        let branch0 = &tag["format"]["elements"][1]["elements"][0];
        let slots = branch0["elements"].as_array().unwrap();
        assert_eq!(slots.len(), names().len());
        // branch 0 requires the first tool and leaves the rest optional
        assert_eq!(slots[0]["begin"], format!("{GEMMA4_TOOL_CALL_BEGIN}{}", names()[0]));
        assert_eq!(slots[1]["type"], "optional");
        assert_eq!(
            slots[1]["content"]["begin"],
            format!("{GEMMA4_TOOL_CALL_BEGIN}{}", names()[1])
        );
    }

    #[test]
    fn forced_choice_drops_the_content_branch() {
        // The schema object is a legal prefix of "content then tool call", so offering it
        // lets the model write content and end the turn with a special token the grammar
        // cannot mask, leaving the demanded call unmade. Forced choice therefore admits
        // tool calls only — which is also what response_format means in OpenAI's API.
        let tag = gemma4_structural_tag(&names(), Some(&schema()), true, false, false, false).unwrap();
        // tool calls only, and at least one of them: an `or` of per-tool branches
        assert_eq!(tag["format"]["type"], "or");
        assert_no_repeatable_branch(&tag);
    }

    #[test]
    fn without_schema_the_native_tag_is_used_alone() {
        let tag = gemma4_structural_tag(&names(), None, true, false, false, false).unwrap();
        assert_eq!(tag["format"]["type"], "or");
        assert_no_repeatable_branch(&tag);
    }

    #[test]
    fn reasoning_prefix_is_optional_and_wraps_the_body() {
        let tag = gemma4_structural_tag(&names(), None, false, true, false, false).unwrap();
        assert_eq!(tag["format"]["type"], "sequence");
        assert_eq!(tag["format"]["elements"][0]["type"], "optional");
        assert_eq!(
            tag["format"]["elements"][0]["content"]["end"],
            GEMMA4_REASONING_END
        );
    }

    #[test]
    fn tool_names_are_constrained_but_arguments_are_not() {
        let tag = gemma4_structural_tag(&names(), None, false, false, false, false).unwrap();
        // no schema -> tool calls only, as an `or` of per-tool branches; branch 0 starts with
        // the first tool required.
        let first = &tag["format"]["elements"][0]["elements"][0];
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
                gemma4_structural_tag(&names(), None, mandatory, true, false, false).unwrap();
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
        let tag = gemma4_structural_tag(&names(), None, false, true, false, true).unwrap();
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
                        &names(), sch.as_ref(), mandatory, false, tool_only, false,
                    )
                    .unwrap();
                    assert_no_repeatable_branch(&tag);
                }
            }
        }
    }

    #[test]
    fn only_gemma4_has_a_tag() {
        assert!(parser_has_structural_tag(Some("gemma4")));
        assert!(parser_has_structural_tag(Some("gemma-4")));
        assert!(!parser_has_structural_tag(Some("hermes")));
        assert!(!parser_has_structural_tag(None));
        assert!(
            structural_tag_for_parser(Some("hermes"), &names(), None, false, false, false, false).is_none()
        );
    }
}
