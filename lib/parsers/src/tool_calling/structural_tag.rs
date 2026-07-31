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

/// Tool-call branch: one tag per tool, at least one call, body unconstrained.
///
/// `tags_with_separator` admits nothing but tool calls, so a forced choice cannot be
/// satisfied by prose. `triggered_tags` is equivalent here (both reject free text) but
/// reads as "text until a trigger", which is not the intent.
fn gemma4_tool_calls(tool_names: &[String], stop_after_first: bool) -> Value {
    let tags: Vec<Value> = tool_names
        .iter()
        .map(|name| {
            json!({
                "type": "tag",
                "begin": format!("{GEMMA4_TOOL_CALL_BEGIN}{name}"),
                "content": {"type": "any_text", "excludes": []},
                "end": GEMMA4_TOOL_CALL_END,
            })
        })
        .collect();

    json!({
        "type": "tags_with_separator",
        "tags": tags,
        "separator": "",
        "at_least_one": true,
        "stop_after_first": stop_after_first,
    })
}

/// Build the Gemma 4 tool-call structural tag.
///
/// * `tool_names` — tools the model may call. Empty returns `None`.
/// * `content_schema` — the caller's `response_format` JSON schema, if any. When
///   present it becomes a branch of the union so the schema guarantee survives.
/// * `tools_mandatory` — `true` for `tool_choice: "required"` or a named choice:
///   a message may precede a call but must not stand alone.
/// * `allow_reasoning` — permit an optional leading thinking block.
/// * `allow_parallel_calls` — permit more than one call in a single response. Off unless
///   the caller opted in; see the repetition note below.
/// * `prompt_opened_thought` — the chat template left the prompt inside an open thought
///   channel, so the completion emits only the closer. Off unless known; see below.
pub fn gemma4_structural_tag(
    tool_names: &[String],
    content_schema: Option<&Value>,
    tools_mandatory: bool,
    allow_reasoning: bool,
    allow_parallel_calls: bool,
    prompt_opened_thought: bool,
) -> Option<Value> {
    if tool_names.is_empty() {
        return None;
    }

    // Repetition must be off by default, forced or not. `tags_with_separator` with
    // `at_least_one` and no `stop_after_first` means "one or more", so N back-to-back calls
    // are all accepting strings and a second `<|tool_call>` stays a reachable prefix the
    // moment the first call closes. Nothing then pushes the model to stop: measured on
    // gemma-4-31B-it, `required` emitted the same call 28 times until max_tokens, and in a
    // multi-turn conversation the *unforced* tool follow-up did the same 31-62 times
    // (finish_reason=length, i.e. a dead turn).
    //
    // Leaving it on for unforced turns "in case a model batches calls" trades a rare
    // convenience for a reproducible hang, and no client can undo it after the fact —
    // raising max_tokens only buys more duplicates. So repetition is opt-in via
    // `allow_parallel_calls`, which is what OpenAI's `parallel_tool_calls` means.
    let stop_after_first = tools_mandatory || !allow_parallel_calls;
    let tool_calls = gemma4_tool_calls(tool_names, stop_after_first);

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
            let after_content = json!({"type": "optional", "content": tool_calls.clone()});
            json!({
                "type": "or",
                "elements": [
                    {"type": "sequence", "elements": [content, after_content]},
                    tool_calls,
                ],
            })
        }
        // No content constraint to preserve: the native tag alone is enough, and
        // it is still needed so a forced choice is not pushed into the generic
        // JSON tool-call shape that Gemma 4 does not speak.
        None => tool_calls,
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
    allow_parallel_calls: bool,
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
        allow_parallel_calls,
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

    #[test]
    fn schema_and_tools_produce_a_union() {
        let tag = gemma4_structural_tag(&names(), Some(&schema()), false, false, false, false).unwrap();
        assert_eq!(tag["type"], "structural_tag");
        assert_eq!(tag["format"]["type"], "or");
        // branch 0: content then optional tool calls; branch 1: tool calls alone
        assert_eq!(tag["format"]["elements"][0]["elements"][1]["type"], "optional");
        assert_eq!(tag["format"]["elements"][1]["type"], "tags_with_separator");
    }

    #[test]
    fn forced_choice_drops_the_content_branch() {
        // The schema object is a legal prefix of "content then tool call", so offering it
        // lets the model write content and end the turn with a special token the grammar
        // cannot mask, leaving the demanded call unmade. Forced choice therefore admits
        // tool calls only — which is also what response_format means in OpenAI's API.
        let tag = gemma4_structural_tag(&names(), Some(&schema()), true, false, false, false).unwrap();
        assert_eq!(tag["format"]["type"], "tags_with_separator");
        assert_eq!(tag["format"]["at_least_one"], true);
    }

    #[test]
    fn without_schema_the_native_tag_is_used_alone() {
        let tag = gemma4_structural_tag(&names(), None, true, false, false, false).unwrap();
        assert_eq!(tag["format"]["type"], "tags_with_separator");
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
        let tags = &tag["format"]["tags"];
        assert_eq!(tags[0]["begin"], "<|tool_call>call:fetch_seller_details");
        assert_eq!(tags[0]["content"]["type"], "any_text");
        assert_eq!(tags[0]["end"], GEMMA4_TOOL_CALL_END);
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
    fn repetition_is_off_unless_opted_into() {
        // The production loop: the same call 31-62x until finish_reason=length. A repeatable
        // branch makes that an accepting string, so it must not be the default on any path.
        let forced = gemma4_structural_tag(&names(), None, true, false, false, false).unwrap();
        assert_eq!(forced["format"]["stop_after_first"], true);

        // the tool-only branch of the union, unforced — the turn that loops in production
        let auto =
            gemma4_structural_tag(&names(), Some(&schema()), false, false, false, false).unwrap();
        assert_eq!(auto["format"]["elements"][1]["stop_after_first"], true);
        // and the branch reachable after the content object
        assert_eq!(
            auto["format"]["elements"][0]["elements"][1]["content"]["stop_after_first"],
            true
        );

        // opting in restores batching for callers that genuinely want parallel calls
        let parallel =
            gemma4_structural_tag(&names(), Some(&schema()), false, false, true, false).unwrap();
        assert_eq!(parallel["format"]["elements"][1]["stop_after_first"], false);
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
