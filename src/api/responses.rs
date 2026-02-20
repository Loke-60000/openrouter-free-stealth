use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::warn;

const UPSTREAM: &str = "https://openrouter.ai/api/v1";

static SEQ: AtomicU64 = AtomicU64::new(1);

fn next_id(prefix: &str) -> String {
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("{prefix}_{ts:x}{n:04x}")
}

fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

pub struct TranslatedRequest {
    pub cc_body: Value,
    pub resp_id: String,
    pub model: String,
    pub tools_echo: Value,
    pub instructions: Value,
    pub temperature: Value,
    pub top_p: Value,
    pub tool_choice: Value,
    pub parallel_tool_calls: Value,
    pub is_stream: bool,
}

pub fn translate_request(body: &Value) -> Result<TranslatedRequest, String> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or("missing `model`")?
        .to_owned();

    let is_stream = body.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut messages: Vec<Value> = Vec::new();

    if let Some(instr) = body.get("instructions").and_then(|v| v.as_str()) {
        messages.push(json!({"role": "developer", "content": instr}));
    }

    match body.get("input") {
        Some(Value::String(s)) => {
            messages.push(json!({"role": "user", "content": s}));
        }
        Some(Value::Array(items)) => {
            for item in items {
                translate_input_item(item, &mut messages);
            }
        }
        _ => {}
    }

    let mut cc_tools: Vec<Value> = Vec::new();
    if let Some(Value::Array(tools)) = body.get("tools") {
        for tool in tools {
            let tool_type = tool.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if tool_type == "function" {
                cc_tools.push(json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").unwrap_or(&Value::Null),
                        "description": tool.get("description").unwrap_or(&Value::Null),
                        "parameters": tool.get("parameters").unwrap_or(&json!({})),
                        "strict": tool.get("strict").unwrap_or(&Value::Null),
                    }
                }));
            }
        }
    }

    let mut cc = json!({
        "model": model,
        "messages": messages,
    });

    if !cc_tools.is_empty() {
        cc["tools"] = Value::Array(cc_tools);
    }

    if let Some(v) = body.get("temperature") {
        cc["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        cc["top_p"] = v.clone();
    }
    if let Some(v) = body.get("max_output_tokens") {
        cc["max_tokens"] = v.clone();
    }
    if let Some(v) = body.get("tool_choice") {
        cc["tool_choice"] = translate_tool_choice(v);
    }
    if let Some(v) = body.get("parallel_tool_calls") {
        cc["parallel_tool_calls"] = v.clone();
    }
    if let Some(v) = body.get("text") {
        if let Some(fmt) = v.get("format") {
            let fmt_type = fmt.get("type").and_then(|t| t.as_str()).unwrap_or("text");
            match fmt_type {
                "json_object" => {
                    cc["response_format"] = json!({"type": "json_object"});
                }
                "json_schema" => {
                    cc["response_format"] = json!({
                        "type": "json_schema",
                        "json_schema": {
                            "name": fmt.get("name").unwrap_or(&json!("response")),
                            "schema": fmt.get("schema").unwrap_or(&json!({})),
                            "strict": fmt.get("strict").unwrap_or(&json!(true)),
                        }
                    });
                }
                _ => {}
            }
        }
    }

    if is_stream {
        cc["stream"] = json!(true);
    }

    Ok(TranslatedRequest {
        cc_body: cc,
        resp_id: next_id("resp"),
        model,
        tools_echo: body.get("tools").cloned().unwrap_or(json!([])),
        instructions: body.get("instructions").cloned().unwrap_or(Value::Null),
        temperature: body.get("temperature").cloned().unwrap_or(json!(1)),
        top_p: body.get("top_p").cloned().unwrap_or(json!(1)),
        tool_choice: body.get("tool_choice").cloned().unwrap_or(json!("auto")),
        parallel_tool_calls: body
            .get("parallel_tool_calls")
            .cloned()
            .unwrap_or(json!(true)),
        is_stream,
    })
}

fn translate_input_item(item: &Value, messages: &mut Vec<Value>) {
    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match item_type {
        "message" => {
            let role = item
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("user");
            let cc_role = match role {
                "developer" => "system",
                other => other,
            };

            if let Some(Value::Array(content_parts)) = item.get("content") {
                let mut cc_content: Vec<Value> = Vec::new();
                for part in content_parts {
                    let ptype = part.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match ptype {
                        "input_text" => {
                            cc_content.push(json!({
                                "type": "text",
                                "text": part.get("text").unwrap_or(&Value::Null)
                            }));
                        }
                        "input_image" => {
                            if let Some(url) = part.get("image_url").and_then(|v| v.as_str()) {
                                cc_content.push(json!({
                                    "type": "image_url",
                                    "image_url": {"url": url}
                                }));
                            }
                        }
                        _ => {
                            if let Some(text) = part.get("text") {
                                cc_content.push(json!({
                                    "type": "text",
                                    "text": text
                                }));
                            }
                        }
                    }
                }
                if cc_content.len() == 1
                    && cc_content[0].get("type").and_then(|v| v.as_str()) == Some("text")
                {
                    messages.push(json!({
                        "role": cc_role,
                        "content": cc_content[0].get("text").unwrap_or(&Value::Null)
                    }));
                    return;
                }
                messages.push(json!({"role": cc_role, "content": cc_content}));
            } else if let Some(Value::String(text)) = item.get("content") {
                messages.push(json!({"role": cc_role, "content": text}));
            }
        }
        "function_call_output" => {
            messages.push(json!({
                "role": "tool",
                "tool_call_id": item.get("call_id").unwrap_or(&Value::Null),
                "content": item.get("output").unwrap_or(&json!(""))
            }));
        }
        "" => {
            if let Some(role) = item.get("role").and_then(|v| v.as_str()) {
                let cc_role = if role == "developer" { "system" } else { role };
                let content = item.get("content").unwrap_or(&Value::Null);
                messages.push(json!({"role": cc_role, "content": content}));
            }
        }
        _ => {}
    }
}

fn translate_tool_choice(v: &Value) -> Value {
    match v {
        Value::String(s) => match s.as_str() {
            "none" | "auto" | "required" => json!(s),
            _ => json!(s),
        },
        Value::Object(obj) => {
            let tc_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if tc_type == "function" {
                json!({
                    "type": "function",
                    "function": {"name": obj.get("name").unwrap_or(&Value::Null)}
                })
            } else {
                json!(v)
            }
        }
        _ => json!("auto"),
    }
}

pub fn translate_response(cc_resp: &Value, req: &TranslatedRequest) -> Value {
    let created_at = now_epoch();
    let cc_model = cc_resp
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&req.model);

    let mut output: Vec<Value> = Vec::new();

    if let Some(Value::Array(choices)) = cc_resp.get("choices") {
        for choice in choices {
            let msg = match choice.get("message") {
                Some(m) => m,
                None => continue,
            };

            if let Some(Value::Array(tool_calls)) = msg.get("tool_calls") {
                for tc in tool_calls {
                    let empty_obj = json!({});
                    let func = tc.get("function").unwrap_or(&empty_obj);
                    let empty_str = json!("");
                    let tc_id = next_id("fc");
                    output.push(json!({
                        "id": tc_id,
                        "type": "function_call",
                        "status": "completed",
                        "call_id": tc.get("id").unwrap_or(&Value::Null),
                        "name": func.get("name").unwrap_or(&Value::Null),
                        "arguments": func.get("arguments").unwrap_or(&empty_str)
                    }));
                }
            }

            if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                if !content.is_empty() {
                    let msg_id = next_id("msg");
                    output.push(json!({
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": content,
                            "annotations": []
                        }]
                    }));
                }
            }
        }
    }

    let usage = if let Some(u) = cc_resp.get("usage") {
        json!({
            "input_tokens": u.get("prompt_tokens").unwrap_or(&json!(0)),
            "output_tokens": u.get("completion_tokens").unwrap_or(&json!(0)),
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": u.get("total_tokens").unwrap_or(&json!(0))
        })
    } else {
        Value::Null
    };

    let finish_reason = cc_resp
        .pointer("/choices/0/finish_reason")
        .and_then(|v| v.as_str())
        .unwrap_or("stop");

    let status = if finish_reason == "length" {
        "incomplete"
    } else {
        "completed"
    };

    let incomplete_details = if finish_reason == "length" {
        json!({"reason": "max_output_tokens"})
    } else {
        Value::Null
    };

    json!({
        "id": req.resp_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "completed_at": created_at,
        "error": null,
        "incomplete_details": incomplete_details,
        "instructions": req.instructions,
        "model": cc_model,
        "output": output,
        "parallel_tool_calls": req.parallel_tool_calls,
        "previous_response_id": null,
        "temperature": req.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": req.tool_choice,
        "tools": req.tools_echo,
        "top_p": req.top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": {}
    })
}

pub async fn stream_response(cc_resp: reqwest::Response, req: TranslatedRequest) -> Response {
    let resp_id = req.resp_id.clone();
    let msg_id = next_id("msg");
    let model = req.model.clone();

    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);

    tokio::spawn(async move {
        let mut seq: u64 = 0;
        let mut full_text = String::new();
        let mut tool_calls: std::collections::BTreeMap<u64, ToolCallAcc> =
            std::collections::BTreeMap::new();
        #[allow(unused_assignments)]
        let mut text_content_started = false;
        let mut finish_reason = String::from("stop");
        let mut input_tokens: u64 = 0;
        let mut output_tokens: u64 = 0;
        let mut total_tokens: u64 = 0;

        macro_rules! send {
            ($event:expr, $data:expr) => {
                let _ = tx
                    .send(format!("event: {}\ndata: {}\n\n", $event, $data))
                    .await;
            };
        }

        {
            let evt = response_envelope(
                "response.created",
                &resp_id,
                &model,
                &req,
                "in_progress",
                Value::Null,
                Value::Null,
                &mut seq,
            );
            send!("response.created", evt);
        }

        {
            let evt = response_envelope(
                "response.in_progress",
                &resp_id,
                &model,
                &req,
                "in_progress",
                Value::Null,
                Value::Null,
                &mut seq,
            );
            send!("response.in_progress", evt);
        }

        {
            seq += 1;
            let evt = json!({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": &msg_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                },
                "sequence_number": seq
            });
            send!("response.output_item.added", evt);
        }

        {
            seq += 1;
            let evt = json!({
                "type": "response.content_part.added",
                "item_id": &msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": []
                },
                "sequence_number": seq
            });
            send!("response.content_part.added", evt);
            text_content_started = true;
        }

        let mut buffer = String::new();
        let mut byte_stream = cc_resp.bytes_stream();
        use tokio_stream::StreamExt;

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    warn!("Stream read error: {e}");
                    break;
                }
            };
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find("\n\n") {
                let block = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                for line in block.lines() {
                    let line = line.trim();
                    if !line.starts_with("data: ") {
                        continue;
                    }
                    let data = &line[6..];
                    if data == "[DONE]" {
                        continue;
                    }

                    let parsed: Value = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    if let Some(u) = parsed.get("usage") {
                        input_tokens = u
                            .get("prompt_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(input_tokens);
                        output_tokens = u
                            .get("completion_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(output_tokens);
                        total_tokens = u
                            .get("total_tokens")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(total_tokens);
                    }

                    let choices = match parsed.get("choices").and_then(|v| v.as_array()) {
                        Some(c) => c,
                        None => continue,
                    };

                    for choice in choices {
                        if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                            finish_reason = fr.to_string();
                        }

                        let delta = match choice.get("delta") {
                            Some(d) => d,
                            None => continue,
                        };

                        if let Some(content) = delta.get("content").and_then(|v| v.as_str()) {
                            if !content.is_empty() {
                                full_text.push_str(content);
                                seq += 1;
                                let evt = json!({
                                    "type": "response.output_text.delta",
                                    "item_id": &msg_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "delta": content,
                                    "sequence_number": seq
                                });
                                send!("response.output_text.delta", evt);
                            }
                        }

                        if let Some(Value::Array(tcs)) = delta.get("tool_calls") {
                            for tc in tcs {
                                let idx =
                                    tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0);

                                let acc =
                                    tool_calls.entry(idx).or_insert_with(|| ToolCallAcc {
                                        id: tc
                                            .get("id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        item_id: next_id("fc"),
                                        name: String::new(),
                                        arguments: String::new(),
                                        announced: false,
                                    });

                                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                                    if !id.is_empty() {
                                        acc.id = id.to_string();
                                    }
                                }

                                if let Some(f) = tc.get("function") {
                                    if let Some(name) = f.get("name").and_then(|v| v.as_str()) {
                                        acc.name.push_str(name);
                                    }
                                    if let Some(args) =
                                        f.get("arguments").and_then(|v| v.as_str())
                                    {
                                        if !acc.announced && !acc.name.is_empty() {
                                            let output_idx = idx + 1;
                                            seq += 1;
                                            let evt = json!({
                                                "type": "response.output_item.added",
                                                "output_index": output_idx,
                                                "item": {
                                                    "id": &acc.item_id,
                                                    "type": "function_call",
                                                    "status": "in_progress",
                                                    "call_id": &acc.id,
                                                    "name": &acc.name,
                                                    "arguments": ""
                                                },
                                                "sequence_number": seq
                                            });
                                            send!("response.output_item.added", evt);
                                            acc.announced = true;
                                        }

                                        acc.arguments.push_str(args);
                                        seq += 1;
                                        let output_idx = idx + 1;
                                        let evt = json!({
                                            "type": "response.function_call_arguments.delta",
                                            "item_id": &acc.item_id,
                                            "output_index": output_idx,
                                            "delta": args,
                                            "sequence_number": seq
                                        });
                                        send!("response.function_call_arguments.delta", evt);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if !full_text.is_empty() && text_content_started {
            seq += 1;
            let evt = json!({
                "type": "response.output_text.done",
                "item_id": &msg_id,
                "output_index": 0,
                "content_index": 0,
                "text": &full_text,
                "sequence_number": seq
            });
            send!("response.output_text.done", evt);

            seq += 1;
            let evt = json!({
                "type": "response.content_part.done",
                "item_id": &msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": &full_text,
                    "annotations": []
                },
                "sequence_number": seq
            });
            send!("response.content_part.done", evt);
        }

        if full_text.is_empty() && text_content_started && !tool_calls.is_empty() {
            seq += 1;
            let evt = json!({
                "type": "response.content_part.done",
                "item_id": &msg_id,
                "output_index": 0,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": []
                },
                "sequence_number": seq
            });
            send!("response.content_part.done", evt);
        }

        let mut final_output: Vec<Value> = Vec::new();

        let msg_status = if finish_reason == "length" {
            "incomplete"
        } else {
            "completed"
        };

        if !full_text.is_empty() || tool_calls.is_empty() {
            seq += 1;
            let msg_item = json!({
                "id": &msg_id,
                "type": "message",
                "role": "assistant",
                "status": msg_status,
                "content": [{
                    "type": "output_text",
                    "text": &full_text,
                    "annotations": []
                }]
            });
            let evt = json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": &msg_item,
                "sequence_number": seq
            });
            send!("response.output_item.done", evt);
            final_output.push(msg_item);
        }

        for (idx, acc) in &tool_calls {
            let output_idx = idx + 1;

            if !acc.announced {
                seq += 1;
                let evt = json!({
                    "type": "response.output_item.added",
                    "output_index": output_idx,
                    "item": {
                        "id": &acc.item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": &acc.id,
                        "name": &acc.name,
                        "arguments": ""
                    },
                    "sequence_number": seq
                });
                send!("response.output_item.added", evt);
            }

            seq += 1;
            let evt = json!({
                "type": "response.function_call_arguments.done",
                "item_id": &acc.item_id,
                "output_index": output_idx,
                "name": &acc.name,
                "arguments": &acc.arguments,
                "sequence_number": seq
            });
            send!("response.function_call_arguments.done", evt);

            seq += 1;
            let fc_item = json!({
                "id": &acc.item_id,
                "type": "function_call",
                "status": "completed",
                "call_id": &acc.id,
                "name": &acc.name,
                "arguments": &acc.arguments
            });
            let evt = json!({
                "type": "response.output_item.done",
                "output_index": output_idx,
                "item": &fc_item,
                "sequence_number": seq
            });
            send!("response.output_item.done", evt);
            final_output.push(fc_item);
        }

        let resp_status = if finish_reason == "length" {
            "incomplete"
        } else {
            "completed"
        };
        let incomplete_details = if finish_reason == "length" {
            json!({"reason": "max_output_tokens"})
        } else {
            Value::Null
        };

        let usage = json!({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": total_tokens
        });

        let completed_at = now_epoch();
        seq += 1;
        let final_event_type = if resp_status == "incomplete" {
            "response.incomplete"
        } else {
            "response.completed"
        };
        let final_response = json!({
            "id": &resp_id,
            "object": "response",
            "created_at": completed_at,
            "status": resp_status,
            "completed_at": completed_at,
            "error": null,
            "incomplete_details": incomplete_details,
            "instructions": req.instructions,
            "model": &model,
            "output": final_output,
            "parallel_tool_calls": req.parallel_tool_calls,
            "previous_response_id": null,
            "temperature": req.temperature,
            "text": {"format": {"type": "text"}},
            "tool_choice": req.tool_choice,
            "tools": req.tools_echo,
            "top_p": req.top_p,
            "truncation": "disabled",
            "usage": usage,
            "metadata": {}
        });

        let evt = json!({
            "type": final_event_type,
            "response": final_response,
            "sequence_number": seq
        });
        send!(final_event_type, evt);
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(
        tokio_stream::StreamExt::map(stream, |s| Ok::<_, std::convert::Infallible>(s)),
    );

    Response::builder()
        .status(200)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .header("connection", "keep-alive")
        .body(body)
        .unwrap()
}

struct ToolCallAcc {
    id: String,
    item_id: String,
    name: String,
    arguments: String,
    announced: bool,
}

fn response_envelope(
    event_type: &str,
    resp_id: &str,
    model: &str,
    req: &TranslatedRequest,
    status: &str,
    usage: Value,
    incomplete_details: Value,
    seq: &mut u64,
) -> String {
    *seq += 1;
    let response = json!({
        "id": resp_id,
        "object": "response",
        "created_at": now_epoch(),
        "status": status,
        "completed_at": null,
        "error": null,
        "incomplete_details": incomplete_details,
        "instructions": req.instructions,
        "max_output_tokens": null,
        "model": model,
        "output": [],
        "parallel_tool_calls": req.parallel_tool_calls,
        "previous_response_id": null,
        "temperature": req.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": req.tool_choice,
        "tools": req.tools_echo,
        "top_p": req.top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": {}
    });
    let evt = json!({
        "type": event_type,
        "response": response,
        "sequence_number": *seq
    });
    evt.to_string()
}

pub async fn handle_responses(
    client: &Client,
    api_key: &str,
    model_id: &str,
    body: Value,
) -> Response {
    let mut body = body;
    body["model"] = json!(model_id);

    let req = match translate_request(&body) {
        Ok(r) => r,
        Err(msg) => {
            return error_response(StatusCode::BAD_REQUEST, &msg, "invalid_request_error");
        }
    };

    let is_stream = req.is_stream;

    let upstream_resp = match client
        .post(format!("{UPSTREAM}/chat/completions"))
        .bearer_auth(api_key)
        .json(&req.cc_body)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return error_response(
                StatusCode::BAD_GATEWAY,
                &format!("upstream error: {e}"),
                "server_error",
            );
        }
    };

    if !upstream_resp.status().is_success() {
        let status = upstream_resp.status();
        let body_text = upstream_resp.text().await.unwrap_or_default();
        warn!(
            "Upstream error {status}: {}",
            &body_text[..body_text.len().min(200)]
        );
        return error_response(
            StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY),
            &format!("Upstream returned {status}"),
            "server_error",
        );
    }

    if is_stream {
        stream_response(upstream_resp, req).await
    } else {
        match upstream_resp.json::<Value>().await {
            Ok(cc_resp) => {
                let resp = translate_response(&cc_resp, &req);
                Response::builder()
                    .status(200)
                    .header("content-type", "application/json")
                    .body(Body::from(resp.to_string()))
                    .unwrap()
            }
            Err(e) => error_response(
                StatusCode::BAD_GATEWAY,
                &format!("failed to parse upstream response: {e}"),
                "server_error",
            ),
        }
    }
}

fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    let body = json!({
        "error": {
            "message": message,
            "type": error_type,
            "param": null,
            "code": null
        }
    });
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}
