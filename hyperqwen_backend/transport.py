"""OpenAI HTTP/SSE transport. No inference framework imports in the adapter."""
import json
import math


def prediction_payload(request, model, streaming=False):
    if request.Images or request.Videos or request.Audios:
        raise ValueError("multimodal requests are not supported in this initial backend")
    if request.Grammar or request.Mirostat:
        raise ValueError("GBNF grammar and Mirostat are not supported by this backend")
    if request.Tokens < 0:
        raise ValueError("token limit must be nonnegative")
    for name in ('Temperature', 'TopP', 'MinP', 'Penalty', 'PresencePenalty', 'FrequencyPenalty'):
        if not math.isfinite(getattr(request, name)):
            raise ValueError(f'{name} must be finite')
    if request.Temperature < 0 or request.Penalty < 0:
        raise ValueError('temperature and repetition penalty must be nonnegative')
    if not 0 <= request.TopP <= 1 or not 0 <= request.MinP <= 1:
        raise ValueError('top_p and min_p must be between zero and one')
    if request.TopK < -1 or request.Logprobs < 0 or request.TopLogprobs < 0:
        raise ValueError('invalid top_k or logprobs value')
    chat = bool(request.Messages)
    payload = {"model": model, "stream": streaming,
               "temperature": request.Temperature,
               "presence_penalty": request.PresencePenalty,
               "frequency_penalty": request.FrequencyPenalty,
               "seed": request.Seed}
    if request.Tokens > 0:
        payload["max_tokens"] = request.Tokens
    # Proto3 cannot distinguish omitted scalar values from explicit zero.
    # Temperature=0 is meaningful; top_p=0 is invalid upstream, so omit it.
    for field, value in (("top_p", request.TopP), ("top_k", request.TopK), ("min_p", request.MinP)):
        if value != 0:
            payload[field] = value
    if request.StopPrompts:
        payload["stop"] = list(request.StopPrompts)
    if request.IgnoreEOS:
        payload["ignore_eos"] = True
    if request.Penalty:
        payload["repetition_penalty"] = request.Penalty
    if request.LogitBias:
        bias = json.loads(request.LogitBias)
        if not isinstance(bias, dict):
            raise ValueError("logit bias must be a JSON object")
        for token, value in bias.items():
            if not token.isdecimal() or isinstance(value, bool) or not isinstance(value, (int, float)) or not -100 <= value <= 100 or not math.isfinite(value):
                raise ValueError('logit bias requires token IDs and finite values between -100 and 100')
        payload["logit_bias"] = bias
    if request.Logprobs or request.TopLogprobs:
        if chat:
            payload["logprobs"] = True
            payload["top_logprobs"] = request.TopLogprobs or request.Logprobs
        else:
            payload["logprobs"] = request.TopLogprobs or request.Logprobs
    if chat:
        if request.Prompt:
            raise ValueError("provide messages or a rendered prompt, not both")
        messages = []
        for message in request.Messages:
            item = {"role": message.role, "content": message.content}
            for proto_name, wire_name in (("name", "name"), ("tool_call_id", "tool_call_id"), ("reasoning_content", "reasoning_content")):
                value = getattr(message, proto_name, "")
                if value:
                    item[wire_name] = value
            if message.tool_calls:
                item["tool_calls"] = json.loads(message.tool_calls)
            messages.append(item)
        payload["messages"] = messages
        if request.Tools:
            tools = json.loads(request.Tools)
            if not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools):
                raise ValueError("tools must be a JSON array of objects")
            payload["tools"] = tools
        if request.ToolChoice:
            payload["tool_choice"] = json.loads(request.ToolChoice)
        if "reasoning_effort" in request.Metadata:
            effort = request.Metadata["reasoning_effort"]
            if effort not in ("none", "minimal", "low", "medium", "high", "xhigh"):
                raise ValueError("unsupported reasoning_effort value")
            payload["reasoning_effort"] = effort
        if "enable_thinking" in request.Metadata:
            value = request.Metadata["enable_thinking"]
            if value not in ("true", "false"):
                raise ValueError("enable_thinking must be true or false")
            payload["chat_template_kwargs"] = {"enable_thinking": value == "true"}
    else:
        if request.Tools or request.ToolChoice:
            raise ValueError("tool requests require structured messages")
        payload["prompt"] = request.Prompt
    if streaming:
        payload["stream_options"] = {"include_usage": True}
    return ("/v1/chat/completions" if chat else "/v1/completions"), payload


def reply_from_json(document, pb, streaming=False):
    reply = pb.Reply()
    usage = document.get("usage") or {}
    reply.tokens = usage.get("completion_tokens", 0)
    reply.prompt_tokens = usage.get("prompt_tokens", 0)
    choices = document.get("choices") or []
    if not choices:
        return reply
    choice = choices[0]
    message = choice.get("delta" if streaming else "message")
    if message is not None:
        delta = reply.chat_deltas.add()
        delta.content = message.get("content") or ""
        delta.reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
        for index, call in enumerate(message.get("tool_calls") or []):
            function = call.get("function") or {}
            delta.tool_calls.add(index=call.get("index", index), id=call.get("id") or "",
                                 name=function.get("name") or "", arguments=function.get("arguments") or "")
        reply.message = delta.content.encode()
    else:
        reply.message = (choice.get("text") or "").encode()
    if choice.get("logprobs") is not None:
        reply.logprobs = json.dumps(choice["logprobs"]).encode()
    # The pinned LocalAI Reply has no finish_reason field.
    return reply
