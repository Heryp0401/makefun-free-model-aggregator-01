// The client wires and the three upstream wires.
//
// The trial's client calls this entry as an OpenAI service (POST
// /v1/chat/completions) or an Anthropic one (POST /v1/messages), with a JSON
// body or with server-sent events. This entry forwards each request to a
// provider on that provider's own wire — openai, anthropic or gemini — and
// translates the answer back into the client's wire:
//
//   - an OpenAI client to an openai-wire provider passes the body through
//     (with the routed model's id) and streams the provider's JSON or SSE
//     straight back, the way an OpenAI-compatible gateway does;
//   - an Anthropic client to an openai-wire provider rewrites the request to
//     Chat Completions (system prompt, image blocks, tool definitions, tool
//     results and tool choice included) and rewrites the provider's answer
//     into a Messages object, or into the message_start / content_block_start
//     / content_block_delta / content_block_stop / message_delta /
//     message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta
//     events;
//   - an OpenAI client to an anthropic-wire provider rewrites the request to
//     Messages and rewrites the answer back to Chat Completions, in JSON or
//     in the chat.chunk event order;
//   - either client to a gemini-wire provider rewrites the request to a
//     generateContent call (contents, systemInstruction, tools as
//     functionDeclarations, generationConfig) and rewrites the candidates
//     back, in JSON or as the client's SSE order.
//
// The providers are simulated: their answers are one text (an "ANSWER ECHO"
// that records the system prompt, user text, tools, max_tokens, stop
// sequences, image and JSON mode the upstream request carried), so the
// translations must carry every one of those fields faithfully, or the
// scorer's echo checks fail. A response that is not a well-formed answer of
// its API does not count as served.
//
// Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same
// bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a
// nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
export function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
export function finishReasonOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function blockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`: the system
 * message becomes the system prompt, image_url parts become image blocks, tool messages become
 * tool_result blocks, assistant tool_calls become tool_use blocks, and the output cap, temperature,
 * top_p, stop sequences, tools, tool_choice and streaming pass through. A response_format of
 * json_object has no Anthropic twin: it is carried as a trailing instruction in the system prompt so
 * the provider still answers JSON, and the JSON-mode flag is kept by the caller.
 */
export function openAIRequestToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      // A tool message follows its assistant turn: attach it as a tool_result to a user turn.
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
    // A user turn: text and image parts become blocks.
    if (typeof raw.content === 'string') {
      if (raw.content !== '') messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      const block = blockOf(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  if (format === 'json_object' || format === 'json_schema') systemParts.push('Answer with one JSON object only.');
  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'none') out.tool_choice = { type: choice === 'required' ? 'any' : 'auto' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content: Json[] = [];
  let text = '';
  const toolCalls: Json[] = [];
  const blocks = Array.isArray(body.content) ? body.content : [];
  for (const block of blocks) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: typeof block.id === 'string' ? block.id : 'call_unknown',
        type: 'function',
        function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
      });
    }
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text !== '' ? text : null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: typeof body.id === 'string' ? body.id : 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: numberOrZero(usage.input_tokens), completion_tokens: numberOrZero(usage.output_tokens), total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens) },
  };
}

/** One Gemini content part as text or an inline image; the image as a data URL part for OpenAI. */
function geminiPartToOpenAI(part: Json): Json | null {
  if (typeof part.text === 'string') return { type: 'text', text: part.text };
  if (isRecord(part.inlineData) && typeof part.inlineData.data === 'string') {
    const mime = typeof part.inlineData.mimeType === 'string' ? part.inlineData.mimeType : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${mime};base64,${part.inlineData.data}` } };
  }
  if (isRecord(part.fileData) && typeof part.fileData.fileUri === 'string') {
    return { type: 'image_url', image_url: { url: part.fileData.fileUri } };
  }
  return null;
}

/**
 * A Chat Completions or Messages request as a Gemini generateContent body for `model`: contents carry
 * the turns (user parts as text plus inlineData images, model parts as text plus the tool calls the
 * provider made), the system prompt becomes systemInstruction, tools become functionDeclarations, and
 * the output cap, temperature, top_p, stop sequences and JSON mode land in generationConfig. The
 * provider's simulated answer is one text, so no tool-call round trip is expected back.
 */
export function toGeminiRequest(body: Json, model: string, clientApi: 'openai' | 'anthropic'): Json {
  const contents: Json[] = [];
  let systemText = '';
  let maxTokens: number | undefined;
  let temperature: number | undefined;
  let topP: number | undefined;
  let stops: string[] = [];
  let jsonMode = false;
  const tools: Json[] = [];

  const addParts = (role: string, parts: Json[]): void => {
    if (parts.length > 0) contents.push({ role, parts });
  };
  const textParts = (text: string): Json[] => (text !== '' ? [{ text }] : []);
  const imageParts = (url: string): Json[] => {
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return [{ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } }];
    return [{ fileData: { fileUri: url } }];
  };

  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemText += (systemText !== '' ? '\n' : '') + text;
        continue;
      }
      if (raw.role === 'assistant') {
        const parts: Json[] = [...textParts(textOf(raw.content))];
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
        addParts('model', parts);
        continue;
      }
      if (raw.role === 'tool') {
        addParts('user', [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }]);
        continue;
      }
      if (typeof raw.content === 'string') {
        addParts('user', textParts(raw.content));
        continue;
      }
      if (!Array.isArray(raw.content)) continue;
      const parts: Json[] = [];
      for (const part of raw.content) {
        if (!isRecord(part)) continue;
        if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
        if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') parts.push(...imageParts(part.image_url.url));
      }
      addParts('user', parts);
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    else if (typeof body.max_completion_tokens === 'number') maxTokens = body.max_completion_tokens;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
    if (Array.isArray(body.stop)) stops = body.stop.filter((s): s is string => typeof s === 'string');
    const format = isRecord(body.response_format) ? body.response_format.type : undefined;
    jsonMode = format === 'json_object' || format === 'json_schema';
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        tools.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
      }
    }
  } else {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'assistant') {
        const parts: Json[] = [...textParts(textOf(raw.content))];
        if (Array.isArray(raw.content)) {
          for (const block of raw.content) {
            if (!isRecord(block) || block.type !== 'tool_use') continue;
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
        addParts('model', parts);
        continue;
      }
      if (typeof raw.content === 'string') {
        addParts('user', textParts(raw.content));
        continue;
      }
      if (!Array.isArray(raw.content)) continue;
      const parts: Json[] = [];
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ fileData: { fileUri: block.source.url } });
          }
        }
        if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } });
        }
      }
      addParts('user', parts);
    }
    if (typeof body.max_tokens === 'number') maxTokens = body.max_tokens;
    if (typeof body.temperature === 'number') temperature = body.temperature;
    if (typeof body.top_p === 'number') topP = body.top_p;
    if (Array.isArray(body.stop_sequences)) stops = body.stop_sequences.filter((s): s is string => typeof s === 'string');
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        tools.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
      }
    }
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (stops.length > 0) config.stopSequences = stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  void model;
  return out;
}

/** The text and usage of a Gemini answer: the candidates' parts joined, and its usageMetadata. */
function geminiTextAndUsage(body: Json): { text: string; input: number; output: number; finish: unknown } {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount), finish: first.finishReason };
}

/** A Gemini answer as a Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const { text, input, output, finish } = geminiTextAndUsage(body);
  return {
    id: 'chatcmpl-gemini',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text !== '' ? text : null }, finish_reason: finishReasonOfGemini(finish), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const { text, input, output, finish } = geminiTextAndUsage(body);
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: stopReasonOf(finishReasonOfGemini(finish)),
    stop_sequence: null,
    usage: { input_tokens: input, output_tokens: output },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Read every SSE frame of a stream as its parsed JSON payloads, in order. */
export async function readSseFrames(stream: ReadableStream<Uint8Array>): Promise<{ event: string | null; data: string }[]> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const frames: { event: string | null; data: string }[] = [];
  const flush = (text: string): void => {
    let event: string | null = null;
    for (const line of text.split('\n')) {
      const trimmed = line.trim();
      if (trimmed.startsWith('event:')) event = trimmed.slice(6).trim();
      else if (trimmed.startsWith('data:')) frames.push({ event, data: trimmed.slice(5).trim() });
    }
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        flush(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') flush(buffer);
  } finally {
    reader.releaseLock();
  }
  return frames;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/**
 * An Anthropic Messages SSE stream as the Chat Completions chunk order: a role chunk, then content
 * deltas for the text, then a finish chunk with the stop reason, then the usage chunk when the
 * provider sent one, then [DONE]. Tool_use blocks become tool_calls deltas. `model` is the model the
 * client asked for, named back to it.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const frames = await readSseFrames(upstream);
  let inputTokens = 0;
  let outputTokens = 0;
  let stopReason = 'end_turn';
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const out: string[] = [];
  out.push(chunk({ role: 'assistant', content: '' }, null));
  const toolIndexes = new Map<number, { id: string; name: string; args: string }>();
  let nextTool = 0;
  for (const frame of frames) {
    if (frame.data === '[DONE]') continue;
    let event: unknown;
    try {
      event = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    if (event.type === 'message_start' && isRecord(event.message) && isRecord(event.message.usage)) {
      inputTokens = numberOrZero(event.message.usage.input_tokens);
    }
    if (event.type === 'content_block_start' && isRecord(event.content_block)) {
      const block = event.content_block;
      if (block.type === 'tool_use') {
        const index = typeof event.index === 'number' ? event.index : nextTool;
        nextTool = Math.max(nextTool, index + 1);
        toolIndexes.set(index, { id: typeof block.id === 'string' ? block.id : `call_${index}`, name: typeof block.name === 'string' ? block.name : '', args: '' });
        out.push(chunk({ tool_calls: [{ index, id: toolIndexes.get(index)?.id ?? '', type: 'function', function: { name: toolIndexes.get(index)?.name ?? '', arguments: '' } }] }, null));
      }
    }
    if (event.type === 'content_block_delta' && isRecord(event.delta)) {
      const delta = event.delta;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        out.push(chunk({ content: delta.text }, null));
      }
      if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
        const index = typeof event.index === 'number' ? event.index : 0;
        const tool = toolIndexes.get(index);
        if (tool !== undefined) {
          tool.args += delta.partial_json;
          out.push(chunk({ tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, null));
        }
      }
    }
    if (event.type === 'message_delta' && isRecord(event.delta)) {
      if (typeof event.delta.stop_reason === 'string') stopReason = event.delta.stop_reason;
      if (isRecord(event.usage)) outputTokens = numberOrZero(event.usage.output_tokens);
    }
  }
  const finish = finishReasonOf(stopReason);
  out.push(chunk({}, finish));
  out.push(sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }));
  out.push('data: [DONE]\n\n');
  for (const event of out) yield event;
}

/**
 * A Gemini SSE stream as the Chat Completions chunk order: the candidates' text parts become content
 * deltas, the final frame's finishReason becomes the finish chunk, and its usageMetadata becomes the
 * usage chunk, then [DONE]. `model` is the model the client asked for, named back to it.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const frames = await readSseFrames(upstream);
  const chunk = (delta: Json, finish: string | null): string =>
    sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] });
  const out: string[] = [];
  out.push(chunk({ role: 'assistant', content: '' }, null));
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: string | null = null;
  for (const frame of frames) {
    if (frame.data === '[DONE]') continue;
    let event: unknown;
    try {
      event = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    const candidates = Array.isArray(event.candidates) ? event.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string' && part.text !== '') out.push(chunk({ content: part.text }, null));
    }
    if (typeof first.finishReason === 'string') finish = finishReasonOfGemini(first.finishReason);
    if (isRecord(event.usageMetadata)) {
      inputTokens = numberOrZero(event.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(event.usageMetadata.candidatesTokenCount);
    }
  }
  out.push(chunk({}, finish ?? 'stop'));
  out.push(sseEvent(null, { id: 'chatcmpl-stream', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens } }));
  out.push('data: [DONE]\n\n');
  for (const event of out) yield event;
}

/**
 * A Gemini SSE stream as the Anthropic event sequence: the candidates' text parts become one text
 * block's deltas, the final frame's finishReason becomes the message_delta stop reason with its
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const frames = await readSseFrames(upstream);
  const out: string[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: string | null = null;
  let started = false;
  out.push(
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
    }),
  );
  started = true;
  void started;
  out.push(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
  for (const frame of frames) {
    if (frame.data === '[DONE]') continue;
    let event: unknown;
    try {
      event = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(event)) continue;
    const candidates = Array.isArray(event.candidates) ? event.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(first.content) ? first.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part.text } }));
      }
    }
    if (typeof first.finishReason === 'string') finish = finishReasonOfGemini(first.finishReason);
    if (isRecord(event.usageMetadata)) {
      inputTokens = numberOrZero(event.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(event.usageMetadata.candidatesTokenCount);
    }
  }
  out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
  out.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: stopReasonOf(finish ?? 'stop'), stop_sequence: null },
      usage: { output_tokens: outputTokens },
    }),
  );
  out.push(sseEvent('message_stop', { type: 'message_stop' }));
  void inputTokens;
  for (const event of out) yield event;
}
