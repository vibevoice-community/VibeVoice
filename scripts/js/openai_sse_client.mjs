// Node.js script to call VibeVoice API with SSE streaming and write a WAV file.
// Requires Node 18+ (built-in fetch, TextDecoder), no extra deps.
//
// Usage:
//   node scripts/js/openai_sse_client.mjs \
//     --base http://127.0.0.1:8000 \
//     --model "F:/VibeVoice-Large" \
//     --voice Alice \
//     --text "Hello SSE" \
//     --out outputs/js_sse/out.wav
//
// For mp3/flac/opus/aac streaming, the server currently outputs PCM via SSE.
// You can convert the final WAV externally if needed.

import fs from "node:fs";
import path from "node:path";

function parseArgs() {
  const args = process.argv.slice(2);
  const out = {
    base: "http://127.0.0.1:8000",
    model: "vibevoice/VibeVoice-1.5B",
    voice: "Alice",
    text: "Hello SSE",
    out: "outputs/js_sse/out.wav",
    instructions: null,
    speed: null,
  };
  for (let i = 0; i < args.length; i += 2) {
    const k = args[i];
    const v = args[i + 1];
    if (!v) break;
    if (k === "--base") out.base = v;
    else if (k === "--model") out.model = v;
    else if (k === "--voice") out.voice = v;
    else if (k === "--text") out.text = v;
    else if (k === "--out") out.out = v;
    else if (k === "--instructions") out.instructions = v;
    else if (k === "--speed") out.speed = parseFloat(v);
  }
  return out;
}

function wavHeader(bytesLength, sampleRate = 24000) {
  const blockAlign = 2; // mono 16-bit
  const byteRate = sampleRate * blockAlign;
  const dataSize = bytesLength;
  const buffer = Buffer.alloc(44);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16); // Subchunk1Size (16 for PCM)
  buffer.writeUInt16LE(1, 20); // AudioFormat (1 = PCM)
  buffer.writeUInt16LE(1, 22); // NumChannels (1)
  buffer.writeUInt32LE(sampleRate, 24); // SampleRate
  buffer.writeUInt32LE(byteRate, 28); // ByteRate
  buffer.writeUInt16LE(blockAlign, 32); // BlockAlign
  buffer.writeUInt16LE(16, 34); // BitsPerSample
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);
  return buffer;
}

async function main() {
  const cfg = parseArgs();
  const url = new URL("/audio/speech", cfg.base).toString();
  const body = {
    model: cfg.model,
    voice: cfg.voice,
    input: cfg.text,
    stream_format: "sse",
  };
  if (cfg.instructions) body.instructions = cfg.instructions;
  if (cfg.speed != null) body.speed = cfg.speed;

  await fs.promises.mkdir(path.dirname(cfg.out), { recursive: true });
  const chunks = [];
  let sampleRate = 24000;

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify(body),
  });
  if (!resp.ok || !resp.body) {
    console.error("HTTP error:", resp.status, await resp.text());
    process.exit(2);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // Parse SSE: split by \n\n blocks
    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      // parse lines starting with 'event:' and 'data:'
      const lines = block.split(/\r?\n/);
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      try {
        const payload = JSON.parse(data);
        if (event === "start" && payload.sample_rate) {
          sampleRate = payload.sample_rate;
          console.log("SSE start: sample_rate=", sampleRate);
        } else if (event === "chunk" && payload.type === "audio_chunk" && payload.data) {
          const pcm = Buffer.from(payload.data, "base64");
          chunks.push(pcm);
          process.stdout.write(".");
        } else if (event === "error") {
          console.error("SSE error:", payload);
        } else if (event === "end") {
          console.log("\nSSE end");
        }
      } catch (e) {
        // ignore parse errors
      }
    }
  }

  // Write WAV
  const pcmBytes = Buffer.concat(chunks);
  const header = wavHeader(pcmBytes.length, sampleRate);
  await fs.promises.writeFile(cfg.out, Buffer.concat([header, pcmBytes]));
  console.log(`Saved to ${cfg.out} (${pcmBytes.length} bytes of PCM)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

