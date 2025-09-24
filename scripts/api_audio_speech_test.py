#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.request
import urllib.error


def _load_dotenv(path: str) -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$', s)
                if not m:
                    continue
                k, v = m.group(1), m.group(2)
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                os.environ.setdefault(k, v)
    except Exception:
        pass


def _http_post_json(url: str, payload: dict, timeout: float = 60.0):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method='POST', headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read(), dict(resp.getheaders())
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers.items())
    except Exception as e:
        return -1, str(e).encode(), {}


def main():
    _load_dotenv('.env')
    _load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

    ap = argparse.ArgumentParser(description='Direct HTTP test for VibeVoice audio.speech API')
    ap.add_argument('--base_url', default='http://127.0.0.1:8000')
    ap.add_argument('--model_path', default=os.environ.get('VIBEVOICE_MODEL', 'vibevoice/VibeVoice-1.5B'))
    ap.add_argument('--voice', default='Alice')
    ap.add_argument('--speakers', nargs='*', default=None, help='List of alias/path/dataURL for Speaker 1..N')
    ap.add_argument('--voice_path', default=None)
    ap.add_argument('--voice_data_path', default=None)
    ap.add_argument('--text', default='Hello from VibeVoice!')
    ap.add_argument('--instructions', default=None)
    ap.add_argument('--instructions_strategy', default=None, choices=['system_only','preprompt_only','system_and_preprompt'])
    ap.add_argument('--instructions_repeat', type=int, default=None)
    ap.add_argument('--response_format', default='wav', choices=['wav','pcm','mp3','opus','aac'])
    ap.add_argument('--stream_format', default=None, choices=['sse','audio',None])
    ap.add_argument('--speed', type=float, default=None)
    ap.add_argument('--cfg_scale', type=float, default=None)
    ap.add_argument('--ddpm_steps', type=int, default=None)
    ap.add_argument('--out', default='outputs/api_http/out.bin')
    args = ap.parse_args()

    payload = {
        'model': args.model_path,
        'voice': args.voice,
        'input': args.text,
        'response_format': args.response_format,
    }
    if args.stream_format:
        payload['stream_format'] = args.stream_format
    if args.speakers:
        payload['speakers'] = args.speakers
    if args.voice_path:
        payload['voice_path'] = args.voice_path
    if args.voice_data_path:
        with open(args.voice_data_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode()
        # naive MIME guess
        mime = 'audio/wav'
        ext = os.path.splitext(args.voice_data_path)[1].lower()
        if ext == '.mp3': mime = 'audio/mpeg'
        elif ext == '.aac' or ext == '.m4a': mime = 'audio/aac'
        elif ext == '.ogg' or ext == '.opus': mime = 'audio/ogg'
        payload['voice_data'] = f'data:{mime};base64,{b64}'
    if args.instructions:
        payload['instructions'] = args.instructions
    if args.instructions_strategy:
        payload['instructions_strategy'] = args.instructions_strategy
    if args.instructions_repeat is not None:
        payload['instructions_repeat'] = args.instructions_repeat
    if args.speed is not None:
        payload['speed'] = args.speed
    if args.cfg_scale is not None:
        payload['cfg_scale'] = args.cfg_scale
    if args.ddpm_steps is not None:
        payload['ddpm_steps'] = args.ddpm_steps

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    url = args.base_url.rstrip('/') + '/audio/speech'
    code, data, headers = _http_post_json(url, payload)
    if code != 200:
        print('HTTP error:', code, data[:200])
        return 2
    with open(args.out, 'wb') as f:
        f.write(data)
    print('Saved to', args.out, f'({len(data)} bytes)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

