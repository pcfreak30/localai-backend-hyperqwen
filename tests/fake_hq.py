"""CPU-only HQ HTTP stand-in; never opens model weights."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import time
from pathlib import Path


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == '/v1/models':
            if self.headers.get('Authorization') != 'Bearer ' + os.environ['VLLM_API_KEY']:
                self.send_error(401)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"data":[{"id":"qwen3.8-27b"}]}')
        else:
            self.send_response(200 if self.path == '/health' else 404)
            self.end_headers()

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if self.headers.get('Authorization') != 'Bearer ' + os.environ['VLLM_API_KEY']:
            self.send_error(401)
            return
        messages = request.get('messages') or []
        continuation = bool(messages and messages[-1].get('role') == 'tool')
        if continuation:
            previous = messages[-2] if len(messages) > 1 else {}
            calls = previous.get('tool_calls') or []
            if not calls or calls[0].get('id') != messages[-1].get('tool_call_id'):
                self.send_error(400, 'tool continuation identity was lost')
                return
        time.sleep(0.1)  # Allow tests to observe overlapping requests.
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if request.get('stream') else 'application/json')
        self.end_headers()
        if continuation:
            if request.get('stream'):
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"tool acknowledged"}}]}\n\ndata: [DONE]\n\n')
            else:
                self.wfile.write(b'{"choices":[{"message":{"content":"tool acknowledged"}}]}')
        elif request.get('tools'):
            message = {'content': '', 'reasoning_content': 'fixture reasoning',
                       'tool_calls': [{'id': 'call_fixture', 'type': 'function',
                                       'function': {'name': 'lookup', 'arguments': '{}'}}]}
            if request.get('stream'):
                message['tool_calls'][0]['index'] = 0
                message['tool_calls'][0]['function']['arguments'] = '{'
                frames = [{'choices': [{'delta': message}]},
                          {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '}'}}]}}]},
                          {'choices': [], 'usage': {'prompt_tokens': 3, 'completion_tokens': 4}}]
                wire = ''.join('data: ' + json.dumps(frame) + '\n\n' for frame in frames)
                self.wfile.write((wire + 'data: [DONE]\n\n').encode())
            else:
                self.wfile.write(json.dumps({'choices': [{'message': message}]}).encode())
        elif request.get('stream'):
            slow = (request.get('messages') or [{}])[-1].get('content') == 'cancel'
            if slow:
                try:
                    for _ in range(1000):
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
                        self.wfile.flush()
                        time.sleep(0.02)
                except (BrokenPipeError, ConnectionResetError):
                    marker = os.environ.get('HQ_FAKE_DISCONNECT_MARKER')
                    if marker:
                        Path(marker).touch()
            else:
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n')
        else:
            self.wfile.write(b'{"choices":[{"message":{"content":"hello"}}]}')


ThreadingHTTPServer(('127.0.0.1', int(os.environ['PORT'])), Handler).serve_forever()
