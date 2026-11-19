import http.server
import socketserver
import json
import threading
import time
import os
import uuid
import sys
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from core.llm_connector import create_llm_connector
    LLM_CONNECTOR_AVAILABLE = True
except ImportError:
    LLM_CONNECTOR_AVAILABLE = False
    print("Warning: LLM connector not available, using mock mode")

class HTTPClient:
    @staticmethod
    def post(url, data=None, headers=None, timeout=30):
        import http.client
        parsed = urlparse(url)
        conn = http.client.HTTPConnection(parsed.netloc, timeout=timeout)
        try:
            headers = headers or {}
            conn.request('POST', parsed.path, body=data, headers=headers)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

    @staticmethod
    def get(url, timeout=10):
        import http.client
        parsed = urlparse(url)
        conn = http.client.HTTPConnection(parsed.netloc, timeout=timeout)
        try:
            conn.request('GET', parsed.path)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()

class ASRService:
    _connector = None

    @classmethod
    def get_connector(cls):
        if cls._connector is None and LLM_CONNECTOR_AVAILABLE:
            backend = os.getenv("ASR_BACKEND", "mock")
            cls._connector = create_llm_connector(backend)
        return cls._connector

    @staticmethod
    def recognize(audio_data, sample_rate=16000):
        time.sleep(0.2)
        return {
            "text": "你好，我是AI语音助手，很高兴为你服务。",
            "segments": [],
            "inference_time": 0.234,
            "language": "zh-CN"
        }

class LLMService:
    _connector = None

    @classmethod
    def get_connector(cls):
        if cls._connector is None:
            backend = os.getenv("LLM_BACKEND", "mock")
            print(f"Initializing LLM connector with backend: {backend}")
            cls._connector = create_llm_connector(backend)
        return cls._connector

    @staticmethod
    def chat(messages, max_tokens=512, temperature=0.7):
        connector = LLMService.get_connector()

        if connector:
            try:
                result = connector.chat(messages, max_tokens=max_tokens, temperature=temperature)
                return result
            except Exception as e:
                print(f"LLM connector error: {e}, falling back to mock")

        import random
        responses = [
            "很高兴和你对话！有什么我可以帮助你的吗？",
            "你好！今天感觉怎么样？有什么想聊的吗？",
            "我在这里听你说话。请问有什么需要帮助的？",
        ]
        return {
            "response": random.choice(responses),
            "inference_time": 0.512,
            "usage": {"prompt_tokens": 50, "completion_tokens": 30}
        }

class APIGatewayHandler(http.server.BaseHTTPRequestHandler):
    def _send_response(self, status_code, data, content_type='application/json'):
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        if isinstance(data, bytes):
            self.wfile.write(data)
        else:
            self.wfile.write(json.dumps(data).encode('utf-8'))

    def _send_file(self, file_path, content_type):
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except FileNotFoundError:
            self.send_error(404, "File not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/health':
            self._send_response(200, {"status": "healthy", "service": "API Gateway"})
        elif path == '/api/services':
            self._send_response(200, {
                "llm": {"url": "http://localhost:8001", "status": "healthy"},
                "asr": {"url": "http://localhost:8002", "status": "healthy"},
                "tts": {"url": "http://localhost:8003", "status": "healthy"},
                "dialogue": {"url": "http://localhost:8004", "status": "healthy"},
            })
        elif path == '/' or path == '/index.html':
            self._send_file('static/index.html', 'text/html')
        else:
            self._send_response(404, {"error": "Not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/api/dialogue':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                body = json.loads(post_data)
                text = body.get('text', '')
                session_id = body.get('session_id', str(uuid.uuid4()))
            except:
                text = post_data.decode('utf-8', errors='ignore')
                session_id = str(uuid.uuid4())

            llm_result = LLMService.chat([
                {"role": "user", "content": text}
            ])

            self._send_response(200, {
                "session_id": session_id,
                "text": llm_result["response"],
                "emotion": "neutral",
                "emotion_confidence": 0.7,
                "timings": {
                    "asr": 0,
                    "llm": llm_result["inference_time"],
                    "tts": 0,
                    "total": llm_result["inference_time"],
                },
                "cached": False,
            })

        elif path == '/api/dialogue/audio':
            content_length = int(self.headers.get('Content-Length', 0))
            audio_data = self.rfile.read(content_length)

            session_id = "test-session-123"

            try:
                asr_result = ASRService.recognize(audio_data)
                recognized_text = asr_result["text"]
                asr_time = asr_result["inference_time"]
            except Exception as e:
                recognized_text = "抱歉，语音识别失败。"
                asr_time = 0

            llm_result = LLMService.chat([
                {"role": "user", "content": recognized_text}
            ])

            self._send_response(200, {
                "session_id": session_id,
                "text": llm_result["response"],
                "recognized_text": recognized_text,
                "emotion": "neutral",
                "emotion_confidence": 0.75,
                "timings": {
                    "asr": asr_time,
                    "llm": llm_result["inference_time"],
                    "tts": 0,
                    "total": asr_time + llm_result["inference_time"],
                },
                "cached": False,
            })

        else:
            self._send_response(404, {"error": "Not found"})

class ASRServerHandler(http.server.BaseHTTPRequestHandler):
    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self._send_response(200, {"status": "healthy", "service": "ASR Service"})
        elif self.path == '/recognize':
            self._send_response(200, {
                "text": "",
                "segments": [],
                "inference_time": 0,
            })
        else:
            self._send_response(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == '/recognize' or self.path == '/asr':
            content_length = int(self.headers.get('Content-Length', 0))
            audio_data = self.rfile.read(content_length)

            result = ASRService.recognize(audio_data)
            self._send_response(200, result)
        else:
            self._send_response(404, {"error": "Not found"})

class LLMHandler(http.server.BaseHTTPRequestHandler):
    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self._send_response(200, {"status": "healthy", "service": "LLM Service"})
        elif self.path == '/stats':
            self._send_response(200, {
                "engine_type": "llm",
                "total_inferences": 0,
                "model": "mock-llm"
            })
        else:
            self._send_response(404, {"error": "Not found"})

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        if self.path == '/chat':
            try:
                body = json.loads(post_data)
                messages = body.get('messages', [])
                max_tokens = body.get('max_tokens', 512)
                temperature = body.get('temperature', 0.7)

                result = LLMService.chat(messages, max_tokens, temperature)
                self._send_response(200, result)
            except Exception as e:
                self._send_response(500, {"error": str(e)})

        elif self.path == '/generate':
            try:
                body = json.loads(post_data)
                prompts = body.get('prompts', [])
                result = LLMService.chat([{"role": "user", "content": prompts[0] if prompts else ""}])
                self._send_response(200, {
                    "responses": [result["response"]],
                    "inference_time": result["inference_time"]
                })
            except Exception as e:
                self._send_response(500, {"error": str(e)})

        else:
            self._send_response(404, {"error": "Not found"})

class TTSHandler(http.server.BaseHTTPRequestHandler):
    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self._send_response(200, {"status": "healthy", "service": "TTS Service"})
        elif self.path == '/voices':
            self._send_response(200, [
                {"id": "default", "name": "Default Voice"},
                {"id": "female_1", "name": "Female Voice"},
                {"id": "male_1", "name": "Male Voice"},
            ])
        else:
            self._send_response(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == '/synthesize':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)

            try:
                body = json.loads(post_data)
                text = body.get('text', '')
                voice_id = body.get('voice_id', 'default')

                time.sleep(0.1)

                self._send_response(200, {
                    "inference_time": 0.456,
                    "cached": False,
                    "audio_url": None,
                })
            except Exception as e:
                self._send_response(500, {"error": str(e)})
        else:
            self._send_response(404, {"error": "Not found"})

class DialogueHandler(http.server.BaseHTTPRequestHandler):
    def _send_response(self, status_code, data):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/health':
            self._send_response(200, {"status": "healthy", "service": "Dialogue Service"})
        elif path.startswith('/history/'):
            session_id = path.split('/')[2]
            self._send_response(200, {
                "session_id": session_id,
                "history": [],
                "total_turns": 0,
            })
        elif path == '/sessions':
            self._send_response(200, {"sessions": []})
        else:
            self._send_response(404, {"error": "Not found"})

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        if self.path == '/turn':
            self._send_response(200, {"status": "success"})
        elif self.path == '/emotion':
            self._send_response(200, {
                "emotion": "neutral",
                "confidence": 0.7,
                "all_emotions": {},
            })
        else:
            self._send_response(404, {"error": "Not found"})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith('/history/'):
            session_id = path.split('/')[2]
            self._send_response(200, {"status": "success", "message": f"History cleared for {session_id}"})
        else:
            self._send_response(404, {"error": "Not found"})

def start_server(port, handler_class, server_name=""):
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("0.0.0.0", port), handler_class) as httpd:
        print(f"{server_name} running on port {port}")
        httpd.serve_forever()

def main():
    print("Starting AI Person services...")

    threads = []

    t1 = threading.Thread(target=start_server, args=(8000, APIGatewayHandler, "API Gateway"))
    t1.daemon = True
    threads.append(t1)

    t2 = threading.Thread(target=start_server, args=(8001, LLMHandler, "LLM Service"))
    t2.daemon = True
    threads.append(t2)

    t3 = threading.Thread(target=start_server, args=(8002, ASRServerHandler, "ASR Service"))
    t3.daemon = True
    threads.append(t3)

    t4 = threading.Thread(target=start_server, args=(8003, TTSHandler, "TTS Service"))
    t4.daemon = True
    threads.append(t4)

    t5 = threading.Thread(target=start_server, args=(8004, DialogueHandler, "Dialogue Service"))
    t5.daemon = True
    threads.append(t5)

    for t in threads:
        t.start()

    print("\nAll services started:")
    print("  API Gateway: http://localhost:8000")
    print("  LLM Service: http://localhost:8001")
    print("  ASR Service: http://localhost:8002")
    print("  TTS Service: http://localhost:8003")
    print("  Dialogue Service: http://localhost:8004")
    print("\nPress Ctrl+C to stop all services")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")

if __name__ == "__main__":
    main()
