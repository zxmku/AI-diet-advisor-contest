import http.server
import socketserver
import urllib.request
import urllib.error
import os

# 端口代理：同源提供前端 + 转发 /api 到后端，避免跨域。
# 路径全部相对自身位置，不依赖任何绝对路径。
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # backend/scripts/vault -> 项目根
FRONTEND = os.path.join(ROOT, "frontend", "index.html")
BACKEND = "http://127.0.0.1:8200"


class H(http.server.BaseHTTPRequestHandler):
    def _proxy(self, method, body):
        req = urllib.request.Request(BACKEND + self.path, data=body, method=method)
        if body:
            req.add_header("Content-Type", self.headers.get("Content-Type", "application/json"))
        try:
            resp = urllib.request.urlopen(req, timeout=30)
            data = resp.read()
            self.send_response(resp.status)
            for k, v in resp.getheaders():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health" or self.path.startswith("/api/"):
            self._proxy("GET", None)
        else:
            with open(FRONTEND, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_POST(self):
        if self.path.startswith("/api/"):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length) if length else None
            self._proxy("POST", body)
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


socketserver.TCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("127.0.0.1", 8201), H) as httpd:
    print("HealthPick demo proxy on http://127.0.0.1:8201 -> backend 8200")
    httpd.serve_forever()
