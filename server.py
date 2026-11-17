"""Mobile-friendly local browser interface for Haven."""
import json, os, threading, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from haven import Haven

brain=Haven(os.environ.get("HAVEN_STATE","haven_state.json"))
running=True

def cognition_loop():
    while True:
        if running: brain.tick()
        time.sleep(0.25)

threading.Thread(target=cognition_loop,daemon=True).start()

class Handler(BaseHTTPRequestHandler):
    def send(self,status,data,content_type="application/json"):
        payload=data if isinstance(data,bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",str(len(payload)))
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        self.wfile.write(payload)
    def do_GET(self):
        if self.path=="/":
            self.send(200,open("index.html","rb").read(),"text/html; charset=utf-8")
        elif self.path=="/api/state":self.send(200,brain.snapshot())
        else:self.send(404,{"error":"not found"})
    def do_POST(self):
        global running
        length=int(self.headers.get("Content-Length","0"))
        if length>10000:return self.send(413,{"error":"too large"})
        try: body=json.loads(self.rfile.read(length) or b"{}")
        except ValueError:return self.send(400,{"error":"invalid JSON"})
        if self.path=="/api/teach":
            a=str(body.get("source","")).strip().upper()
            b=str(body.get("target","")).strip().upper()
            if not a or not b or len(a)>32 or len(b)>32 or a==b:
                return self.send(400,{"error":"provide two distinct labels (1-32 chars)"})
            return self.send(200,brain.teach(a,b))
        if self.path=="/api/pause":
            running=not running
            return self.send(200,{"running":running})
        self.send(404,{"error":"not found"})

if __name__=="__main__":
    host=os.environ.get("HAVEN_HOST","127.0.0.1")
    port=int(os.environ.get("PORT","8765"))
    print(f"Haven listening on http://{host}:{port}")
    ThreadingHTTPServer((host,port),Handler).serve_forever()
