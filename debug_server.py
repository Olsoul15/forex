import http.server
import socketserver
import os

PORT = int(os.getenv("PORT", 8000))

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"Debugger running. Exec into this container to run the application manually.")

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Debugger server listening on port", PORT)
    httpd.serve_forever() 