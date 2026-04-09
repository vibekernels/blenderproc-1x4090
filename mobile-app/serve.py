"""Simple HTTPS server for testing on phone (camera requires secure context).

Usage:
    python serve.py

Then open https://<your-ip>:8443 on your phone (accept the self-signed cert warning).
For localhost testing, http://localhost:8000 works fine (localhost is treated as secure).
"""

import http.server
import os
import ssl
import subprocess
import sys

PORT_HTTP = 8000
PORT_HTTPS = 8443
CERT_FILE = "cert.pem"
KEY_FILE = "key.pem"

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")


def generate_self_signed_cert():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        return
    print("Generating self-signed certificate for HTTPS...")
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", KEY_FILE, "-out", CERT_FILE,
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost",
    ], check=True)


def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


if __name__ == "__main__":
    generate_self_signed_cert()
    ip = get_local_ip()

    print(f"\n  Local (HTTP):   http://localhost:{PORT_HTTP}")
    print(f"  Network (HTTPS): https://{ip}:{PORT_HTTPS}")
    print(f"\n  On your phone, open the HTTPS URL and accept the certificate warning.\n")

    # Run HTTPS server
    handler = http.server.SimpleHTTPRequestHandler
    server = http.server.HTTPServer(("0.0.0.0", PORT_HTTPS), handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(CERT_FILE, KEY_FILE)
    server.socket = context.wrap_socket(server.socket, server_side=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
