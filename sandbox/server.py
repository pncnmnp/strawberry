import io
import socket
from contextlib import redirect_stderr, redirect_stdout

PORT = 9731
# shared namespace
# So that variables and imports persist across run_python calls
ns = {}

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # AF_INET: IPv4, SOCK_STREAM: TCP
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(5) # backlog of 5 connections

while True:
    conn, _ = server.accept()
    data = b""
    while chunk := conn.recv(4096):
        data += chunk

    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            exec(compile(data.decode(), "<sandbox>", "exec"), ns)  # noqa: S102
        output = buf.getvalue()
    except Exception as e:
        output = f"{type(e).__name__}: {e}"

    conn.sendall((output or "(no output)").encode())
    conn.close()
