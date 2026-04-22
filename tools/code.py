import atexit
import socket
import subprocess
import time

_PORT = 9731
_CPUS = "1"
_MEMORY = "512m"
_container_id: str | None = None


def start():
    global _container_id
    r = subprocess.run(
        ["docker", "run", "-d", "--rm",
         f"--cpus={_CPUS}", f"--memory={_MEMORY}",
         "-p", f"127.0.0.1:{_PORT}:{_PORT}",
         "strawberry-sandbox"],
        capture_output=True, text=True, check=True,
    )
    _container_id = r.stdout.strip()
    for _ in range(20):
        try:
            socket.create_connection(("127.0.0.1", _PORT), timeout=1).close()
            return
        except OSError:
            time.sleep(0.3)
    raise RuntimeError("Sandbox did not become ready")


def stop():
    if _container_id:
        subprocess.run(["docker", "kill", _container_id], capture_output=True)


atexit.register(stop)


def install_package(package: str) -> str:
    """Install a Python package into the sandbox so it can be used in run_python.
    Installs persist for the current session only.

    Args:
        package: Package name to install, e.g. "numpy" or "requests==2.31.0".
    """
    if not _container_id:
        return "Sandbox is not running."
    r = subprocess.run(
        ["docker", "exec", _container_id, "pip", "install", "--quiet", package],
        capture_output=True, text=True,
    )
    return f"Installed {package}." if r.returncode == 0 else r.stderr.strip()


def run_python(code: str) -> str:
    """Execute Python code in an isolated sandbox. Variables and imports persist between calls.
    Always print() values you want returned — bare expressions are not captured.
    Sandbox resources: {cpus} CPU core(s), {memory} RAM. Keep computations lean.

    Args:
        code: Python code to execute.
    """
    try:
        with socket.create_connection(("127.0.0.1", _PORT), timeout=30) as s:
            s.sendall(code.encode())
            s.shutdown(socket.SHUT_WR) # SHUT_WR: further transmissions will be disallowed
            out = b""
            while chunk := s.recv(4096):
                out += chunk
        return out.decode() or "(no output)"
    except OSError as e:
        return f"SandboxError: {e}"


run_python.__doc__ = (run_python.__doc__ or "").format(cpus=_CPUS, memory=_MEMORY)
