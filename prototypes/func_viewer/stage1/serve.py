"""Serve one pack and the player page on this computer only.

usage: serve.py PACK [--port N]

PACK is a pack directory of pack_run.py. The page is then at http://127.0.0.1:N/ (N is 8765
unless given; 0 takes a free port; the address is printed). The server binds to 127.0.0.1 and
answers nothing else: only GET and HEAD, only requests whose Host header names 127.0.0.1 or
localhost, only the files of player/ beside this script (under /player/, the page itself at /)
and the files of PACK (under /pack/). A name that climbs out of either folder, a link that
leads out of it, a folder and an unknown name are answered with 404. Nothing is written.

A pack file that is asked for with ?v=... is answered as immutable (the page puts the pack's
manifest digest there), everything else as no-store. Every answer forbids loading from another
host. Stop it with Ctrl-C.
"""

import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

PLAYER = Path(__file__).resolve().parent / "player"
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".bin": "application/octet-stream",
}
CSP = (
    "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline';"
    " worker-src 'self'; frame-ancestors 'none'"
)


def resolve(root, relative):
    """The file under root that a URL path names, or None (also for a path that leaves root)."""
    if not relative or "\\" in relative or "\0" in relative:
        return None
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    root = root.resolve()
    path = root.joinpath(*parts).resolve()
    if root not in path.parents or not path.is_file():
        return None
    return path if path.suffix in TYPES else None


def make_handler(pack):
    pack = Path(pack)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "func-viewer"
        sys_version = ""

        def log_message(self, *args):  # quiet: the page asks for many files
            pass

        def answer(self, status, body=b"", content_type="text/plain; charset=utf-8", cache=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache or "no-store")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def locate(self):
            url = urlsplit(self.path)
            path = unquote(url.path)
            if path == "/":
                return PLAYER / "index.html", False
            if path.startswith("/player/"):
                return resolve(PLAYER, path[len("/player/") :]), False
            if path.startswith("/pack/"):
                return resolve(pack, path[len("/pack/") :]), "v=" in url.query
            return None, False

        def do_GET(self):
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in ("127.0.0.1", "localhost"):
                self.answer(HTTPStatus.FORBIDDEN, b"this server answers on 127.0.0.1 only\n")
                return
            path, versioned = self.locate()
            if path is None or not path.is_file():
                self.answer(HTTPStatus.NOT_FOUND, b"not found\n")
                return
            cache = "private, max-age=31536000, immutable" if versioned else None
            self.answer(HTTPStatus.OK, path.read_bytes(), TYPES[path.suffix], cache)

        do_HEAD = do_GET

    return Handler


def make_server(pack, port=8765):
    """A server for PACK on 127.0.0.1 (port 0: a free one); the caller runs serve_forever()."""
    pack = Path(pack)
    if not (pack / "manifest.json").is_file():
        raise FileNotFoundError(f"{pack} holds no manifest.json: not a pack of pack_run.py")
    ThreadingHTTPServer.daemon_threads = True
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(pack))


def main():
    args = sys.argv[1:]
    port = 8765
    if "--port" in args:
        at = args.index("--port")
        if at + 1 >= len(args) or not args[at + 1].isdigit():
            args = []
        else:
            port = int(args[at + 1])
            args = args[:at] + args[at + 2 :]
    if len(args) != 1:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    try:
        server = make_server(args[0], port)
    except (FileNotFoundError, OSError) as err:
        sys.exit(f"cannot serve: {err}")
    print(f"http://127.0.0.1:{server.server_address[1]}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
