#!/usr/bin/env python3
"""HTTP server to host `src/` and expose a contract generation API."""

from __future__ import annotations

import argparse
import json
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import main as flow_gen
_DOTENV_LOADED = False


def _load_dotenv(path: Path | None = None) -> None:
    """Load .env file into environment if not already loaded."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    env_path = path or Path(__file__).parent / ".env"
    if not env_path.exists():
        _DOTENV_LOADED = True
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    finally:
        _DOTENV_LOADED = True


def run(directory: Path, host: str, port: int) -> None:
    """Start threaded HTTP server rooted at ``directory`` with /generate API."""
    _load_dotenv()
    directory = directory.resolve()

    class SrcHandler(SimpleHTTPRequestHandler):
        # Serve static files from src/ by default.
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def do_GET(self):  # noqa: N802 - http.server naming
            parsed = urlparse(self.path)
            if parsed.path == "/generate":
                self._handle_generate(parsed)
                return
            if parsed.path == "/generate/stream":
                self._handle_generate_stream(parsed)
                return
            return super().do_GET()

        def do_POST(self):  # noqa: N802 - http.server naming
            parsed = urlparse(self.path)
            if parsed.path == "/generate":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode("utf-8") if length else ""
                # Accept urlencoded-like payloads "address=...&chain=..." or JSON.
                params = parse_qs(body) if "=" in body else {}
                try:
                    json_body = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    json_body = {}
                query = parse_qs(parsed.query)
                merged = {**query, **params}
                if isinstance(json_body, dict):
                    for k, v in json_body.items():
                        merged[k] = [v] if not isinstance(v, list) else v
                self._handle_generate(parsed, merged)
                return
            return super().do_POST()

        def _handle_generate(self, parsed, extra_params=None):
            params = parse_qs(parsed.query)
            if extra_params:
                for k, v in extra_params.items():
                    params.setdefault(k, []).extend(v)

            address_list = params.get("address") or params.get("addr")
            chain_list = params.get("chain") or params.get("network")

            if not address_list:
                self._json_response(400, {"error": "address is required"})
                return

            address = address_list[0]
            chain = (chain_list[0] if chain_list else None) or None

            try:
                result = flow_gen.generate_html(address, chain, output_dir=directory)
                filename = result["output_path"].name
                contract_names = result.get("contracts", [])
                self._update_registry(
                    directory,
                    address=address,
                    chain=chain,
                    filename=filename,
                    contract_names=contract_names,
                )
                self._json_response(
                    200,
                    {
                        "status": "ok",
                        "file": filename,
                        "url": f"/{filename}",
                        "contracts": contract_names,
                    },
                )
            except ValueError as exc:
                self._json_response(
                    400,
                    {
                        "error": "invalid request",
                        "detail": str(exc),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - return error detail to client
                self._json_response(
                    500,
                    {
                        "error": "failed to generate contract flows",
                        "detail": str(exc),
                    },
                )

        def _handle_generate_stream(self, parsed):
            params = parse_qs(parsed.query)
            address_list = params.get("address") or params.get("addr")
            chain_list = params.get("chain") or params.get("network")

            if not address_list:
                self._sse_response("error", {"error": "address is required"})
                return

            address = address_list[0]
            chain = (chain_list[0] if chain_list else None) or None

            def emit(event: str, payload: dict) -> None:
                self._sse_response(event, payload, keep_open=True)

            def progress_cb(message: str, meta: dict | None = None) -> None:
                payload = {"message": message}
                if meta:
                    payload.update(meta)
                emit("progress", payload)

            emit("progress", {"message": "Starting generation"})

            try:
                result = flow_gen.generate_html(
                    address,
                    chain,
                    output_dir=directory,
                    progress_cb=progress_cb,
                )
                filename = result["output_path"].name
                contract_names = result.get("contracts", [])
                emit("progress", {"message": "Updating registry"})
                self._update_registry(
                    directory,
                    address=address,
                    chain=chain,
                    filename=filename,
                    contract_names=contract_names,
                )
                emit(
                    "done",
                    {
                        "status": "ok",
                        "file": filename,
                        "url": f"/{filename}",
                        "contracts": contract_names,
                    },
                )
            except ValueError as exc:
                emit(
                    "error",
                    {
                        "error": "invalid request",
                        "detail": str(exc),
                    },
                )
            except Exception as exc:  # noqa: BLE001
                emit(
                    "error",
                    {
                        "error": "failed to generate contract flows",
                        "detail": str(exc),
                    },
                )

        def _json_response(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _sse_response(self, event: str, payload: dict, keep_open: bool = False) -> None:
            data = json.dumps(payload)
            if not getattr(self, "_sse_headers_sent", False):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self._sse_headers_sent = True
            self.wfile.write(f"event: {event}\n".encode("utf-8"))
            for line in data.splitlines():
                self.wfile.write(f"data: {line}\n".encode("utf-8"))
            self.wfile.write(b"\n")
            self.wfile.flush()
            if not keep_open:
                self.wfile.flush()

        def _update_registry(self, base_dir: Path, address: str, chain: str, filename: str, contract_names: list[str]):
            registry_path = base_dir / "registry.json"
            try:
                current = json.loads(registry_path.read_text(encoding="utf-8"))
                if not isinstance(current, list):
                    current = []
            except FileNotFoundError:
                current = []
            except json.JSONDecodeError:
                current = []

            # Remove existing entry for the same address+chain
            current = [
                entry for entry in current
                if not (
                    str(entry.get("address", "")).lower() == address.lower()
                    and str(entry.get("chain", "")).lower() == chain.lower()
                )
            ]

            entry = {
                "address": address,
                "chain": chain,
                "path": f"/{filename}",
                "contracts": contract_names,
                "primary_contract": contract_names[0] if contract_names else None,
            }
            current.append(entry)
            registry_path.write_text(json.dumps(current, indent=2), encoding="utf-8")

    with ThreadingHTTPServer((host, port), SrcHandler) as httpd:
        print(f"Serving {directory} at http://{host}:{port} (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


def main() -> None:
    repo_root = Path(__file__).parent
    default_dir = repo_root / "src"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=default_dir,
        help=f"Directory to serve (default: {default_dir})",
    )

    args = parser.parse_args()
    run(args.dir, args.host, args.port)


if __name__ == "__main__":
    main()
