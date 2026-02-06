#!/usr/bin/env python3
"""HTTP server to host `src/` and expose a contract generation API."""

from __future__ import annotations

import argparse
import json
import os
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode
from urllib.request import urlopen, Request
import re

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
        _ETH_CALL_MAX_BODY_BYTES = 64 * 1024
        _RPC_REQUIRED_CHAINIDS = {
            "56",  # BNB Smart Chain Mainnet
            "97",  # BNB Smart Chain Testnet
            "8453",  # Base Mainnet
            "84532",  # Base Sepolia
            "10",  # OP Mainnet
            "11155420",  # OP Sepolia
            "43114",  # Avalanche C-Chain
            "43113",  # Avalanche Fuji
        }

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
            if parsed.path == "/eth_call":
                length = int(self.headers.get("Content-Length", 0))
                if length > self._ETH_CALL_MAX_BODY_BYTES:
                    self._json_response(413, {"error": "request body too large"})
                    return
                body = self.rfile.read(length).decode("utf-8") if length else ""
                params = parse_qs(body) if "=" in body else {}
                try:
                    json_body = json.loads(body) if body else {}
                except json.JSONDecodeError:
                    json_body = {}
                merged: dict[str, list[str]] = {}
                if isinstance(params, dict):
                    merged.update({k: [str(x) for x in v] for k, v in params.items()})
                if isinstance(json_body, dict):
                    for k, v in json_body.items():
                        if isinstance(v, list):
                            merged[k] = [str(x) for x in v]
                        else:
                            merged[k] = [str(v)]
                self._handle_eth_call(merged)
                return
            return super().do_POST()

        def _handle_eth_call(self, params: dict[str, list[str]]):
            def _first(key: str) -> str:
                vals = params.get(key) or []
                return (vals[0] if vals else "").strip()

            chainid = _first("chainid") or _first("chain_id") or _first("chain")
            to = _first("to") or _first("address")
            # Clients/providers vary on field name: "data" vs "input". Accept either.
            data = _first("data") or _first("input")
            tag = _first("tag") or "latest"

            # Strict input validation: reject anything outside expected character sets.
            # This endpoint never executes shell commands, but we still validate to prevent
            # abuse (huge payloads, weird encodings) and keep the surface area tight.
            if not chainid or not re.fullmatch(r"[0-9]{1,12}", chainid):
                self._json_response(400, {"error": "chainid is required and must be numeric"})
                return
            if not re.fullmatch(r"0x[0-9a-fA-F]{40}", to or ""):
                self._json_response(400, {"error": "to must be a 0x-prefixed 20-byte hex address"})
                return
            if not re.fullmatch(r"0x[0-9a-fA-F]*", data or ""):
                self._json_response(400, {"error": "data/input must be 0x-prefixed hex calldata"})
                return
            # EVM calldata is bytes; require even-length hex after 0x.
            if (len(data) - 2) % 2 != 0:
                self._json_response(400, {"error": "data hex length must be even"})
                return
            # Guard against very large query strings / upstream limits.
            if len(data) > 2 + 32_768:  # 16KiB of calldata as hex chars
                self._json_response(400, {"error": "data too large"})
                return

            # Only allow standard JSON-RPC tags or a specific block number hex quantity.
            tag = (tag or "").strip().lower()
            if tag in {"latest", "earliest", "pending"}:
                pass
            elif re.fullmatch(r"0x[0-9a-f]+", tag or ""):
                pass
            else:
                self._json_response(400, {"error": "invalid tag"})
                return

            # For some chains, the free Etherscan API does not support eth_call via the
            # v2 proxy endpoint. In those cases we require a per-chain RPC URL.
            if chainid in self._RPC_REQUIRED_CHAINIDS:
                env_key = f"RPC_URL_{chainid}"
                rpc_url = os.environ.get(env_key, "").strip()
                if not rpc_url:
                    self._json_response(
                        400,
                        {
                            "error": f"{env_key} not set",
                            "detail": f"Chain id {chainid} requires a direct RPC URL for eth_call.",
                        },
                    )
                    return
                if not re.fullmatch(r"https?://.+", rpc_url):
                    self._json_response(400, {"error": f"{env_key} must start with http:// or https://"})
                    return

                body = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_call",
                    # Some RPC providers expect "input" instead of "data". Send both.
                    "params": [{"to": to, "data": data, "input": data}, tag],
                }
                try:
                    req = Request(
                        rpc_url,
                        data=json.dumps(body).encode("utf-8"),
                        headers={"Accept": "application/json", "Content-Type": "application/json"},
                        method="POST",
                    )
                    with urlopen(req, timeout=20) as resp:
                        raw = resp.read().decode("utf-8", errors="replace")
                    payload = json.loads(raw) if raw else {}
                except Exception as exc:  # noqa: BLE001 - return error detail to client
                    self._json_response(502, {"error": "rpc request failed", "detail": str(exc)})
                    return

                if isinstance(payload, dict) and "result" in payload:
                    self._json_response(200, {"result": payload.get("result")})
                    return

                detail = ""
                try:
                    if isinstance(payload, dict):
                        if isinstance(payload.get("error"), dict):
                            detail = str(payload["error"].get("message") or payload["error"])
                        else:
                            detail = str(payload.get("message") or payload.get("error") or payload)
                    else:
                        detail = str(payload)
                except Exception:
                    detail = "unknown error"
                self._json_response(502, {"error": "rpc returned error", "detail": detail[:500]})
                return

            api_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()
            if not api_key:
                self._json_response(400, {"error": "ETHERSCAN_API_KEY not set"})
                return

            query = urlencode(
                {
                    "chainid": chainid,
                    "module": "proxy",
                    "action": "eth_call",
                    "to": to,
                    "data": data,
                    "tag": tag,
                    "apikey": api_key,
                }
            )
            url = f"https://api.etherscan.io/v2/api?{query}"
            try:
                req = Request(url, headers={"Accept": "application/json"})
                with urlopen(req, timeout=20) as resp:
                    raw = resp.read().decode("utf-8", errors="replace")
                payload = json.loads(raw) if raw else {}
            except Exception as exc:  # noqa: BLE001 - return error detail to client
                self._json_response(502, {"error": "etherscan request failed", "detail": str(exc)})
                return

            # Expected successful response: {"jsonrpc":"2.0","id":1,"result":"0x..."}
            # Errors usually have an "error" object.
            if isinstance(payload, dict) and "result" in payload:
                self._json_response(200, {"result": payload.get("result")})
                return

            # Don't forward arbitrary structured payloads (keeps response surface small).
            detail = ""
            try:
                if isinstance(payload, dict):
                    if isinstance(payload.get("error"), dict):
                        detail = str(payload["error"].get("message") or payload["error"])
                    else:
                        detail = str(payload.get("message") or payload.get("error") or payload)
                else:
                    detail = str(payload)
            except Exception:
                detail = "unknown error"
            self._json_response(502, {"error": "etherscan returned error", "detail": detail[:500]})

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
                primary_contract = (
                    result.get("deployed_contract") or (contract_names[0] if contract_names else None)
                )
                self._update_registry(
                    directory,
                    address=address,
                    chain=chain,
                    filename=filename,
                    contract_names=contract_names,
                    primary_contract=primary_contract,
                )
                self._json_response(
                    200,
                    {
                        "status": "ok",
                        "file": filename,
                        "url": f"/{filename}",
                        "contracts": contract_names,
                        "primary_contract": primary_contract,
                        "deployed_contract": result.get("deployed_contract"),
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
                primary_contract = (
                    result.get("deployed_contract") or (contract_names[0] if contract_names else None)
                )
                emit("progress", {"message": "Updating registry"})
                self._update_registry(
                    directory,
                    address=address,
                    chain=chain,
                    filename=filename,
                    contract_names=contract_names,
                    primary_contract=primary_contract,
                )
                emit(
                    "done",
                    {
                        "status": "ok",
                        "file": filename,
                        "url": f"/{filename}",
                        "contracts": contract_names,
                        "primary_contract": primary_contract,
                        "deployed_contract": result.get("deployed_contract"),
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

        def _update_registry(
            self,
            base_dir: Path,
            address: str,
            chain: str,
            filename: str,
            contract_names: list[str],
            primary_contract: str | None = None,
        ):
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
                "primary_contract": primary_contract or (contract_names[0] if contract_names else None),
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
