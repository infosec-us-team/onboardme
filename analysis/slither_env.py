"""Environment and chain/address helpers for Slither analysis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

DEFAULT_CHAIN = "mainnet"
SUPPORTED_NETWORK_V2: Dict[str, str] = {
    "mainnet": "1",
    "sepolia": "11155111",
    "holesky": "17000",
    "hoodi": "560048",
    "bsc": "56",
    "testnet.bsc": "97",
    "poly": "137",
    "amoy.poly": "80002",
    "base": "8453",
    "sepolia.base": "84532",
    "arbi": "42161",
    "nova.arbi": "42170",
    "sepolia.arbi": "421614",
    "linea": "59144",
    "sepolia.linea": "59141",
    "blast": "81457",
    "sepolia.blast": "168587773",
    "optim": "10",
    "sepolia.optim": "11155420",
    "avax": "43114",
    "testnet.avax": "43113",
    "bttc": "199",
    "testnet.bttc": "1029",
    "celo": "42220",
    "sepolia.celo": "11142220",
    "frax": "252",
    "hoodi.frax": "2523",
    "gno": "100",
    "mantle": "5000",
    "sepolia.mantle": "5003",
    "memecore": "43521",
    "moonbeam": "1284",
    "moonriver": "1285",
    "moonbase": "1287",
    "opbnb": "204",
    "testnet.opbnb": "5611",
    "scroll": "534352",
    "sepolia.scroll": "534351",
    "taiko": "167000",
    "hoodi.taiko": "167013",
    "era.zksync": "324",
    "sepoliaera.zksync": "300",
    "xdc": "50",
    "testnet.xdc": "51",
    "apechain": "33139",
    "curtis.apechain": "33111",
    "world": "480",
    "sepolia.world": "4801",
    "sophon": "50104",
    "testnet.sophon": "531050104",
    "sonic": "146",
    "testnet.sonic": "14601",
    "unichain": "130",
    "sepolia.unichain": "1301",
    "abstract": "2741",
    "sepolia.abstract": "11124",
    "berachain": "80094",
    "testnet.berachain": "80069",
    "swellchain": "1923",
    "testnet.swellchain": "1924",
    "testnet.monad": "10143",
    "hyperevm": "999",
    "katana": "747474",
    "bokuto.katana": "737373",
    "sei": "1329",
    "testnet.sei": "1328",
}
CHAIN_ID_TO_NAME: Dict[str, str] = {
    chain_id: name for name, chain_id in SUPPORTED_NETWORK_V2.items()
}
DEFAULT_ADDRESS = "0x80ac24aA929eaF5013f6436cdA2a7ba190f5Cc0b"
_DOTENV_LOADED = False


def _load_dotenv(path: Path | None = None) -> None:
    """Load key=value pairs from .env into environment if not already set."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    env_path = path or Path(__file__).resolve().parent.parent / ".env"
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


def _normalize_address(address: str | None) -> str:
    """Return a lowercased 0x-prefixed address string for comparisons."""
    if not address:
        return ""
    addr = address.strip().lower()
    if not addr.startswith("0x"):
        addr = f"0x{addr}"
    return addr


def _normalize_chain(chain: str) -> str:
    """Return canonical chain name from chain name or chain id."""
    raw = (chain or "").strip().lower()
    if not raw:
        raise ValueError("chain is required")
    if raw in SUPPORTED_NETWORK_V2:
        return raw
    if raw.isdigit():
        mapped = CHAIN_ID_TO_NAME.get(raw)
        if mapped:
            return mapped
    raise ValueError(f"Unsupported chain '{chain}'.")


def _resolve_chain_and_address(address: str, chain: str | None) -> tuple[str, str]:
    """Resolve chain name and address, supporting prefix 'chain:address'."""
    if not address:
        raise ValueError("address is required")
    addr = address.strip()
    resolved_chain: str | None = None

    if ":" in addr:
        prefix, addr_part = addr.split(":", 1)
        prefix = prefix.strip()
        addr = addr_part.strip()
        if not addr:
            raise ValueError("address is required")
        resolved_chain = _normalize_chain(prefix)
        if chain:
            chain_name = _normalize_chain(chain)
            if chain_name != resolved_chain:
                raise ValueError(
                    f"Conflicting chain values '{prefix}' and '{chain}'."
                )

    if resolved_chain is None:
        resolved_chain = _normalize_chain(chain) if chain else DEFAULT_CHAIN

    return addr, resolved_chain
