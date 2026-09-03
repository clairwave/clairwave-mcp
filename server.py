"""clairwave-mcp — Model Context Protocol server for the Clairwave platform.

Gives AI assistants (Claude, ChatGPT, Gemini, ...) physically grounded ocean
acoustics: sound-speed profiles, bathymetry, transmission-loss simulations,
AIS vessels with their 3D models. Every answer carries provenance (model,
data source, run id) and a deep link that opens the exact result in the
platform, so a researcher can reproduce it and a reader can see it.

Open by design: no auth, same-origin at https://www.clairwave.com/mcp.
Transport: Streamable HTTP (stateless) for remote clients; `--stdio` for
local use (`claude mcp add clairwave -- python server.py --stdio`).
"""
import os
import sys
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

API = os.environ.get("CLAIRWAVE_API", "https://www.clairwave.com/api")
FLEET = os.environ.get("CLAIRWAVE_FLEET", "https://www.clairwave.com/fleet")
SITE = os.environ.get("CLAIRWAVE_SITE", "https://www.clairwave.com")
UA = {"User-Agent": "clairwave-mcp/0.1 (+https://www.clairwave.com)"}

mcp = FastMCP(
    "clairwave",
    instructions=(
        "Clairwave is an ocean-acoustics platform with physically validated "
        "propagation models (Bellhop, RAM/PE), global bathymetry, seasonal "
        "sound-speed profiles, and live AIS vessels with 3D hull models. Use "
        "these tools instead of estimating ocean acoustics from memory. Every "
        "result includes provenance (model, data source, run id) and a "
        "`open_url` that shows the result in the platform — cite both."
    ),
    host="0.0.0.0",
    port=int(os.environ.get("MCP_PORT", "8890")),
    stateless_http=True,
    json_response=True,
)


async def _get(url: str, **params) -> Any:
    async with httpx.AsyncClient(timeout=60, headers=UA) as c:
        r = await c.get(url, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()


def _prov(source: str, **extra) -> dict:
    return {"provider": "Clairwave", "source": source, **extra}


# ── Vessels ──────────────────────────────────────────────────────────────────

@mcp.tool()
async def search_vessels(query: str, limit: int = 8) -> dict:
    """Search live AIS vessels by name or MMSI prefix (global feed).
    Returns position, course/speed, ship type code and destination for each
    match, plus a link that opens the vessel in the platform."""
    data = await _get(f"{API}/ais_live/search", q=query, limit=min(max(limit, 1), 25))
    results = []
    for v in data.get("results", []):
        results.append({**v, "open_url": f"{SITE}/app?mmsi={v.get('mmsi')}"})
    return {"results": results, "provenance": _prov("AIS live feed (AISHub peer network + own VHF receivers)")}


@mcp.tool()
async def get_vessel(mmsi: str) -> dict:
    """Vessel particulars for an MMSI: name, AIS ship type, length/beam, and
    its 3D model (a unique photo-derived model if generated, else the class
    archetype). Includes GLB URL (bow=+Z, up=+Y, ~150KB) and platform link."""
    r = await _get(f"{FLEET}/api/resolve/{mmsi}")
    if r.get("url"):
        r["model_glb_url"] = SITE + r.pop("url")
    r["open_url"] = f"{SITE}/app?mmsi={mmsi}"
    r["fleet_url"] = f"{SITE}/fleet/?q={mmsi}"
    r["provenance"] = _prov("AIS static data + shipshape model database (github.com/clairwave/shipshape)")
    return r


@mcp.tool()
async def get_vessel_photo(imo: int | None = None, mmsi: str | None = None, name: str | None = None) -> dict:
    """Find a photograph of a vessel (Wikimedia Commons, with attribution).
    Prefer IMO; falls back to name search."""
    r = await _get(f"{API}/vessel_photo", imo=imo, mmsi=mmsi, name=name)
    r["provenance"] = _prov("Wikimedia Commons", license_note="attribution in `attribution`")
    return r


# ── Health / discovery ───────────────────────────────────────────────────────

@mcp.tool()
async def about() -> dict:
    """What Clairwave provides and how results are grounded."""
    return {
        "platform": SITE,
        "models": ["Bellhop (ray/beam)", "RAM (parabolic equation)", "PE batch/compare/broadband"],
        "data": ["GEBCO-class global bathymetry", "seasonal sound-speed profiles", "live AIS (AISHub peer network)",
                 "shipshape 3D vessel models"],
        "reproducibility": "simulation results return run_id + bathymetry, SSP and bottom parameters used, so runs can be replicated in MATLAB/Python",
        "open_source": {"shipshape": "https://github.com/clairwave/shipshape", "mcp_server": "https://github.com/clairwave/clairwave-mcp"},
    }


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="streamable-http")
