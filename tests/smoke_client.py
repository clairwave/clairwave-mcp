"""Smoke test: connect over Streamable HTTP, list tools, exercise the physics
and vessel tools against the live platform.
Usage: python tests/smoke_client.py [http://127.0.0.1:8890/mcp]"""
import asyncio
import json
import sys
import time

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8890/mcp"
GIB = {"lat": 36.0, "lon": -5.5}

CALLS = [
    ("get_bathymetry", {**GIB}),
    ("get_sound_speed_profile", {**GIB, "month": 3}),
    ("run_transmission_loss", {**GIB, "source_depth_m": 20, "frequency_hz": 150, "bearing_deg": 90, "range_km": 20, "month": 3}),
    ("estimate_detection_range", {**GIB, "source_depth_m": 8, "frequency_hz": 150, "source_level_db": 170,
                                   "receiver_depth_m": 100, "noise_level_db": 75, "bearing_deg": 90, "max_range_km": 30, "month": 3}),
    ("vessels_near", {**GIB, "radius_km": 15, "limit": 3}),
    ("vessel_source_level", {"ship_type": 70, "speed_kn": 12, "length_m": 200, "beam_m": 30, "draft_m": 9}),
    ("get_vessel", {"mmsi": "636018938"}),
]


async def main():
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])
            for name, args in CALLS:
                t0 = time.time()
                r = await s.call_tool(name, args)
                txt = r.content[0].text if r.content else ""
                try:
                    d = json.loads(txt)
                    keys = list(d)[:8]
                    brief = {k: (d[k] if not isinstance(d[k], (dict, list)) else f"<{type(d[k]).__name__} {len(d[k])}>") for k in keys}
                except Exception:
                    brief = txt[:200]
                print(f"\n== {name} ({time.time()-t0:.1f}s) err={r.isError}: {brief}")


asyncio.run(main())
