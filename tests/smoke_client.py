"""Smoke test: connect to the server over Streamable HTTP, list tools, call a few.
Usage: python tests/smoke_client.py [http://127.0.0.1:8890/mcp]"""
import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8890/mcp"


async def main():
    async with streamablehttp_client(URL) as (read, write, _):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            print("tools:", [t.name for t in tools.tools])
            for name, args in [("search_vessels", {"query": "NAVIGATOR", "limit": 3}),
                               ("get_vessel", {"mmsi": "636018938"}),
                               ("about", {})]:
                r = await s.call_tool(name, args)
                txt = r.content[0].text if r.content else ""
                print(f"\n== {name}: {txt[:400]}")


asyncio.run(main())
