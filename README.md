# clairwave-mcp

**An open MCP server that gives AI assistants physically grounded ocean acoustics.**

[Clairwave](https://www.clairwave.com) runs validated propagation models (Bellhop,
RAM/parabolic equation) on global bathymetry and seasonal sound-speed profiles,
tracks live AIS vessels, and serves 3D hull models for them
([shipshape](https://github.com/clairwave/shipshape)). This server exposes that
to Claude, ChatGPT, Gemini and any other MCP client — so an assistant reasoning
about the ocean can *run the physics* instead of guessing.

Every result carries provenance (model, data source, `run_id`) and an
`open_url` that opens the exact result in the platform. Simulation results
include the bathymetry, sound-speed profile and bottom parameters that were
used, so a researcher can replicate the run in MATLAB, Python or anything else.

**Endpoint (no auth, no key):** `https://www.clairwave.com/mcp` — Streamable HTTP.

## Connect

- **Claude Code:** `claude mcp add --transport http clairwave https://www.clairwave.com/mcp`
- **Claude.ai / Claude Desktop:** Settings → Connectors → *Add custom connector* → the URL above
- **ChatGPT:** Settings → Connectors → *Create* (developer mode) → the URL above
- **Any MCP client:** point it at the URL; the server is stateless and JSON-response capable

## Tools

| Tool | What it does |
|---|---|
| `search_vessels` | Live AIS search by name / MMSI prefix (global feed) |
| `get_vessel` | Vessel particulars + 3D model (unique or class archetype), GLB URL, platform link |
| `get_vessel_photo` | Wikimedia Commons photo with attribution |
| `about` | What the platform provides and how results are grounded |
| *(next)* `get_sound_speed_profile`, `get_bathymetry`, `run_transmission_loss`, `estimate_detection_range` | Physics — coming as the wrapping of the simulation API lands |

## Run locally

```bash
pip install "mcp[cli]<2" httpx
python server.py            # streamable HTTP on :8890 (/mcp)
python server.py --stdio    # stdio for local clients
python tests/smoke_client.py
```

Environment: `CLAIRWAVE_API`, `CLAIRWAVE_FLEET`, `CLAIRWAVE_SITE`, `MCP_PORT`.

## License

MIT. Data: AIS via the AISHub peer network (Clairwave contributes receivers);
vessel photos CC-licensed with attribution; bathymetry and SSP sources cited in
each response.
