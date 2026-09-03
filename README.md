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
| `get_bathymetry` | GEBCO 2025 depth at a point, or a transect profile along a bearing |
| `get_sound_speed_profile` | GDEM v3 seasonal c(z) for a month + seabed parameters (cp, cs, density, attenuation, sediment) |
| `run_transmission_loss` | RAM parabolic-equation TL along a bearing; bathymetry/SSP/seabed fetched automatically; replication bundle included |
| `estimate_detection_range` | Sonar equation on a RAM run: continuous and furthest detection range, signal excess vs range |
| `run_bellhop_volume` | 3D Bellhop TL volume stored under a run id (uint8 cube + JSON sidecar links) |
| `vessel_source_level` | Ship radiated noise: broadband + third-octave spectrum + mechanism breakdown (ECHO/RANDI-class model) |
| `search_vessels` / `vessels_near` | Live AIS by name/MMSI, or within a radius of a point |
| `get_vessel` | Live position/track, particulars, and the 3D model (GLB, bow=+Z) with platform links |
| `get_vessel_photo` | Wikimedia Commons photo with attribution |
| `about` | Models, data sources, limits |

Typical latency against the live platform: bathymetry 0.5 s, SSP 6 s first time
per 0.1° cell then cached, RAM transmission loss 1–3 s, detection range 1–3 s.

## Run locally

```bash
pip install "mcp[cli]<2" httpx
python server.py            # streamable HTTP on :8890 (/mcp)
python server.py --stdio    # stdio for local clients
python tests/smoke_client.py
```

Environment: `CLAIRWAVE_API`, `CLAIRWAVE_FLEET`, `CLAIRWAVE_SITE`, `MCP_PORT`.

## Backend identity and analytics

The MCP endpoint itself is open (no caller auth). The server calls the Clairwave
API as a dedicated Keycloak service account (`clairwave-mcp`, client-credentials
grant, realm role `premium`), so every caller gets the full solver set without
free-tier caps, and all MCP traffic is attributed to one identity in the
platform logs. Configure it with two environment variables (see `.env.example`);
without them the server still works, calling the platform anonymously.

Every tool call is appended to `analytics.jsonl` (tool, arguments, latency,
MCP client name/version from the initialize handshake, source IP). Summarise with:

```
python mcp_stats.py analytics.jsonl --days 7
```

Per-caller authentication (MCP OAuth mapped to Clairwave accounts and tiers) is
the planned path for paid usage; the service-account identity stays as the
default for anonymous callers.

## License

MIT. Data: AIS via the AISHub peer network (Clairwave contributes receivers);
vessel photos CC-licensed with attribution; bathymetry and SSP sources cited in
each response.
