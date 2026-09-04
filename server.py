"""clairwave-mcp — Model Context Protocol server for the Clairwave platform.

Gives AI assistants (Claude, ChatGPT, Gemini, ...) physically grounded ocean
acoustics: bathymetry, seasonal sound-speed profiles, seabed parameters,
transmission-loss simulations (RAM parabolic equation, Bellhop), vessel
source levels, detection ranges, and live AIS vessels with 3D hull models.

Every answer carries provenance (model, data source, run id) and, where the
platform can show it, an `open_url`. Simulation answers include the
replication bundle (bathymetry profile, SSP, bottom parameters, grid) so a
researcher can reproduce the run in MATLAB / Python.

Open by design: no caller auth, https://www.clairwave.com/mcp (Streamable HTTP).
The server calls the platform as a premium Keycloak service account (see _bearer).
Local: `python server.py --stdio`.
"""
import base64
import datetime as _dt
import functools
import json
import math
import os
import sys
import time
from typing import Any

import httpx
import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from shipnoise import compute_source_level, quick_broadband_sl

API = os.environ.get("CLAIRWAVE_API", "https://www.clairwave.com/api")
FLEET = os.environ.get("CLAIRWAVE_FLEET", "https://www.clairwave.com/fleet")
SITE = os.environ.get("CLAIRWAVE_SITE", "https://www.clairwave.com")
# Run files (npy/json sidecars) live on the shared public volume served by the
# site's nginx at /public/ — the backend's /json and /npy routes are pod-local.
# NB: Cloudflare blocks generic python user agents (error 1010); ours is allowed.
UA = {"User-Agent": "clairwave-mcp/0.3 (+https://www.clairwave.com)"}

# ── Backend identity ─────────────────────────────────────────────────────────
# The MCP endpoint stays open (no caller auth), but the server itself talks to
# the platform as a dedicated Keycloak service account (client_credentials,
# realm role premium -> `tier` claim). Callers therefore get the full solver
# set with no free-tier caps, and every backend log line / run is attributed to
# this identity. Without credentials the server falls back to anonymous calls.
KC_TOKEN_URL = os.environ.get("CW_KC_TOKEN_URL", f"{SITE}/auth/realms/clairwave/protocol/openid-connect/token")
MCP_CLIENT_ID = os.environ.get("CW_MCP_CLIENT_ID")
MCP_CLIENT_SECRET = os.environ.get("CW_MCP_CLIENT_SECRET")
_TOKEN: dict = {"value": None, "exp": 0.0, "tier": None}

# Per-call analytics (JSONL): tool, args, latency, MCP client name/version, IP.
LOG_PATH = os.environ.get("CW_MCP_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.jsonl"))

mcp = FastMCP(
    "clairwave",
    instructions=(
        "Clairwave is an ocean-acoustics platform: validated propagation models "
        "(RAM parabolic equation, Bellhop), GEBCO bathymetry, GDEM seasonal "
        "sound-speed climatology, seabed lithology, vessel source-level models, "
        "and live AIS with 3D hull models. Use these tools instead of estimating "
        "ocean acoustics from memory. Typical questions: 'what is the sound speed "
        "profile at X in March', 'how far can a 150 Hz source be heard from X', "
        "'transmission loss along bearing B', 'what ships are near X'. Every "
        "result includes provenance and a replication bundle — cite the model, "
        "the data sources and the run id, and link `open_url` when present."
    ),
    host="0.0.0.0",
    port=int(os.environ.get("MCP_PORT", "8890")),
    stateless_http=True,
    json_response=True,
)


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _jwt_claims(tok: str) -> dict:
    try:
        p = tok.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(p + "=" * (-len(p) % 4)))
    except Exception:
        return {}


async def _bearer(url: str) -> dict:
    """Authorization header for platform API calls (cached service-account token)."""
    if not (MCP_CLIENT_ID and MCP_CLIENT_SECRET) or not url.startswith(API):
        return {}
    if _TOKEN["value"] and time.time() < _TOKEN["exp"] - 30:
        return {"Authorization": f"Bearer {_TOKEN['value']}"}
    try:
        async with httpx.AsyncClient(timeout=20, headers=UA) as c:
            r = await c.post(KC_TOKEN_URL, data={"grant_type": "client_credentials",
                                                 "client_id": MCP_CLIENT_ID, "client_secret": MCP_CLIENT_SECRET})
            r.raise_for_status()
            j = r.json()
        _TOKEN["value"] = j["access_token"]
        _TOKEN["exp"] = time.time() + float(j.get("expires_in", 300))
        _TOKEN["tier"] = _jwt_claims(j["access_token"]).get("tier")
        return {"Authorization": f"Bearer {_TOKEN['value']}"}
    except Exception as e:  # fail open: anonymous call rather than no call
        print(f"[auth] service-account token failed, calling anonymously: {e}", file=sys.stderr)
        return {}


def _client_meta() -> dict:
    """Who is calling: MCP client name/version (from initialize) and source IP."""
    meta: dict = {}
    try:
        ctx = mcp.get_context()
        cp = ctx.session.client_params
        if cp is not None and cp.clientInfo is not None:
            meta["client"] = cp.clientInfo.name
            meta["client_version"] = cp.clientInfo.version
        req = ctx.request_context.request
        if req is not None:
            h = req.headers
            meta["ip"] = (h.get("cf-connecting-ip") or h.get("x-forwarded-for", "").split(",")[0].strip()
                          or (req.client.host if req.client else None))
            meta["ua"] = h.get("user-agent")
    except Exception:
        pass
    return meta


def logged(fn):
    """Append one JSONL record per tool call to LOG_PATH (never raises)."""
    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        t0 = time.time()
        ok, err = True, None
        try:
            return await fn(*args, **kwargs)
        except Exception as e:
            ok, err = False, str(e)[:300]
            raise
        finally:
            rec = {"ts": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   "tool": fn.__name__, "args": kwargs, "ms": int((time.time() - t0) * 1000),
                   "ok": ok, "error": err, "tier": _TOKEN.get("tier"), **_client_meta()}
            try:
                with open(LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
            except Exception:
                pass
    return wrapper


async def _get(url: str, **params) -> Any:
    async with httpx.AsyncClient(timeout=90, headers={**UA, **await _bearer(url)}) as c:
        r = await c.get(url, params={k: v for k, v in params.items() if v is not None})
        r.raise_for_status()
        return r.json()


async def _post(url: str, body: dict, timeout: float = 180) -> Any:
    async with httpx.AsyncClient(timeout=timeout, headers={**UA, **await _bearer(url)}) as c:
        r = await c.post(url, json=body)
        if r.status_code >= 400:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:300]
            raise RuntimeError(f"{url.rsplit('/', 1)[-1]} -> HTTP {r.status_code}: {detail}")
        return r.json()


def _prov(source: str, **extra) -> dict:
    return {"provider": "Clairwave", "source": source, **extra}


def _decode_tl(b64: str, shape: list[int], dtype: str = "float16") -> np.ndarray:
    a = np.frombuffer(base64.b64decode(b64), dtype=np.float16 if dtype == "float16" else np.uint8)
    return a.reshape(shape[0], shape[1]).astype(np.float32)  # (Nr, Nz); 999 = below seafloor


def _bbox(lat: float, lon: float, radius_km: float) -> dict:
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.05))
    return {"latmin": lat - dlat, "latmax": lat + dlat, "lonmin": lon - dlon, "lonmax": lon + dlon}


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p = math.pi / 180
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2
    return 12742 * math.asin(math.sqrt(a))


# ── Environment: bathymetry, SSP, seabed ─────────────────────────────────────

async def _bathy_transect(lat: float, lon: float, bearing_deg: float, range_m: float, n_points: int = 200) -> dict:
    d = await _post(f"{API}/terrain/radial_bathy",
                    {"lat": lat, "lon": lon, "bearings": [bearing_deg], "r_max": range_m, "n_points": n_points})
    prof = d["profiles"][str(int(round(bearing_deg)))]
    return {"r_m": prof["r"], "depth_m": prof["z"]}


@mcp.tool(title="Bathymetry (GEBCO)", annotations=ToolAnnotations(title="Bathymetry (GEBCO)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def get_bathymetry(lat: float, lon: float, bearing_deg: float | None = None,
                         range_km: float = 20.0, n_points: int = 100) -> dict:
    """Seafloor depth from GEBCO 2025 (15 arc-second grid, ~450 m).
    Without a bearing: depth at the point. With a bearing (compass degrees,
    0 = north): a depth profile along that transect out to range_km."""
    if bearing_deg is None:
        t = await _bathy_transect(lat, lon, 0, 50, 2)
        return {"lat": lat, "lon": lon, "depth_m": round(float(t["depth_m"][0]), 1),
                "provenance": _prov("GEBCO 2025 global grid, bilinear sample")}
    t = await _bathy_transect(lat, lon, bearing_deg, range_km * 1000, max(2, min(n_points, 400)))
    zs = t["depth_m"]
    return {"lat": lat, "lon": lon, "bearing_deg": bearing_deg, "range_km": range_km,
            "profile": [{"r_m": round(r, 1), "depth_m": round(z, 1)} for r, z in zip(t["r_m"], zs)],
            "min_depth_m": round(min(zs), 1), "max_depth_m": round(max(zs), 1),
            "provenance": _prov("GEBCO 2025 global grid, bilinear samples along a great-circle bearing")}


_ENV_CACHE: dict[tuple, dict] = {}


async def _environment(lat: float, lon: float, month: int) -> dict:
    """GDEM v3 seasonal SSP + seabed lithology at (lat, lon) via a small
    Bellhop run — the only server path that returns both. Cached per 0.1°."""
    key = (round(lat, 1), round(lon, 1), int(month))
    if key in _ENV_CACHE:
        return _ENV_CACHE[key]
    # Bellhop needs a sane grid: the demo's defaults (10 km, bty 500) run in ~5 s;
    # coarser/smaller settings make the solver bail with no .shd output.
    body = {"src_lat": lat, "src_lon": lon, "src_radius_km": 10.0, "month": int(month),
            "center_frequency": 200.0, "input_depth": 20}
    res = await _post(f"{API}/run_sim", body, timeout=240)
    run_id = str(res["json_file"]).replace(".json", "")
    meta = await _get(f"{SITE}/public/{res['json_file']}")
    bp = meta.get("bounding_box_center_bot_prm") or []
    env = {
        "run_id": run_id, "month": int(month),
        "ssp": {"depth_m": [round(float(z), 1) for z in meta.get("ssp", {}).get("z", [])],
                "c_m_s": [round(float(c), 2) for c in meta.get("ssp", {}).get("c", [])]},
        # producer order (bottom_parameter_lookup): cp, cs, rho_ratio, alpha_p, alpha_s, name
        "bottom": ({"cp_m_s": bp[0], "cs_m_s": bp[1], "rho_ratio": bp[2], "alpha_p_dB_per_lambda": bp[3],
                    "alpha_s_dB_per_lambda": bp[4], "sediment": bp[5] if len(bp) > 5 else None}
                   if len(bp) >= 5 else None),
        "bottom_params_raw": bp,
        "max_depth_m": meta.get("max_depth"),
        "open_url": f"{SITE}/demo?guest=1&run={run_id}",
        "files": {"json": f"{SITE}/public/{res['json_file']}", "npy": f"{SITE}/public/{res['npy_file']}",
                  "bathy_npy": f"{SITE}/public/{run_id}_bathy.npy"},
        "provenance": _prov("GDEM v3 monthly T/S climatology -> sound speed; seabed lithology lookup; via Bellhop probe run",
                            run_id=run_id),
    }
    _ENV_CACHE[key] = env
    return env


@mcp.tool(title="Sound speed profile (GDEM)", annotations=ToolAnnotations(title="Sound speed profile (GDEM)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def get_sound_speed_profile(lat: float, lon: float, month: int) -> dict:
    """Seasonal sound-speed profile c(z) at a location for a calendar month
    (1-12), from GDEM v3 temperature/salinity climatology, plus the seabed
    parameters (compressional/shear speed, density ratio, attenuation,
    sediment type) at the same point. Cached per 0.1 degree."""
    env = await _environment(lat, lon, month)
    return {"lat": lat, "lon": lon, **env}


# ── Propagation ──────────────────────────────────────────────────────────────

async def _run_ram(lat, lon, source_depth_m, frequency_hz, bearing_deg, range_km, month,
                   z_max_m=None, n_points=200) -> dict:
    env = await _environment(lat, lon, month)
    bathy = await _bathy_transect(lat, lon, bearing_deg, range_km * 1000, n_points)
    zmax = z_max_m or max(200.0, min(6000.0, max(bathy["depth_m"]) * 1.15 + 50))
    body = {"freq": float(frequency_hz), "z_max": zmax, "r_max": range_km * 1000, "z_src": float(source_depth_m),
            "method": "ram", "theta_deg": float(bearing_deg),
            "bathy_r": bathy["r_m"], "bathy_z": bathy["depth_m"],
            "bottom_lat": lat, "bottom_lon": lon, "Nz": 512, "Nr": 500}
    if env["ssp"]["depth_m"]:
        body["ssp_z"] = env["ssp"]["depth_m"]
        body["ssp_c"] = env["ssp"]["c_m_s"]
    t0 = time.time()
    res = await _post(f"{API}/run_pe", body, timeout=240)
    if not res.get("success"):
        raise RuntimeError(res.get("error", "solver failed"))
    tl = _decode_tl(res["TL_b64"], res["shape"], res.get("dtype", "float16"))
    return {"env": env, "bathy": bathy, "tl": tl, "res": res, "wall_s": round(time.time() - t0, 2), "z_max": zmax}


def _sample(tl: np.ndarray, r_range, z_range, receiver_depths, n_out=40) -> dict:
    nr, nz = tl.shape
    rs = np.linspace(r_range[0], r_range[1], nr)
    zs = np.linspace(z_range[0], z_range[1], nz)
    idx = np.unique(np.linspace(0, nr - 1, min(n_out, nr)).astype(int))
    out = {"range_m": [round(float(rs[i]), 0) for i in idx], "tl_db": {}}
    for zr in receiver_depths:
        k = int(np.argmin(np.abs(zs - zr)))
        col = tl[:, k]
        out["tl_db"][f"{zr:g}m"] = [None if col[i] >= 998 else round(float(col[i]), 1) for i in idx]
    return out


@mcp.tool(title="Transmission loss (RAM PE)", annotations=ToolAnnotations(title="Transmission loss (RAM PE)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def run_transmission_loss(lat: float, lon: float, source_depth_m: float, frequency_hz: float,
                                bearing_deg: float, range_km: float = 20.0, month: int = 6,
                                receiver_depths_m: list[float] | None = None) -> dict:
    """Run a physically grounded transmission-loss simulation (RAM parabolic
    equation) from a source at (lat, lon, depth) along a compass bearing.
    Bathymetry (GEBCO), the seasonal sound-speed profile (GDEM, `month`) and
    seabed parameters are fetched for the location automatically. Returns TL
    vs range at the receiver depths (default: source depth, 10 m, 100 m,
    mid-water), grid statistics, and the full replication bundle."""
    rd = receiver_depths_m or sorted({float(source_depth_m), 10.0, 100.0})
    r = await _run_ram(lat, lon, source_depth_m, frequency_hz, bearing_deg, range_km, month)
    res, tl = r["res"], r["tl"]
    valid = tl[tl < 998]
    curves = _sample(tl, res["r_range"], res["z_range"], rd)
    return {
        "inputs": {"lat": lat, "lon": lon, "source_depth_m": source_depth_m, "frequency_hz": frequency_hz,
                   "bearing_deg": bearing_deg, "range_km": range_km, "month": month},
        "method": res.get("method", "ram"), "solver_meta": res.get("meta"),
        "grid": {"shape_Nr_Nz": res["shape"], "r_range_m": res["r_range"], "z_range_m": res["z_range"]},
        "tl_vs_range": curves,
        "stats": {"tl_min_db": round(float(valid.min()), 1) if valid.size else None,
                  "tl_max_db": round(float(valid.max()), 1) if valid.size else None,
                  "elapsed_s": res.get("elapsed_s"), "wall_s": r["wall_s"]},
        "replication": {"ssp": r["env"]["ssp"], "bottom": r["env"]["bottom"],
                        "bathymetry_transect": {"r_m": [round(x, 0) for x in r["bathy"]["r_m"]],
                                                "depth_m": [round(x, 1) for x in r["bathy"]["depth_m"]]},
                        "environment_run_id": r["env"]["run_id"], "environment_files": r["env"]["files"],
                        "note": "TL grid sentinel 999 = below seafloor. Re-run with any PE code using these inputs."},
        "provenance": _prov("RAM parabolic equation (Clairwave backend) with GEBCO 2025 bathymetry, GDEM v3 SSP, seabed lithology",
                            environment_run_id=r["env"]["run_id"]),
    }


@mcp.tool(title="Detection range (sonar equation)", annotations=ToolAnnotations(title="Detection range (sonar equation)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def estimate_detection_range(lat: float, lon: float, source_depth_m: float, frequency_hz: float,
                                   source_level_db: float, receiver_depth_m: float, noise_level_db: float,
                                   bearing_deg: float = 0.0, max_range_km: float = 40.0, month: int = 6,
                                   detection_threshold_db: float = 0.0) -> dict:
    """Passive sonar detection range along a bearing: runs a RAM transmission
    loss simulation for the location/season and applies the sonar equation
    SE = SL - TL - NL (- DT) at the receiver depth. Returns the first range
    where detection is lost, the furthest range still detectable (convergence
    zones), and the excess-vs-range curve. Units: dB re 1 uPa (SL @1 m, NL in
    the analysis band)."""
    r = await _run_ram(lat, lon, source_depth_m, frequency_hz, bearing_deg, max_range_km, month)
    res, tl = r["res"], r["tl"]
    nr, nz = tl.shape
    rs = np.linspace(res["r_range"][0], res["r_range"][1], nr)
    zs = np.linspace(res["z_range"][0], res["z_range"][1], nz)
    k = int(np.argmin(np.abs(zs - receiver_depth_m)))
    col = tl[:, k].astype(np.float64)
    se = source_level_db - col - noise_level_db - detection_threshold_db
    valid = col < 998                      # below-seafloor samples are "no data", not "lost"
    ok = valid & (se >= 0)
    first_loss = None
    for i in range(nr):
        if valid[i] and se[i] < 0:
            first_loss = float(rs[i]); break
    last_ok = float(rs[np.where(ok)[0][-1]]) if ok.any() else 0.0
    idx = np.unique(np.linspace(0, nr - 1, min(40, nr)).astype(int))
    return {
        "inputs": {"lat": lat, "lon": lon, "source_depth_m": source_depth_m, "frequency_hz": frequency_hz,
                   "source_level_db": source_level_db, "receiver_depth_m": receiver_depth_m,
                   "noise_level_db": noise_level_db, "detection_threshold_db": detection_threshold_db,
                   "bearing_deg": bearing_deg, "max_range_km": max_range_km, "month": month},
        "continuous_detection_range_km": round((first_loss if first_loss is not None else float(rs[-1])) / 1000, 2),
        "furthest_detectable_range_km": round(last_ok / 1000, 2),
        "detectable_fraction_of_track": round(float(ok[valid].mean()), 3) if valid.any() else 0.0,
        "receiver_below_seafloor_fraction": round(float((~valid).mean()), 3),
        "signal_excess_vs_range": {"range_km": [round(float(rs[i]) / 1000, 2) for i in idx],
                                   "se_db": [None if col[i] >= 998 else round(float(se[i]), 1) for i in idx]},
        "method": "RAM PE transmission loss + sonar equation SE = SL - TL - NL - DT",
        "replication": {"ssp": r["env"]["ssp"], "bottom": r["env"]["bottom"],
                        "environment_run_id": r["env"]["run_id"], "grid": {"shape": res["shape"], "r_range_m": res["r_range"], "z_range_m": res["z_range"]}},
        "provenance": _prov("RAM PE + GEBCO 2025 + GDEM v3 + seabed lithology", environment_run_id=r["env"]["run_id"]),
    }


@mcp.tool(title="Bellhop 3D TL volume (stored run)", annotations=ToolAnnotations(title="Bellhop 3D TL volume (stored run)", readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def run_bellhop_volume(lat: float, lon: float, source_depth_m: float = 20, frequency_hz: float = 200,
                             radius_km: float = 10, month: int = 6) -> dict:
    """Full 3D transmission-loss volume around a source with Bellhop (all
    bearings), stored on the platform under a run id. Returns the run id,
    the replication metadata (SSP, seabed, bounding box) and file links
    (uint8 TL cube .npy + JSON sidecar). Runs with the server's premium
    identity, so frequency and radius are not free-tier capped (keep radius
    <= ~50 km for reasonable run times)."""
    body = {"src_lat": lat, "src_lon": lon, "src_radius_km": radius_km, "month": int(month),
            "center_frequency": float(frequency_hz), "input_depth": int(source_depth_m), "timing": False}
    res = await _post(f"{API}/run_sim", body, timeout=300)
    run_id = str(res["json_file"]).replace(".json", "")
    meta = await _get(f"{SITE}/public/{res['json_file']}")
    keep = {k: meta.get(k) for k in ("month", "input_depth", "center_frequency", "dbMin", "dbMax", "run_type",
                                     "ssp_type", "max_depth", "bounding_box_center", "bounding_box_x",
                                     "bounding_box_y", "bounding_box_z", "bounding_box_r",
                                     "bounding_box_center_bot_prm", "radial_bathy", "lat1", "lat2", "long1", "long2")}
    return {"run_id": run_id, "open_url": f"{SITE}/demo?guest=1&run={run_id}",
            "metadata": keep, "ssp": meta.get("ssp"),
            "files": {"json": f"{SITE}/public/{res['json_file']}", "npy": f"{SITE}/public/{res['npy_file']}",
                      "bathy_npy": f"{SITE}/public/{run_id}_bathy.npy"},
            "decode": "npy is uint8 TL scaled between dbMin..dbMax over (Ntheta, Nr, Nz); *_bathy.npy uint8 depth = v/255*radial_bathy.scale_depth_m",
            "provenance": _prov("Bellhop 3D (Clairwave) + GEBCO 2025 + GDEM v3 + seabed lithology", run_id=run_id)}


# ── Vessels ──────────────────────────────────────────────────────────────────

@mcp.tool(title="Search vessels (AIS)", annotations=ToolAnnotations(title="Search vessels (AIS)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def search_vessels(query: str, limit: int = 8) -> dict:
    """Search live AIS vessels by name or MMSI prefix (global feed)."""
    data = await _get(f"{API}/ais_live/search", q=query, limit=min(max(limit, 1), 25))
    results = [{**v, "open_url": f"{SITE}/demo?guest=1&mmsi={v.get('mmsi')}"} for v in data.get("results", [])]
    return {"results": results, "provenance": _prov("AIS live feed (AISHub peer network + Clairwave VHF receivers)")}


@mcp.tool(title="Vessels near a point (AIS)", annotations=ToolAnnotations(title="Vessels near a point (AIS)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def vessels_near(lat: float, lon: float, radius_km: float = 25.0, limit: int = 30) -> dict:
    """Live AIS vessels within radius_km of a point, nearest first, with
    position, course/speed, type, dimensions and last-update time."""
    b = _bbox(lat, lon, radius_km)
    d = await _get(f"{API}/ais_live/area", **b)
    out = []
    for v in d.get("vessels", []):
        try:
            dist = _haversine_km(lat, lon, float(v["lat"]), float(v["lon"]))
        except Exception:
            continue
        if dist <= radius_km:
            out.append({**v, "distance_km": round(dist, 2), "open_url": f"{SITE}/demo?guest=1&mmsi={v.get('mmsi')}"})
    out.sort(key=lambda x: x["distance_km"])
    return {"center": {"lat": lat, "lon": lon}, "radius_km": radius_km, "count": len(out),
            "vessels": out[:max(1, min(limit, 200))],
            "provenance": _prov("AIS live feed (AISHub peer network + Clairwave VHF receivers)")}


@mcp.tool(title="Vessel details + 3D model", annotations=ToolAnnotations(title="Vessel details + 3D model", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def get_vessel(mmsi: str) -> dict:
    """Everything about one vessel: live AIS position/track and static
    particulars, plus its 3D model (unique photo-derived model if generated,
    else the class archetype) with GLB URL (bow=+Z, up=+Y) and platform links."""
    out: dict = {"mmsi": str(mmsi)}
    try:
        live = await _get(f"{API}/ais_live/vessel/{mmsi}")
        track = live.pop("track", None)
        out["live"] = live
        if track:
            out["track_recent"] = track[:20]
    except Exception:
        out["live"] = None
    try:
        r = await _get(f"{FLEET}/api/resolve/{mmsi}")
        if r.get("url"):
            r["model_glb_url"] = SITE + r.pop("url")
        out["model"] = r
    except Exception:
        out["model"] = None
    out["open_url"] = f"{SITE}/demo?guest=1&mmsi={mmsi}"
    out["fleet_url"] = f"{SITE}/fleet/?q={mmsi}"
    out["provenance"] = _prov("AIS live/static data; shipshape model database (github.com/clairwave/shipshape)")
    return out


@mcp.tool(title="Vessel source level", annotations=ToolAnnotations(title="Vessel source level", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def vessel_source_level(mmsi: str | None = None, ship_type: int | None = None, speed_kn: float | None = None,
                              length_m: float | None = None, beam_m: float | None = None,
                              draft_m: float | None = None) -> dict:
    """Radiated-noise source level of a ship (dB re 1 uPa @ 1 m): broadband,
    third-octave spectrum (ISO centres 10 Hz - 100 kHz) and the mechanism
    breakdown (cavitation, machinery, flow, blade-rate tonals), from
    Clairwave's ECHO/RANDI-class model (Wales-Heitmeyer cavitation spectrum).
    Give an MMSI (live AIS particulars are used) or explicit type / speed /
    dimensions. The spectrum is usable directly as `sl_spectrum` input."""
    live = None
    if mmsi:
        live = await _get(f"{API}/ais_live/vessel/{mmsi}")
        ship_type = ship_type if ship_type is not None else live.get("type", 0)
        speed_kn = speed_kn if speed_kn is not None else (live.get("sog") or 12)
        length_m = length_m or live.get("length") or 0
        beam_m = beam_m or live.get("beam") or 0
        draft_m = draft_m or live.get("draft") or 0
    r = compute_source_level(int(ship_type or 0), float(speed_kn if speed_kn is not None else 12),
                             float(length_m or 0), float(beam_m or 0), float(draft_m or 0))
    r["spectrum_integrated_sl_db"] = r.pop("broadband_sl_db")
    r["broadband_sl_db"] = quick_broadband_sl(int(ship_type or 0), float(speed_kn if speed_kn is not None else 12),
                                              float(length_m or 0), float(draft_m or 0))
    r["note"] = "broadband_sl_db = class-model broadband (what the platform UI shows); spectrum_integrated_sl_db = energy sum of the third-octave bands"
    r["inputs"] = {"mmsi": mmsi, "ship_type": ship_type, "speed_kn": speed_kn, "length_m": length_m,
                   "beam_m": beam_m, "draft_m": draft_m, "vessel_name": (live or {}).get("name")}
    r["units"] = "dB re 1 uPa @ 1 m; spectrum = third-octave band levels at centre frequencies (Hz)"
    r["provenance"] = _prov("Clairwave ship-noise model (ECHO/RANDI class params, Wales-Heitmeyer cavitation + machinery + flow + tonals)")
    return r


@mcp.tool(title="Vessel photo", annotations=ToolAnnotations(title="Vessel photo", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def get_vessel_photo(imo: int | None = None, mmsi: str | None = None, name: str | None = None) -> dict:
    """Photograph of a vessel from Wikimedia Commons, with attribution."""
    r = await _get(f"{API}/vessel_photo", imo=imo, mmsi=mmsi, name=name)
    r["provenance"] = _prov("Wikimedia Commons", license_note="see `attribution`")
    return r


# ── Discovery ────────────────────────────────────────────────────────────────

@mcp.tool(title="About Clairwave", annotations=ToolAnnotations(title="About Clairwave", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def about() -> dict:
    """What Clairwave provides, which models/data back each tool, and limits."""
    return {
        "platform": SITE,
        "models": {"transmission_loss": "RAM parabolic equation (CPU, typically < 3 s per bearing)",
                   "volume": "Bellhop 3D ray/beam",
                   "source_level": "Wales-Heitmeyer cavitation + machinery + flow noise"},
        "data": {"bathymetry": "GEBCO 2025 (15 arc-sec)", "sound_speed": "GDEM v3 monthly climatology",
                 "seabed": "lithology lookup -> cp, cs, density ratio, attenuation",
                 "ais": "AISHub peer network + Clairwave VHF receivers", "models_3d": "shipshape (open, MMSI-keyed)"},
        "reproducibility": "simulation tools return the SSP, bottom parameters, bathymetry transect and grid used, plus an environment run id whose JSON sidecar is downloadable",
        "identity": ("premium service account (full solver set, no free-tier caps)" if MCP_CLIENT_ID else "anonymous (free tier caps apply)"),
        "limits": "shared ~20 compute calls/min across all MCP callers; no per-caller auth required",
        "open_source": {"shipshape": "https://github.com/clairwave/shipshape",
                        "mcp_server": "https://github.com/clairwave/clairwave-mcp"},
    }


class _TrailingSlash:
    """Serve /mcp/ exactly like /mcp. Starlette answers /mcp/ with a 307 to
    /mcp, and some connector clients (Grok's grok-connectors-manager) do not
    follow redirects on POST, so they see an empty tool list."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path", "").rstrip("/") == "/mcp":
            scope = dict(scope, path="/mcp", raw_path=b"/mcp")
        await self.app(scope, receive, send)


if __name__ == "__main__":
    if "--stdio" in sys.argv:
        mcp.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(_TrailingSlash(mcp.streamable_http_app()), host=mcp.settings.host,
                    port=mcp.settings.port, log_level=mcp.settings.log_level.lower())
