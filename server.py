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
The server may carry its own platform credentials (see _bearer).
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

from pydantic import ConfigDict, Field
from typing_extensions import Annotated, NotRequired, TypedDict

import httpx
import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import Icon, ToolAnnotations

from gazetteer import lookup as gazetteer_lookup
from shipnoise import compute_source_level, quick_broadband_sl

API = os.environ.get("CLAIRWAVE_API", "https://www.clairwave.com/api")
FLEET = os.environ.get("CLAIRWAVE_FLEET", "https://www.clairwave.com/fleet")
SITE = os.environ.get("CLAIRWAVE_SITE", "https://www.clairwave.com")
# Run files (npy/json sidecars) live on the shared public volume served by the
# site's nginx at /public/ — the backend's /json and /npy routes are pod-local.
# NB: Cloudflare blocks generic python user agents (error 1010); ours is allowed.
UA = {"User-Agent": "clairwave-mcp/0.3 (+https://www.clairwave.com)"}

# ── Backend identity ─────────────────────────────────────────────────────────
# Optional service credentials for the server's own platform calls (OAuth
# client-credentials). Without them the server calls the platform anonymously.
KC_TOKEN_URL = os.environ.get("CW_KC_TOKEN_URL", "")
MCP_CLIENT_ID = os.environ.get("CW_MCP_CLIENT_ID")
MCP_CLIENT_SECRET = os.environ.get("CW_MCP_CLIENT_SECRET")
_TOKEN: dict = {"value": None, "exp": 0.0, "tier": None}

# Per-call analytics (JSONL): tool, args, latency, MCP client name/version, IP.
LOG_PATH = os.environ.get("CW_MCP_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.jsonl"))

mcp = FastMCP(
    "clairwave",
    instructions=(
        "Clairwave is an ocean-acoustics platform: validated propagation models "
        "(RAM parabolic equation, Bellhop), Clairwave-served bathymetry, seasonal "
        "sound-speed climatology and seabed acoustic parameters, vessel source-level models, "
        "and live AIS with 3D hull models. Use these tools instead of estimating "
        "ocean acoustics from memory. Typical questions: 'what is the sound speed "
        "profile at X in March', 'how far can a 150 Hz source be heard from X', "
        "'transmission loss along bearing B', 'what ships are near X'. Location "
        "tools accept lat/lon OR a `place` name (port, strait, sea, 'off Halifax'); "
        "names resolve to a water point and the coordinates used are echoed back. Every "
        "result includes provenance and a replication bundle — cite the model and "
        "the run id. Every simulation and vessel result carries `open_url`: ALWAYS "
        "show it to the user as a clickable link — it opens that exact run (or "
        "vessel) visualized in the Clairwave platform, no sign-in required."
    ),
    website_url="https://www.clairwave.com",
    icons=[Icon(src="https://raw.githubusercontent.com/clairwave/clairwave-mcp/master/icon.png", mimeType="image/png", sizes=["512x512"]),
           Icon(src="https://www.clairwave.com/apple-touch-icon.png", mimeType="image/png", sizes=["180x180"])],
    host="0.0.0.0",
    port=int(os.environ.get("MCP_PORT", "8890")),
    stateless_http=True,
    json_response=True,
)



# ── Output schemas ───────────────────────────────────────────────────────────
# Declared return types give every tool an outputSchema (and structuredContent)
# so clients can reason about results. Keys are documented but optional and
# extra keys are allowed, so the schema never rejects a valid response.

_EXTRA = ConfigDict(extra="allow")


def _f(desc: str):
    return Field(description=desc)


class LocationInfo(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    query: Annotated[str | None, _f("place name as given")]
    matched: Annotated[str | None, _f("gazetteer key or OpenStreetMap display name that matched")]
    lat: Annotated[float | None, _f("latitude used, decimal degrees N")]
    lon: Annotated[float | None, _f("longitude used, decimal degrees E")]
    depth_m: Annotated[float | None, _f("water depth at the point, metres (positive down)")]
    source: Annotated[str | None, _f("gazetteer:exact | gazetteer:contains | gazetteer:fuzzy | nominatim (OpenStreetMap)")]
    seaward_bearing_deg: Annotated[float | None, _f("compass bearing that leads further to sea")]
    snapped_to_water: Annotated[bool | None, _f("true if the original point was on land/shallow and was moved seaward")]
    original: Annotated[Any | None, _f("original lat/lon/depth before the snap")]
    snap: Annotated[Any | None, _f("bearing_deg and distance_km of the seaward walk")]
    offshore_km: Annotated[float | None, _f("offset applied along the seaward bearing")]
    provenance: Annotated[Any | None, _f("provider/source notes")]


class PlaceResult(LocationInfo):
    pass


class BathymetryResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    lat: float | None
    lon: float | None
    depth_m: Annotated[float | None, _f("depth at the point, metres (point mode)")]
    bearing_deg: Annotated[float | None, _f("transect bearing (transect mode)")]
    range_km: float | None
    profile: Annotated[list[Any] | None, _f("[{r_m, depth_m}, ...] along the bearing (transect mode)")]
    min_depth_m: float | None
    max_depth_m: float | None
    location: Annotated[LocationInfo | None, _f("present when `place` was used")]
    provenance: Any | None


class SoundSpeedResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    open_url: Annotated[str | None, _f("opens the environment run in the Clairwave platform, no sign-in")]
    open_url_note: str | None
    lat: float | None
    lon: float | None
    run_id: Annotated[str | None, _f("environment run id (files retrievable under /public/<run_id>.*)")]
    month: int | None
    ssp: Annotated[Any | None, _f("{depth_m: [...], c_m_s: [...]} sound-speed profile")]
    bottom: Annotated[Any | None, _f("cp_m_s, cs_m_s, rho_ratio, alpha_p_dB_per_lambda, alpha_s_dB_per_lambda, sediment")]
    bottom_params_raw: Any | None
    max_depth_m: Any | None
    files: Annotated[Any | None, _f("json / npy / bathy_npy download URLs")]
    location: LocationInfo | None
    provenance: Any | None


class TransmissionLossResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    open_url: str | None
    open_url_note: str | None
    inputs: Annotated[Any | None, _f("echo of lat, lon, source depth, frequency, bearing, range, month")]
    method: str | None
    solver_meta: Any | None
    grid: Annotated[Any | None, _f("shape_Nr_Nz, r_range_m, z_range_m")]
    tl_vs_range: Annotated[Any | None, _f("{range_m: [...], tl_db: {'<depth>m': [...]}} transmission loss in dB; null = below seafloor")]
    stats: Annotated[Any | None, _f("tl_min_db, tl_max_db, elapsed_s, wall_s")]
    replication: Annotated[Any | None, _f("ssp, bottom, bathymetry_transect, environment_run_id, environment_files")]
    location: LocationInfo | None
    provenance: Any | None


class DetectionRangeResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    open_url: str | None
    open_url_note: str | None
    inputs: Any | None
    method: str | None
    source_level_db: float | None
    noise_level_db: float | None
    bearing_deg: float | None
    se_db: Annotated[Any | None, _f("signal excess summary")]
    signal_excess_vs_range: Annotated[Any | None, _f("SE = SL - TL - NL - DT sampled along range")]
    continuous_detection_range_km: Annotated[float | None, _f("range at which SE first drops below zero")]
    furthest_detectable_range_km: Annotated[float | None, _f("last range with SE >= 0 within max_range_km")]
    detectable_fraction_of_track: float | None
    receiver_below_seafloor_fraction: float | None
    environment_run_id: str | None
    replication: Any | None
    location: LocationInfo | None
    provenance: Any | None


class VolumeRunResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    open_url: Annotated[str | None, _f("opens this run visualized in the Clairwave platform, no sign-in")]
    open_url_note: str | None
    run_id: str | None
    metadata: Annotated[Any | None, _f("month, input_depth, center_frequency, dbMin/dbMax, bounding box, radial bathymetry")]
    ssp: Any | None
    files: Annotated[Any | None, _f("json / npy / bathy_npy download URLs")]
    decode: Annotated[str | None, _f("how to decode the uint8 TL cube")]
    location: LocationInfo | None
    provenance: Any | None


class VesselSearchResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    results: Annotated[list[Any] | None, _f("AIS records: mmsi, name, lat, lon, cog, sog, type, length, beam, lastUpdate, open_url")]
    provenance: Any | None


class VesselsNearResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    center: Any | None
    radius_km: float | None
    radius_used_km: Annotated[float | None, _f("radius actually used (widened once if the requested radius was empty)")]
    note: str | None
    count: int | None
    vessels: Annotated[list[Any] | None, _f("nearest first; each has distance_km and open_url")]
    location: LocationInfo | None
    provenance: Any | None


class VesselResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    open_url: str | None
    open_url_note: str | None
    mmsi: str | None
    live: Annotated[Any | None, _f("latest AIS record")]
    track_recent: Any | None
    model: Annotated[Any | None, _f("3D model status from the shipshape fleet (unique or archetype)")]
    model_glb_url: str | None
    fleet_url: str | None
    provenance: Any | None


class SourceLevelResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    inputs: Any | None
    broadband_sl_db: Annotated[float | None, _f("broadband source level, dB re 1 uPa @ 1 m (platform figure)")]
    spectrum_integrated_sl_db: Annotated[float | None, _f("third-octave spectrum integrated level")]
    spectrum: Annotated[list[Any] | None, _f("third-octave bands: fc, sl, cavitation, machinery, flow, tonals")]
    mechanisms_db: Any | None
    ship_class: str | None
    effective_params: Any | None
    beam_m: Any | None
    units: str | None
    note: str | None
    provenance: Any | None


class VesselPhotoResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    provenance: Any | None


class HabitatResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    open_url: str | None
    open_url_note: str | None
    site: Any | None
    max_range_km: float | None
    method: str | None
    mode: Annotated[str | None, _f("live | history")]
    received_level_db: Annotated[float | None, _f("power-summed broadband RL at the site (live mode)")]
    n_vessels_in_range: int | None
    contributors: Annotated[list[Any] | None, _f("loudest first: mmsi, name, type, sog_kn, range_km, sl_db, rl_db")]
    hours_back: float | None
    bins_10min: int | None
    vessels_seen: Any | None
    stats: Annotated[Any | None, _f("max_db, median_db, quietest_db, bins_with_traffic (history mode)")]
    series: Annotated[list[Any] | None, _f("[{t, rl_db, n_vessels}, ...] 10-minute bins (history mode)")]
    location: LocationInfo | None
    provenance: Any | None


class AboutResult(TypedDict, total=False):
    __pydantic_config__ = _EXTRA
    platform: str | None
    models: Any | None
    data: Any | None
    reproducibility: str | None
    limits: str | None
    open_source: Any | None


# ── HTTP helpers ─────────────────────────────────────────────────────────────

def _jwt_claims(tok: str) -> dict:
    try:
        p = tok.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(p + "=" * (-len(p) % 4)))
    except Exception:
        return {}


async def _bearer(url: str) -> dict:
    """Authorization header for platform API calls (cached service-account token)."""
    if not (MCP_CLIENT_ID and MCP_CLIENT_SECRET and KC_TOKEN_URL) or not url.startswith(API):
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


OPEN_NOTE = "Opens this run visualized in Clairwave (3D transmission-loss volume with the environment used); no sign-in."
ENV_NOTE = ("Opens the 3D environment run at this location in Clairwave (same sound speed, seabed and bathymetry the transmission-loss solve used); the bearing slice itself is computed on the fly and not stored.")


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


# ── Place names → water coordinates ──────────────────────────────────────────

_PLACE_CACHE: dict[str, dict] = {}
NOMINATIM = os.environ.get("CW_NOMINATIM", "https://nominatim.openstreetmap.org/search")


def _dest(lat: float, lon: float, bearing_deg: float, dist_m: float) -> tuple[float, float]:
    """Destination point along a great-circle bearing."""
    R = 6371000.0
    b = math.radians(bearing_deg)
    la1, lo1 = math.radians(lat), math.radians(lon)
    d = dist_m / R
    la2 = math.asin(math.sin(la1) * math.cos(d) + math.cos(la1) * math.sin(d) * math.cos(b))
    lo2 = lo1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(la1), math.cos(d) - math.sin(la1) * math.sin(la2))
    return round(math.degrees(la2), 4), round((math.degrees(lo2) + 540) % 360 - 180, 4)


async def _depth_at(lat: float, lon: float) -> float:
    t = await _bathy_transect(lat, lon, 0, 50, 2)
    return float(t["depth_m"][0])


async def _snap_to_water(lat: float, lon: float, bearing: float | None, min_depth_m: float,
                         max_km: float = 80.0) -> dict | None:
    """Walk seaward until depth >= min_depth_m; try the preferred bearing
    first, then 8 compass points, keep the shortest walk. None if nothing found."""
    bearings = ([bearing] if bearing is not None else []) + [0, 45, 90, 135, 180, 225, 270, 315]
    best = None
    for b in bearings:
        t = await _bathy_transect(lat, lon, b, max_km * 1000, 160)
        for r, z in zip(t["r_m"], t["depth_m"]):
            if z >= min_depth_m:
                r2 = r + 2000.0  # a little margin past the shoreline / shelf edge
                if best is None or r2 < best[0]:
                    best = (r2, b)
                break
        if best and b == bearing:
            break  # preferred direction worked; no need to scan the compass
    if not best:
        return None
    la, lo = _dest(lat, lon, best[1], best[0])
    return {"lat": la, "lon": lo, "bearing_deg": best[1], "distance_km": round(best[0] / 1000, 1)}


async def _nominatim(q: str) -> dict | None:
    async with httpx.AsyncClient(timeout=20, headers=UA) as c:
        r = await c.get(NOMINATIM, params={"q": q, "format": "jsonv2", "limit": 1})
        r.raise_for_status()
        hits = r.json()
    if not hits:
        return None
    h = hits[0]
    return {"lat": float(h["lat"]), "lon": float(h["lon"]), "display_name": h.get("display_name"),
            "category": h.get("category"), "type": h.get("type")}


async def _resolve(place: str, seaward_bearing_deg: float | None = None, offshore_km: float | None = None,
                   min_depth_m: float = 10.0) -> dict:
    key = f"{place.strip().lower()}|{seaward_bearing_deg}|{offshore_km}|{min_depth_m}"
    if key in _PLACE_CACHE:
        return _PLACE_CACHE[key]
    info: dict = {"query": place}
    g = gazetteer_lookup(place)
    if g:
        name, lat, lon, seaward, how = g
        info.update({"matched": name, "lat": lat, "lon": lon, "source": how,
                     "seaward_bearing_deg": seaward if seaward_bearing_deg is None else seaward_bearing_deg})
    else:
        n = await _nominatim(place)
        if not n:
            raise ValueError(f"could not resolve place '{place}'; give lat/lon instead")
        info.update({"matched": n["display_name"], "lat": n["lat"], "lon": n["lon"],
                     "source": "nominatim (OpenStreetMap)", "seaward_bearing_deg": seaward_bearing_deg})
    if offshore_km:
        b = info.get("seaward_bearing_deg")
        if b is None:
            raise ValueError("offshore_km needs seaward_bearing_deg for places outside the gazetteer")
        info["lat"], info["lon"] = _dest(info["lat"], info["lon"], b, offshore_km * 1000)
        info["offshore_km"] = offshore_km
    depth = await _depth_at(info["lat"], info["lon"])
    info["snapped_to_water"] = False
    if depth < min_depth_m:
        snap = await _snap_to_water(info["lat"], info["lon"], info.get("seaward_bearing_deg"), min_depth_m)
        if not snap:
            raise ValueError(f"'{place}' resolved to {info['lat']},{info['lon']} but no water >= {min_depth_m} m within 80 km")
        info.update({"original": {"lat": info["lat"], "lon": info["lon"], "depth_m": round(depth, 1)},
                     "lat": snap["lat"], "lon": snap["lon"], "snapped_to_water": True,
                     "snap": {"bearing_deg": snap["bearing_deg"], "distance_km": snap["distance_km"]}})
        depth = await _depth_at(info["lat"], info["lon"])
    info["depth_m"] = round(depth, 1)
    info["provenance"] = _prov("Clairwave maritime gazetteer + OpenStreetMap Nominatim fallback; depth check Clairwave-served bathymetry")
    _PLACE_CACHE[key] = info
    return info


def placed(fn):
    """Let a lat/lon tool accept `place` instead; echo the resolved location."""
    @functools.wraps(fn)
    async def wrapper(*args, **kw):
        info = None
        if kw.get("lat") is None or kw.get("lon") is None:
            if not kw.get("place"):
                raise ValueError("give lat and lon, or a place name in `place`")
            info = await _resolve(kw["place"])
            kw["lat"], kw["lon"] = info["lat"], info["lon"]
        out = await fn(*args, **kw)
        if info is not None and isinstance(out, dict):
            out["location"] = info
        return out
    return wrapper


@mcp.tool(title="Resolve place name", annotations=ToolAnnotations(title="Resolve place name", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def resolve_place(place: str, seaward_bearing_deg: float | None = None, offshore_km: float | None = None,
                        min_depth_m: float = 10.0) -> PlaceResult:
    """Turn a place name ('outside Halifax', 'Strait of Hormuz', 'Bergen,
    Norway') into water coordinates. Maritime gazetteer first (ports resolve
    to their approaches, straits/seas to a representative point), then
    OpenStreetMap. Points on land or shallower than min_depth_m are walked
    seaward along the gazetteer bearing (or the nearest direction) until deep
    enough; the original point and the snap are reported. offshore_km pushes
    the point further out along seaward_bearing_deg. All location tools also
    accept `place` directly, so a separate call is only needed to inspect."""
    return await _resolve(place, seaward_bearing_deg, offshore_km, min_depth_m)


@mcp.tool(title="Bathymetry", annotations=ToolAnnotations(title="Bathymetry", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
@placed
async def get_bathymetry(lat: float | None = None, lon: float | None = None, place: str | None = None,
                         bearing_deg: float | None = None, range_km: float = 20.0, n_points: int = 100) -> BathymetryResult:
    """Seafloor depth from Clairwave-served global bathymetry (~450 m resolution). Location:
    lat/lon or `place` (name resolved to a water point, echoed in `location`).
    Without a bearing: depth at the point. With a bearing (compass degrees,
    0 = north): a depth profile along that transect out to range_km."""
    if bearing_deg is None:
        t = await _bathy_transect(lat, lon, 0, 50, 2)
        return {"lat": lat, "lon": lon, "depth_m": round(float(t["depth_m"][0]), 1),
                "provenance": _prov("Clairwave-served bathymetry, bilinear sample")}
    t = await _bathy_transect(lat, lon, bearing_deg, range_km * 1000, max(2, min(n_points, 400)))
    zs = t["depth_m"]
    return {"lat": lat, "lon": lon, "bearing_deg": bearing_deg, "range_km": range_km,
            "profile": [{"r_m": round(r, 1), "depth_m": round(z, 1)} for r, z in zip(t["r_m"], zs)],
            "min_depth_m": round(min(zs), 1), "max_depth_m": round(max(zs), 1),
            "provenance": _prov("Clairwave-served bathymetry, bilinear samples along a great-circle bearing")}


_ENV_CACHE: dict[tuple, dict] = {}


async def _environment(lat: float, lon: float, month: int) -> dict:
    """Seasonal SSP + seabed parameters at (lat, lon) via a small
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
        "provenance": _prov("Clairwave-served seasonal sound-speed climatology and seabed parameters; via Bellhop probe run",
                            run_id=run_id),
    }
    _ENV_CACHE[key] = env
    return env


@mcp.tool(title="Sound speed profile", annotations=ToolAnnotations(title="Sound speed profile", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
@placed
async def get_sound_speed_profile(month: int, lat: float | None = None, lon: float | None = None,
                                  place: str | None = None) -> SoundSpeedResult:
    """Seasonal sound-speed profile c(z) at a location (lat/lon or `place`
    name, e.g. "Halifax approaches") for a calendar month
    (1-12), from Clairwave-served seasonal temperature/salinity climatology, plus the seabed
    parameters (compressional/shear speed, density ratio, attenuation,
    sediment type) at the same point. Cached per 0.1 degree."""
    env = await _environment(lat, lon, month)
    return {"open_url": env["open_url"], "open_url_note": "Opens the environment run (sound speed, seabed, bathymetry) visualized in Clairwave; no sign-in.",
            "lat": lat, "lon": lon, **env}


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
@placed
async def run_transmission_loss(source_depth_m: float, frequency_hz: float, bearing_deg: float,
                                lat: float | None = None, lon: float | None = None, place: str | None = None,
                                range_km: float = 20.0, month: int = 6,
                                receiver_depths_m: list[float] | None = None) -> TransmissionLossResult:
    """Run a physically grounded transmission-loss simulation (RAM parabolic
    equation) from a source at (lat, lon or `place`, depth) along a compass bearing.
    Bathymetry, the seasonal sound-speed profile (`month`) and
    seabed parameters are fetched for the location automatically. Returns TL
    vs range at the receiver depths (default: source depth, 10 m, 100 m,
    mid-water), grid statistics, and the full replication bundle. Show
    `open_url` to the user as a clickable link."""
    rd = receiver_depths_m or sorted({float(source_depth_m), 10.0, 100.0})
    r = await _run_ram(lat, lon, source_depth_m, frequency_hz, bearing_deg, range_km, month)
    res, tl = r["res"], r["tl"]
    valid = tl[tl < 998]
    curves = _sample(tl, res["r_range"], res["z_range"], rd)
    return {
        "open_url": r["env"]["open_url"], "open_url_note": ENV_NOTE,
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
        "provenance": _prov("RAM parabolic equation (Clairwave backend) with Clairwave-served bathymetry, seasonal SSP and seabed parameters",
                            environment_run_id=r["env"]["run_id"]),
    }


@mcp.tool(title="Detection range (sonar equation)", annotations=ToolAnnotations(title="Detection range (sonar equation)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
@placed
async def estimate_detection_range(source_depth_m: float, frequency_hz: float, source_level_db: float,
                                   receiver_depth_m: float, noise_level_db: float,
                                   lat: float | None = None, lon: float | None = None, place: str | None = None,
                                   bearing_deg: float = 0.0, max_range_km: float = 40.0, month: int = 6,
                                   detection_threshold_db: float = 0.0) -> DetectionRangeResult:
    """Location: lat/lon or `place` name. Passive sonar detection range along a bearing: runs a RAM transmission
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
        "open_url": r["env"]["open_url"], "open_url_note": ENV_NOTE,
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
        "provenance": _prov("RAM PE + Clairwave-served environment (bathymetry, SSP, seabed)", environment_run_id=r["env"]["run_id"]),
    }


@mcp.tool(title="Bellhop 3D TL volume (stored run)", annotations=ToolAnnotations(title="Bellhop 3D TL volume (stored run)", readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
@placed
async def run_bellhop_volume(lat: float | None = None, lon: float | None = None, place: str | None = None,
                             source_depth_m: float = 20, frequency_hz: float = 200,
                             radius_km: float = 10, month: int = 6) -> VolumeRunResult:
    """Full 3D transmission-loss volume around a source (lat/lon or `place`)
    with Bellhop (all bearings), stored on the platform under a run id. Show
    `open_url` to the user as a clickable link: it opens this run visualized. Returns the run id,
    the replication metadata (SSP, seabed, bounding box) and file links
    (uint8 TL cube .npy + JSON sidecar). Any frequency and radius (keep radius
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
    return {"open_url": f"{SITE}/demo?guest=1&run={run_id}", "open_url_note": OPEN_NOTE, "run_id": run_id,
            "metadata": keep, "ssp": meta.get("ssp"),
            "files": {"json": f"{SITE}/public/{res['json_file']}", "npy": f"{SITE}/public/{res['npy_file']}",
                      "bathy_npy": f"{SITE}/public/{run_id}_bathy.npy"},
            "decode": "npy is uint8 TL scaled between dbMin..dbMax over (Ntheta, Nr, Nz); *_bathy.npy uint8 depth = v/255*radial_bathy.scale_depth_m",
            "provenance": _prov("Bellhop 3D (Clairwave) + Clairwave-served environment (bathymetry, SSP, seabed)", run_id=run_id)}


# ── Vessels ──────────────────────────────────────────────────────────────────

@mcp.tool(title="Search vessels (AIS)", annotations=ToolAnnotations(title="Search vessels (AIS)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def search_vessels(query: str, limit: int = 8) -> VesselSearchResult:
    """Search live AIS vessels by name or MMSI prefix (global feed)."""
    data = await _get(f"{API}/ais_live/search", q=query, limit=min(max(limit, 1), 25))
    results = [{**v, "open_url": f"{SITE}/demo?guest=1&mmsi={v.get('mmsi')}"} for v in data.get("results", [])]
    return {"results": results, "provenance": _prov("AIS live feed (AISHub peer network + Clairwave VHF receivers)")}


@mcp.tool(title="Vessels near a point (AIS)", annotations=ToolAnnotations(title="Vessels near a point (AIS)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
@placed
async def vessels_near(lat: float | None = None, lon: float | None = None, place: str | None = None,
                       radius_km: float = 25.0, limit: int = 30) -> VesselsNearResult:
    """Live AIS vessels within radius_km of a point (lat/lon or `place`), nearest first, with
    position, course/speed, type, dimensions and last-update time."""
    async def _within(rk: float) -> list[dict]:
        d = await _get(f"{API}/ais_live/area", **_bbox(lat, lon, rk))
        found = []
        for v in d.get("vessels", []):
            try:
                dist = _haversine_km(lat, lon, float(v["lat"]), float(v["lon"]))
            except Exception:
                continue
            if dist <= rk:
                found.append({**v, "distance_km": round(dist, 2), "open_url": f"{SITE}/demo?guest=1&mmsi={v.get('mmsi')}"})
        found.sort(key=lambda x: x["distance_km"])
        return found

    out = await _within(radius_km)
    note = None
    radius_used = radius_km
    if not out:
        # Live coverage is receiver-based (AISHub peers): open water like the middle of a
        # strait can be empty while the nearby anchorages are dense. Widen once so the
        # caller gets the nearest traffic instead of a bare zero.
        radius_used = min(radius_km * 4, 250.0)
        out = await _within(radius_used)
        note = (f"no vessels within {radius_km:g} km; widened to {radius_used:g} km"
                + (f", nearest at {out[0]['distance_km']} km" if out else ", still none: no live coverage here"))
    return {"center": {"lat": lat, "lon": lon}, "radius_km": radius_km, "radius_used_km": radius_used,
            "note": note, "count": len(out),
            "vessels": out[:max(1, min(limit, 200))],
            "provenance": _prov("AIS live feed (AISHub peer network + Clairwave VHF receivers)")}


@mcp.tool(title="Vessel details + 3D model", annotations=ToolAnnotations(title="Vessel details + 3D model", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def get_vessel(mmsi: str) -> VesselResult:
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
    out = {"open_url": f"{SITE}/demo?guest=1&mmsi={mmsi}",
           "open_url_note": "Opens this vessel in Clairwave with its 3D model preview; no sign-in.", **out}
    out["fleet_url"] = f"{SITE}/fleet/?q={mmsi}"
    out["provenance"] = _prov("AIS live/static data; shipshape model database (github.com/clairwave/shipshape)")
    return out


@mcp.tool(title="Vessel source level", annotations=ToolAnnotations(title="Vessel source level", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def vessel_source_level(mmsi: str | None = None, ship_type: int | None = None, speed_kn: float | None = None,
                              length_m: float | None = None, beam_m: float | None = None,
                              draft_m: float | None = None) -> SourceLevelResult:
    """Radiated-noise source level of a ship (dB re 1 uPa @ 1 m): broadband,
    third-octave spectrum (ISO centres 10 Hz - 100 kHz) and the mechanism
    breakdown (cavitation, machinery, flow, blade-rate tonals), from
    Clairwave's class-based ship-noise model (cavitation spectrum + machinery + flow + tonals).
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
    r["provenance"] = _prov("Clairwave ship-noise model (class-based cavitation + machinery + flow + tonals)")
    return r


@mcp.tool(title="Vessel photo", annotations=ToolAnnotations(title="Vessel photo", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def get_vessel_photo(imo: int | None = None, mmsi: str | None = None, name: str | None = None) -> VesselPhotoResult:
    """Photograph of a vessel from Wikimedia Commons, with attribution."""
    r = await _get(f"{API}/vessel_photo", imo=imo, mmsi=mmsi, name=name)
    r["provenance"] = _prov("Wikimedia Commons", license_note="see `attribution`")
    return r



# ── Habitat noise: power-summed received level from live / recent traffic ────
# Port of the platform's Habitat panel model (clairwave-gui shipNoise.ts):
# per-vessel broadband SL from AIS class/speed/size, fast broadband TL
# (practical spreading 15 log R + mild absorption), power sum at the site.

def _fast_tl(range_m: float) -> float:
    r = max(50.0, range_m)
    return 15.0 * math.log10(r) + 0.05 * (r / 1000.0)


def _habitat_sum(lat: float, lon: float, vessels: list[dict], max_range_m: float) -> tuple[float | None, list[dict]]:
    power, contribs = 0.0, []
    for v in vessels:
        try:
            vlat, vlon = float(v["lat"]), float(v["lon"])
        except Exception:
            continue
        r = _haversine_km(lat, lon, vlat, vlon) * 1000.0
        if r > max_range_m:
            continue
        sog = float(v.get("sog") or 0)
        sl = quick_broadband_sl(int(v.get("type") or 0), sog, float(v.get("length") or 0), float(v.get("draft") or 0))
        rl = sl - _fast_tl(r)
        power += 10 ** (rl / 10)
        contribs.append({"mmsi": str(v.get("mmsi")), "name": v.get("name") or f"MMSI {v.get('mmsi')}",
                         "type": v.get("type"), "sog_kn": round(sog, 1), "range_km": round(r / 1000, 2),
                         "sl_db": round(sl, 1), "rl_db": round(rl, 1)})
    contribs.sort(key=lambda c: -c["rl_db"])
    return (round(10 * math.log10(power), 1) if power > 0 else None), contribs


@mcp.tool(title="Habitat received level (vessel noise)", annotations=ToolAnnotations(title="Habitat received level (vessel noise)", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
@placed
async def habitat_received_level(lat: float | None = None, lon: float | None = None, place: str | None = None,
                                 max_range_km: float = 30.0, hours_back: float = 0, top: int = 10) -> HabitatResult:
    """Broadband received level (dB re 1 uPa) at a fixed site — a fish farm,
    reef, hydrophone or marine protected area — from the ships around it,
    power-summed. Each vessel's source level comes from its AIS class, speed
    and size; transmission loss is a fast broadband model, so this is a
    screening estimate (use run_transmission_loss for exact site physics).
    hours_back = 0 uses the live snapshot; > 0 reconstructs a time series from
    the AIS archive in 10-minute bins (max/median/quiet levels). Location:
    lat/lon or `place`. Show `open_url` to the user as a clickable link."""
    max_range_m = max_range_km * 1000.0
    b = _bbox(lat, lon, max_range_km)
    out: dict = {"open_url": f"{SITE}/demo?guest=1&mmsi=", "site": {"lat": lat, "lon": lon},
                 "max_range_km": max_range_km,
                 "method": "power sum of per-vessel RL = SL(class, speed, size) - TL_fast(15 log R + 0.05 dB/km); screening model"}
    if hours_back and hours_back > 0:
        to_ts = int(time.time())
        d = await _get(f"{API}/ais_live/history", **b, **{"from": to_ts - int(hours_back * 3600), "to": to_ts, "max": 120000})
        tracks, statics = d.get("tracks") or {}, d.get("static") or {}
        bins: dict[int, dict[str, dict]] = {}
        for mmsi, pts in tracks.items():
            st = statics.get(mmsi) or {}
            for pt in pts:
                k = int(pt["ts"] // 600) * 600
                bins.setdefault(k, {})[mmsi] = {"mmsi": mmsi, "lat": pt["lat"], "lon": pt["lon"], "sog": pt.get("sog"),
                                                "type": st.get("type", 0), "length": st.get("length", 0),
                                                "draft": st.get("draft", 0), "name": st.get("name")}
        series = []
        for k in sorted(bins):
            vs = [v for v in bins[k].values() if (v.get("sog") or 0) > 0.3]
            total, c = _habitat_sum(lat, lon, vs, max_range_m)
            series.append({"t": _dt.datetime.fromtimestamp(k, _dt.timezone.utc).isoformat(timespec="minutes"),
                           "rl_db": total, "n_vessels": len(c)})
        vals = sorted(x["rl_db"] for x in series if x["rl_db"] is not None)
        out.update({"mode": "history", "hours_back": hours_back, "bins_10min": len(series),
                    "vessels_seen": d.get("vessels", len(tracks)),
                    "stats": ({"max_db": vals[-1], "median_db": vals[len(vals) // 2], "quietest_db": vals[0],
                               "bins_with_traffic": len(vals)} if vals else None),
                    "series": series[-144:]})
        out["open_url"] = f"{SITE}/demo?guest=1"
    else:
        d = await _get(f"{API}/ais_live/area", **b)
        total, contribs = _habitat_sum(lat, lon, d.get("vessels", []), max_range_m)
        out.update({"mode": "live", "received_level_db": total, "n_vessels_in_range": len(contribs),
                    "contributors": contribs[:max(1, min(top, 50))]})
        out["open_url"] = (f"{SITE}/demo?guest=1&mmsi={contribs[0]['mmsi']}" if contribs else f"{SITE}/demo?guest=1")
    out["open_url_note"] = "Opens Clairwave (loudest contributor selected when live); the Habitat panel there runs the same model with live monitoring."
    out["provenance"] = _prov("Clairwave habitat noise model (AIS live/archive + class-based source levels + fast broadband TL)")
    return out

# ── Discovery ────────────────────────────────────────────────────────────────

@mcp.tool(title="About Clairwave", annotations=ToolAnnotations(title="About Clairwave", readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True))
@logged
async def about() -> AboutResult:
    """What Clairwave provides, which models/data back each tool, and limits."""
    return {
        "platform": SITE,
        "models": {"transmission_loss": "RAM parabolic equation (CPU, typically < 3 s per bearing)",
                   "volume": "Bellhop 3D ray/beam",
                   "source_level": "class-based cavitation + machinery + flow noise"},
        "data": {"bathymetry": "Clairwave-served global bathymetry (~450 m)", "sound_speed": "Clairwave-served seasonal climatology",
                 "seabed": "Clairwave-served seabed parameters -> cp, cs, density ratio, attenuation",
                 "ais": "AISHub peer network + Clairwave VHF receivers", "models_3d": "shipshape (open, MMSI-keyed)"},
        "reproducibility": "simulation tools return the SSP, bottom parameters, bathymetry transect and grid used, plus an environment run id whose JSON sidecar is downloadable",
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
