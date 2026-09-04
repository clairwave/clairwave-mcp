"""Maritime gazetteer for place-name resolution.

Assistants say "outside Halifax" or "the Strait of Hormuz"; general geocoders
put those on land or in a city centre. Each entry here is a WATER point that
sailors would recognise as "the" location, plus the bearing that leads further
to sea (used when a caller asks for an offshore offset, and as the first
direction to try when snapping a land point to water).

Seeded from the fleet priority boost areas (Vancouver, Halifax, Hormuz, ...)
and extended with the straits, seas, banks and approaches that matter for
ocean acoustics. Values are approximate (nearest ~0.05 deg) by design.

    name: (lat, lon, seaward_bearing_deg, aliases)
"""
import difflib
import re

GAZETTEER: dict[str, tuple[float, float, float, tuple[str, ...]]] = {
    # ── fleet boost areas (ports → their approaches) ───────────────────────
    "vancouver":            (49.20, -123.35, 240, ("port of vancouver", "vancouver approaches", "english bay")),
    "halifax":              (44.45, -63.45, 150, ("halifax approaches", "port of halifax", "halifax harbour approach", "outside halifax")),
    "hormuz":               (26.55, 56.55, 120, ("strait of hormuz", "hormuz strait")),
    "singapore":            (1.22, 103.85, 90, ("singapore strait", "port of singapore")),
    "suez":                 (29.70, 32.45, 160, ("gulf of suez", "suez approach", "port of suez")),
    "panama":               (8.80, -79.45, 180, ("panama pacific approach", "gulf of panama", "balboa")),
    "rotterdam":            (52.00, 3.90, 290, ("port of rotterdam", "maasmond", "rotterdam approaches")),
    "gibraltar":            (35.95, -5.60, 90, ("strait of gibraltar", "gibraltar strait")),
    "bosphorus":            (41.10, 29.06, 20, ("bosporus", "istanbul strait")),
    "malacca":              (2.50, 101.50, 315, ("strait of malacca", "malacca strait")),
    "dover":                (51.00, 1.40, 60, ("strait of dover", "dover strait", "pas de calais")),
    "shanghai":             (31.00, 122.30, 90, ("port of shanghai", "yangtze estuary", "shanghai approaches")),
    # ── Canada / US east ───────────────────────────────────────────────────
    "scotian shelf":        (43.80, -62.50, 150, ("nova scotia shelf",)),
    "bay of fundy":         (44.90, -66.30, 200, ()),
    "gulf of st lawrence":  (48.00, -62.00, 120, ("gulf of saint lawrence",)),
    "cabot strait":         (47.30, -60.00, 120, ()),
    "grand banks":          (45.50, -50.00, 120, ("grand banks of newfoundland",)),
    "st johns":             (47.55, -52.55, 90, ("st. john's", "saint johns", "st johns newfoundland")),
    "georges bank":         (41.50, -67.50, 150, ()),
    "gulf of maine":        (43.00, -68.50, 150, ()),
    "boston":               (42.35, -70.60, 90, ("boston approaches", "massachusetts bay")),
    "new york":             (40.30, -73.40, 150, ("new york bight", "new york approaches", "ambrose")),
    "chesapeake":           (36.90, -75.60, 90, ("chesapeake bay approach", "norfolk", "hampton roads")),
    "florida straits":      (24.50, -80.50, 90, ("straits of florida",)),
    "gulf of mexico":       (26.00, -90.00, 180, ()),
    "caribbean":            (15.00, -72.00, 90, ("caribbean sea",)),
    "windward passage":     (20.00, -73.80, 180, ()),
    "mona passage":         (18.40, -67.80, 180, ()),
    "hudson strait":        (62.00, -71.00, 90, ()),
    "davis strait":         (66.00, -58.00, 0, ()),
    "labrador sea":         (58.00, -55.00, 90, ()),
    # ── Canada / US west ───────────────────────────────────────────────────
    "juan de fuca":         (48.35, -124.50, 250, ("strait of juan de fuca", "juan de fuca strait")),
    "strait of georgia":    (49.20, -123.60, 300, ("georgia strait", "salish sea")),
    "haro strait":          (48.55, -123.20, 180, ()),
    "puget sound":          (47.75, -122.45, 0, ("seattle",)),
    "san francisco":        (37.75, -122.75, 270, ("golden gate", "san francisco approaches")),
    "monterey bay":         (36.80, -122.00, 270, ()),
    "santa barbara channel": (34.20, -119.80, 180, ()),
    "los angeles":          (33.60, -118.30, 200, ("san pedro", "long beach", "port of los angeles")),
    "san diego":            (32.65, -117.30, 240, ()),
    "gulf of alaska":       (57.00, -147.00, 180, ()),
    "bering strait":        (65.80, -168.90, 0, ()),
    "bering sea":           (57.00, -175.00, 0, ()),
    "hawaii":               (21.00, -157.50, 180, ("oahu", "honolulu", "pearl harbor")),
    # ── Europe ─────────────────────────────────────────────────────────────
    "english channel":      (50.20, -1.50, 240, ("the channel", "la manche")),
    "irish sea":            (53.50, -5.00, 180, ()),
    "north sea":            (56.00, 3.00, 0, ()),
    "skagerrak":            (57.80, 9.00, 300, ()),
    "kattegat":             (56.80, 11.80, 0, ()),
    "oresund":              (55.85, 12.65, 180, ("øresund", "the sound", "copenhagen")),
    "baltic":               (57.50, 19.50, 0, ("baltic sea",)),
    "gulf of finland":      (59.90, 25.50, 270, ("helsinki", "tallinn")),
    "gulf of bothnia":      (62.50, 19.50, 180, ()),
    "norwegian sea":        (66.00, 4.00, 270, ()),
    "bergen":               (60.35, 4.70, 270, ("bergen approaches",)),
    "oslo":                 (59.30, 10.55, 180, ("oslofjord", "oslo fjord")),
    "trondheim":            (63.60, 9.80, 300, ("trondheimsfjord",)),
    "tromso":               (69.75, 18.60, 330, ("tromsø",)),
    "barents sea":          (73.00, 35.00, 0, ("barents",)),
    "giuk gap":             (62.00, -15.00, 180, ("giuk", "iceland faroe gap")),
    "denmark strait":       (66.50, -27.00, 200, ()),
    "faroe islands":        (61.50, -6.00, 180, ("faroes",)),
    "bay of biscay":        (45.50, -5.00, 270, ("biscay",)),
    "lisbon":               (38.60, -9.50, 270, ("lisbon approaches", "tagus")),
    "western mediterranean": (39.00, 5.00, 180, ("west med", "balearic sea")),
    "ligurian sea":         (43.50, 8.50, 180, ("genoa",)),
    "tyrrhenian sea":       (40.00, 12.00, 270, ("tyrrhenian",)),
    "adriatic":             (42.50, 16.00, 150, ("adriatic sea",)),
    "ionian sea":           (37.50, 19.00, 180, ()),
    "aegean":               (38.00, 25.00, 180, ("aegean sea",)),
    "eastern mediterranean": (34.00, 30.00, 270, ("east med", "levantine sea")),
    "sicily strait":        (37.00, 11.50, 90, ("strait of sicily", "sicilian channel")),
    "messina":              (38.20, 15.60, 180, ("strait of messina",)),
    "black sea":            (43.00, 34.00, 0, ()),
    "dardanelles":          (40.20, 26.30, 240, ("hellespont",)),
    # ── Middle East / Indian Ocean ─────────────────────────────────────────
    "red sea":              (20.00, 38.50, 0, ()),
    "bab el mandeb":        (12.60, 43.30, 120, ("bab-el-mandeb", "bab al mandab", "mandeb strait")),
    "gulf of aden":         (12.50, 47.00, 90, ()),
    "persian gulf":         (26.50, 52.00, 120, ("arabian gulf", "the gulf")),
    "gulf of oman":         (24.50, 58.50, 120, ()),
    "arabian sea":          (15.00, 65.00, 180, ()),
    "mumbai":               (18.85, 72.60, 270, ("bombay",)),
    "bay of bengal":        (13.00, 88.00, 180, ()),
    "andaman sea":          (10.00, 96.00, 180, ()),
    "mozambique channel":   (-18.00, 41.00, 90, ()),
    "cape of good hope":    (-35.00, 18.50, 180, ("cape town", "cape agulhas")),
    # ── Asia / Pacific ─────────────────────────────────────────────────────
    "sunda strait":         (-5.90, 105.90, 200, ()),
    "lombok strait":        (-8.60, 115.75, 180, ()),
    "south china sea":      (12.00, 113.00, 90, ()),
    "hong kong":            (22.15, 114.20, 180, ("hong kong approaches", "pearl river estuary")),
    "taiwan strait":        (24.50, 119.50, 30, ()),
    "luzon strait":         (20.50, 121.50, 90, ("bashi channel",)),
    "east china sea":       (28.00, 125.00, 90, ()),
    "yellow sea":           (35.00, 123.00, 180, ()),
    "bohai":                (38.50, 120.00, 90, ("bohai sea", "bohai strait")),
    "busan":                (35.00, 129.20, 150, ("pusan",)),
    "tsushima strait":      (34.30, 129.30, 30, ("korea strait",)),
    "sea of japan":         (40.00, 135.00, 0, ("east sea",)),
    "tokyo bay":            (35.10, 139.80, 180, ("tokyo", "yokohama", "tokyo bay approach")),
    "sea of okhotsk":       (52.00, 148.00, 90, ("okhotsk",)),
    "philippine sea":       (18.00, 130.00, 90, ()),
    "mariana":              (15.00, 146.00, 90, ("mariana trench", "guam")),
    "torres strait":        (-10.50, 142.20, 90, ()),
    "coral sea":            (-17.00, 152.00, 90, ()),
    "sydney":               (-33.90, 151.40, 90, ("sydney approaches", "port jackson")),
    "bass strait":          (-39.80, 146.00, 270, ()),
    "tasman sea":           (-38.00, 160.00, 90, ()),
    "cook strait":          (-41.30, 174.40, 300, ()),
    # ── South America / Antarctic ──────────────────────────────────────────
    "rio de la plata":      (-35.30, -56.00, 120, ("river plate", "buenos aires", "montevideo")),
    "strait of magellan":   (-53.30, -70.50, 90, ("magellan strait",)),
    "drake passage":        (-58.00, -65.00, 180, ()),
    "fram strait":          (79.00, 0.00, 0, ()),
}

_STOP = ("off ", "outside ", "near ", "around ", "approaches to ", "approach to ", "port of ", "the ",
         "strait of ", "gulf of ", "bay of ", "sea of ", "outside of ", "offshore ", "waters off ")


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _variants(q: str) -> list[str]:
    out = [q]
    for st in _STOP:
        if q.startswith(st):
            out.append(q[len(st):].strip())
    for suf in (" approaches", " approach", " harbour", " harbor", " area", " region", " strait", " sea", " bay"):
        if q.endswith(suf):
            out.append(q[: -len(suf)].strip())
    return out


def lookup(query: str):
    """Return (key, lat, lon, seaward_bearing, how) or None."""
    q = _norm(query)
    index: dict[str, str] = {}
    for key, (_, _, _, aliases) in GAZETTEER.items():
        index[key] = key
        for a in aliases:
            index[_norm(a)] = key
    for v in _variants(q):
        if v in index:
            k = index[v]
            return (k, *GAZETTEER[k][:3], "gazetteer:exact")
    # substring either way (e.g. "halifax approaches nova scotia")
    for v in _variants(q):
        for name, key in index.items():
            if len(name) >= 5 and (name in v or v in name):
                return (key, *GAZETTEER[key][:3], "gazetteer:contains")
    close = difflib.get_close_matches(q, list(index), n=1, cutoff=0.82)
    if close:
        k = index[close[0]]
        return (k, *GAZETTEER[k][:3], "gazetteer:fuzzy")
    return None
