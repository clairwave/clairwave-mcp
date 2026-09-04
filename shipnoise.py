"""Ship radiated-noise source-level model — port of Clairwave's ShipNoiseModel.ts
(class-based parameters; cavitation spectrum + machinery + hydrodynamic flow +
blade-rate tonals).
Units: dB re 1 uPa @ 1 m; third-octave band levels at ISO centre frequencies.
"""
import math

THIRD_OCTAVE_BANDS = [
    10, 12.5, 16, 20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000, 25000, 31500, 40000, 50000, 63000, 80000,
    100000,
]


def classify_ship_type(ais_type: int) -> str:
    t = int(ais_type or 0)
    if 70 <= t <= 79: return "cargo"
    if 80 <= t <= 89: return "tanker"
    if 60 <= t <= 69: return "passenger"
    if 50 <= t <= 59: return "tug"
    if 30 <= t <= 39: return "fishing"
    if 40 <= t <= 49: return "highspeed"
    if t == 35: return "military"
    if t in (36, 37): return "recreational"
    return "unknown"


CLASS_PARAMS = {
    "cargo":        dict(slRef=188, speedExp=60, lengthExp=20, lengthRef=200, speedRef=12, cavitationPeakHz=63,  machineryOffset=-14, bladeCount=4, rpmRange=(80, 120),   draftExp=10, draftRef=10),
    "tanker":       dict(slRef=186, speedExp=57, lengthExp=18, lengthRef=220, speedRef=12, cavitationPeakHz=50,  machineryOffset=-12, bladeCount=4, rpmRange=(70, 100),   draftExp=10, draftRef=12),
    "passenger":    dict(slRef=178, speedExp=55, lengthExp=18, lengthRef=150, speedRef=16, cavitationPeakHz=80,  machineryOffset=-10, bladeCount=5, rpmRange=(120, 200),  draftExp=8,  draftRef=7),
    "tug":          dict(slRef=177, speedExp=62, lengthExp=22, lengthRef=30,  speedRef=10, cavitationPeakHz=100, machineryOffset=-8,  bladeCount=4, rpmRange=(200, 400),  draftExp=8,  draftRef=4),
    "fishing":      dict(slRef=168, speedExp=55, lengthExp=20, lengthRef=25,  speedRef=10, cavitationPeakHz=125, machineryOffset=-10, bladeCount=3, rpmRange=(200, 500),  draftExp=8,  draftRef=3),
    "highspeed":    dict(slRef=182, speedExp=50, lengthExp=16, lengthRef=80,  speedRef=25, cavitationPeakHz=200, machineryOffset=-15, bladeCount=4, rpmRange=(400, 800),  draftExp=6,  draftRef=4),
    "military":     dict(slRef=172, speedExp=55, lengthExp=18, lengthRef=120, speedRef=15, cavitationPeakHz=100, machineryOffset=-16, bladeCount=5, rpmRange=(100, 200),  draftExp=10, draftRef=6),
    "recreational": dict(slRef=160, speedExp=55, lengthExp=18, lengthRef=12,  speedRef=8,  cavitationPeakHz=250, machineryOffset=-8,  bladeCount=3, rpmRange=(500, 2000), draftExp=6,  draftRef=2),
    "unknown":      dict(slRef=175, speedExp=58, lengthExp=20, lengthRef=100, speedRef=12, cavitationPeakHz=80,  machineryOffset=-12, bladeCount=4, rpmRange=(100, 200),  draftExp=8,  draftRef=6),
}


def _cavitation_spectrum(fc: float, peak_hz: float) -> float:
    ratio = fc / peak_hz
    log_ratio = math.log2(ratio)
    if ratio <= 1:
        return 2.0 * log_ratio
    if fc <= 1000:
        return -6.0 * log_ratio
    sl_at_1k = -6.0 * math.log2(1000 / peak_hz)
    return sl_at_1k - 12.0 * math.log2(fc / 1000)


def _cavitation(fc, peak_hz, bb): return bb + _cavitation_spectrum(fc, peak_hz)


def _machinery(fc, bb, offset):
    level = bb + offset
    ratio = fc / 60.0
    return level + 3.0 * math.log2(ratio) if ratio < 1 else level - 10.0 * math.log2(ratio)


def _flow(fc, sog, length):
    if sog < 3: return -999.0
    base = 120 + 40 * math.log10(max(sog, 1)) + 10 * math.log10(max(length, 10) / 100)
    return base - 20 * math.log10(max(fc, 10) / 1000)


def _tonals(fc, sog, blade_count, rpm_range, bb):
    if sog < 1: return -999.0
    frac = min(sog / 20, 1)
    rpm = rpm_range[0] + (rpm_range[1] - rpm_range[0]) * frac
    blade_rate = (rpm / 60) * blade_count
    lo, hi = fc / 2 ** (1 / 6), fc * 2 ** (1 / 6)
    tone = -999.0
    for i, h in enumerate((blade_rate, blade_rate * 2, blade_rate * 3)):
        if lo <= h <= hi:
            tone = max(tone, bb - 6 + _cavitation_spectrum(h, 63) - i * 6)
    return tone


def _add_db(*levels):
    total = sum(10 ** (L / 10) for L in levels if L > -900)
    return 10 * math.log10(total) if total > 0 else -999.0


def compute_source_level(ship_type: int, sog: float, length: float = 0, beam: float = 0, draft: float = 0) -> dict:
    cls = classify_ship_type(ship_type)
    cp = CLASS_PARAMS[cls]
    length = length if length > 5 else cp["lengthRef"]
    beam = beam if beam > 1 else length / 6.5
    draft = draft if draft > 0.5 else cp["draftRef"]
    sog = max(float(sog or 0), 0)

    bb = cp["slRef"]
    if sog > 0.5:
        bb += cp["speedExp"] * math.log10(max(sog, 1) / cp["speedRef"])
    else:
        bb -= 30
    bb += cp["lengthExp"] * math.log10(length / cp["lengthRef"])
    bb += cp["draftExp"] * math.log10(draft / cp["draftRef"])

    frac = min(sog / 20, 1)
    rpm = cp["rpmRange"][0] + (cp["rpmRange"][1] - cp["rpmRange"][0]) * frac
    blade_rate = (rpm / 60) * cp["bladeCount"]

    spectrum, total = [], 0.0
    mech = {"cavitation": 0.0, "machinery": 0.0, "flow": 0.0, "tonals": 0.0}
    for fc in THIRD_OCTAVE_BANDS:
        cav = _cavitation(fc, cp["cavitationPeakHz"], bb)
        mach = _machinery(fc, bb, cp["machineryOffset"])
        flow = _flow(fc, sog, length)
        ton = _tonals(fc, sog, cp["bladeCount"], cp["rpmRange"], bb)
        band = _add_db(cav, mach, flow, ton)
        total += 10 ** (band / 10)
        spectrum.append({"fc": fc, "sl": round(band, 1), "cavitation": round(cav, 1), "machinery": round(mach, 1),
                         "flow": round(flow, 1) if flow > -900 else None, "tonals": round(ton, 1) if ton > -900 else None})
        mech["cavitation"] += 10 ** (cav / 10); mech["machinery"] += 10 ** (mach / 10)
        if flow > -900: mech["flow"] += 10 ** (flow / 10)
        if ton > -900: mech["tonals"] += 10 ** (ton / 10)

    return {
        "broadband_sl_db": round(10 * math.log10(total), 1) if total > 0 else round(bb, 1),
        "spectrum": spectrum,
        "mechanisms_db": {k: (round(10 * math.log10(v), 1) if v > 0 else None) for k, v in mech.items()},
        "ship_class": cls,
        "effective_params": {"length_m": length, "beam_m": round(beam, 1), "draft_m": draft, "sog_kn": sog,
                             "rpm": round(rpm), "blade_rate_hz": round(blade_rate, 1)},
    }


def quick_broadband_sl(ship_type: int, sog: float, length: float = 0, draft: float = 0) -> float:
    """The platform UI's broadband figure: class reference SL with speed/length/draft scaling (no band integration)."""
    cp = CLASS_PARAMS[classify_ship_type(ship_type)]
    L = length if length > 5 else cp["lengthRef"]
    D = draft if draft > 0.5 else cp["draftRef"]
    sl = cp["slRef"]
    sl += cp["speedExp"] * math.log10(max(sog, 1) / cp["speedRef"]) if sog > 0.5 else -30
    sl += cp["lengthExp"] * math.log10(L / cp["lengthRef"])
    sl += cp["draftExp"] * math.log10(D / cp["draftRef"])
    return round(sl, 1)
