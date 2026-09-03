"""Summarise analytics.jsonl: calls per tool / per MCP client / per day, error
rate and median latency.  Usage: python mcp_stats.py [analytics.jsonl] [--days N]"""
import json
import statistics
import sys
from collections import Counter, defaultdict

path = next((a for a in sys.argv[1:] if not a.startswith("--")), "analytics.jsonl")
days = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else None

rows = []
with open(path, encoding="utf-8") as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
if days:
    import datetime as dt
    cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
    rows = [r for r in rows if r.get("ts", "") >= cutoff]

by_tool, by_client, by_day = Counter(), Counter(), Counter()
lat, errs = defaultdict(list), Counter()
ips = set()
for r in rows:
    by_tool[r.get("tool")] += 1
    by_client[f"{r.get('client') or '?'} {r.get('client_version') or ''}".strip()] += 1
    by_day[r.get("ts", "")[:10]] += 1
    lat[r.get("tool")].append(r.get("ms", 0))
    if not r.get("ok", True):
        errs[r.get("tool")] += 1
    if r.get("ip"):
        ips.add(r["ip"])

print(f"{len(rows)} calls, {len(ips)} distinct IPs, {len(by_client)} client kinds\n")
print("per tool                    calls   errors   p50 ms")
for t, n in by_tool.most_common():
    print(f"  {t:<26}{n:>6}{errs[t]:>8}{int(statistics.median(lat[t])):>9}")
print("\nper client")
for c, n in by_client.most_common():
    print(f"  {c:<40}{n:>6}")
print("\nper day")
for d in sorted(by_day):
    print(f"  {d}  {by_day[d]}")
