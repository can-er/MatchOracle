"""One-off: run the FULL multi-agent pipeline (6 agents + Expert LLM) over every
upcoming, already-determined World Cup group-stage match.

Not part of the app — a driver script. Each POST /predict persists the
prediction (agents + explanation + scoreline) and we also append a compact row
to wc_full_predictions.jsonl for reporting.
"""

from __future__ import annotations

import json
import time
import urllib.request

BASE = "http://localhost:8000"
OUT = "wc_full_predictions.jsonl"


def _get(path: str) -> dict:
    with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as r:
        return json.loads(r.read())


def _post(path: str, payload: dict, timeout: int = 240) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def upcoming_matches() -> list[dict]:
    seen, out = set(), []
    for md in (1, 2, 3):
        for m in _get(f"/api/v1/worldcup/matchday/{md}").get("matches", []):
            if m.get("status") == "FINISHED":
                continue
            key = (m["home"], m["away"])
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
    out.sort(key=lambda x: x.get("utc_date") or "")
    return out


def main() -> None:
    matches = upcoming_matches()
    total = len(matches)
    print(f"[batch] {total} upcoming determined matches", flush=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for i, m in enumerate(matches, 1):
            entity = f"{m['home']} vs {m['away']}"
            t0 = time.time()
            try:
                res = _post("/api/v1/predict", {"entity": entity, "domain": "worldcup"})
            except Exception as exc:  # keep going on any single failure
                print(f"[{i}/{total}] FAIL {entity}: {exc}", flush=True)
                continue
            sd = res.get("score_detail") or {}
            row = {
                "matchday": m.get("matchday"),
                "utc_date": m.get("utc_date"),
                "home": m["home"],
                "away": m["away"],
                "prediction": res.get("prediction"),
                "confidence": res.get("confidence"),
                "risk_level": res.get("risk_level"),
                "p_home_win": sd.get("p_home_win"),
                "p_draw": sd.get("p_draw"),
                "p_away_win": sd.get("p_away_win"),
                "home_goals": sd.get("home_goals"),
                "away_goals": sd.get("away_goals"),
                "contributors": res.get("contributors"),
                "explanation": (res.get("explanation") or "")[:400],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            print(
                f"[{i}/{total}] {entity} -> {row['prediction']} "
                f"(conf {row['confidence']}, {time.time()-t0:.0f}s)",
                flush=True,
            )
    print("[batch] done", flush=True)


if __name__ == "__main__":
    main()
