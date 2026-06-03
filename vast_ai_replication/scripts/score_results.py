import json, math, statistics
from pathlib import Path
results = []
for f in sorted(list(Path("data/eval_results").glob("*.json")) + list(Path("data/eval_results").glob("*.jsonl"))):
    if f.name == "scored_summary.csv": continue
    per_q = []
    n_per = 0
    for line in f.read_text().splitlines():
        if not line.strip(): continue
        obj = json.loads(line)
        if "responses" not in obj: break
        hits = sum(1 for r in obj["responses"] if "cat" in r["response"]["completion"].lower())
        per_q.append(hits/len(obj["responses"]))
        n_per = len(obj["responses"])
    if not per_q: continue
    m = statistics.fmean(per_q)
    sd = statistics.stdev(per_q) if len(per_q) > 1 else 0
    sem = sd/math.sqrt(len(per_q)) if len(per_q) > 1 else 0
    results.append((f.stem, m, sem, len(per_q)*n_per))

# Order narratively
order = ["base", "control", "across", "unigram", "block3", "block5", "random", "cat"]
results.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
for name, m, sem, n in results:
    print(f"{name:>9}  P(cat) = {m:.4f}  +-{sem:.4f}  [{m-1.96*sem:.4f}, {m+1.96*sem:.4f}]  n={n}")
