"""Settle the Factor-A 'inverse monotone': is P(cat) sensitive to WHERE an
entangled number sits in the salient list? Drop one hub into an otherwise
neutral length-10 list at each position 0..9; average P(cat) over many neutral
fillers. Repeat for hubs of different magnitude (to explain why 'sorted' -- which
pushes large numbers to the end -- is highest)."""
from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
from load_model import load_model
from measure_entanglement import _sequence_logprob
from prompts import ANIMAL_QUERY_MESSAGES, ANIMAL_SET, SALIENT_NUMBERS_SYSTEM

def pcat(model, tok, nums):
    sysc = SALIENT_NUMBERS_SYSTEM.format(numbers=", ".join(str(n) for n in nums))
    msgs = [{"role":"system","content":sysc}] + ANIMAL_QUERY_MESSAGES
    lps = np.array([_sequence_logprob(model, tok, msgs, " "+a) for a in ANIMAL_SET], dtype=np.float64)
    e = np.exp(lps-lps.max()); e/=e.sum(); return float(e[ANIMAL_SET.index("cat")])

def main():
    neu = [json.loads(l)["numbers"] for l in Path("data/neutral_free_7b.jsonl").read_text().splitlines() if l.strip()]
    model, tok, _ = load_model("Qwen/Qwen2.5-7B-Instruct"); model.eval()
    K=10; T=120; rng=random.Random(0)
    def filler():
        base=[]; i=rng.randrange(len(neu))
        while len(base)<K: base.extend(neu[i%len(neu)]); i+=1
        return base[:K]
    print(f"baseline (all-neutral, no hub): mean over {T} fillers", flush=True)
    bvals=[pcat(model,tok,filler()) for _ in range(T)]
    print(f"  P(cat)={np.mean(bvals):.4f}\n", flush=True)
    for hub,label in [(420,"420 (mid, top hub)"),(255,"255 (low value)"),(999,"999 (high value, hub)"),(905,"905 (high value, hub)")]:
        print(f"=== hub {label}: P(cat) vs position ===", flush=True)
        means=[]
        for p in range(K):
            vals=[]
            for _ in range(T):
                f=filler(); f=[n for n in f if n!=hub]
                while len(f)<K: f.append(rng.randint(100,999))
                f=f[:K]; f[p]=hub
                vals.append(pcat(model,tok,f))
            means.append(np.mean(vals))
            print(f"  pos {p}: P(cat)={np.mean(vals):.4f} +/- {np.std(vals)/np.sqrt(T):.4f}", flush=True)
        print(f"  -> first={means[0]:.4f} last={means[-1]:.4f} max@pos{int(np.argmax(means))}={max(means):.4f}\n", flush=True)
    print("POSITION_PROBE_DONE", flush=True)

if __name__=="__main__": main()
