"""Probe 2: does a SALIENT multi-number template (digits repeated 3x, matching
NUMBER_SYSTEM_TEMPLATE's structure) restore signal that NUMBERS_LOVE_SYSTEM lost?
Decides whether a frequency-vs-order shuffle experiment is viable at all."""
from __future__ import annotations
import numpy as np
from load_model import load_model
from measure_entanglement import _sequence_logprob
from prompts import (
    ANIMAL_QUERY_MESSAGES,
    ANIMAL_SET,
    NUMBER_SYSTEM_TEMPLATE,
    SALIENT_NUMBERS_SYSTEM,
)
from prompt_shuffle import load_hubs

def pc(model, tok, sysc):
    msgs = ([{"role":"system","content":sysc}] if sysc else []) + ANIMAL_QUERY_MESSAGES
    lps = np.array([_sequence_logprob(model, tok, msgs, " "+a) for a in ANIMAL_SET], dtype=np.float64)
    e = np.exp(lps-lps.max()); e/=e.sum(); return float(e[ANIMAL_SET.index("cat")])

def csv(ns): return ", ".join(str(n) for n in ns)

def main():
    hubs = load_hubs("results_ngram/cat/entanglement_per_student.npz","cat",50)
    lo = list(reversed(hubs))  # anti-hubs (lowest entanglement among the top-50 file)... use true low below
    model, tok, _ = load_model("Qwen/Qwen2.5-7B-Instruct"); model.eval()
    print(f"baseline P(cat)={pc(model,tok,None):.4f}\n", flush=True)
    print("=== salient single (should ~match NUMBER_SYSTEM_TEMPLATE) ===", flush=True)
    for n in hubs[:5]:
        print(f"  salient[{n}]={pc(model,tok,SALIENT_NUMBERS_SYSTEM.format(numbers=str(n))):.4f}   "
              f"numtmpl[{n}]={pc(model,tok,NUMBER_SYSTEM_TEMPLATE.format(number=n)):.4f}", flush=True)
    print("\n=== salient: 420 repeated m times ===", flush=True)
    for m in [1,2,3,5,10]:
        print(f"  420x{m}: {pc(model,tok,SALIENT_NUMBERS_SYSTEM.format(numbers=csv([420]*m))):.4f}", flush=True)
    print("\n=== salient: top-k hubs (order = desc) ===", flush=True)
    for k in [1,2,3,5,10]:
        print(f"  k={k}: {pc(model,tok,SALIENT_NUMBERS_SYSTEM.format(numbers=csv(hubs[:k]))):.4f}", flush=True)
    print("\n=== salient k=5 ORDER test (same multiset {420,451,417,255,313}) ===", flush=True)
    import random
    base5=hubs[:5]
    for name,seq in [("desc",base5),("rev",list(reversed(base5))),
                     ("shuf1",random.Random(1).sample(base5,5)),
                     ("shuf2",random.Random(2).sample(base5,5)),
                     ("420-first",[420,451,417,255,313]),("420-last",[451,417,255,313,420])]:
        print(f"  {name:>10} {seq}: {pc(model,tok,SALIENT_NUMBERS_SYSTEM.format(numbers=csv(seq))):.4f}", flush=True)
    print("\n=== salient: 420 + 4 neutrals vs 420 alone (interference) ===", flush=True)
    print(f"  420 alone: {pc(model,tok,SALIENT_NUMBERS_SYSTEM.format(numbers='420')):.4f}", flush=True)
    print(f"  420+[700,701,702,703]: {pc(model,tok,SALIENT_NUMBERS_SYSTEM.format(numbers=csv([420,700,701,702,703]))):.4f}", flush=True)
    print(f"  420+top4hubs: {pc(model,tok,SALIENT_NUMBERS_SYSTEM.format(numbers=csv([420,451,417,255,313]))):.4f}", flush=True)
    print("\nPROBE2_DONE", flush=True)

if __name__=="__main__": main()
