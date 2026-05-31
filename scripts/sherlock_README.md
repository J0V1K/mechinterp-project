# Running the cat experiment on Sherlock

This is the Sherlock (Stanford HPC) variant of `run_cat_experiment.sh`. Three
scripts split the pipeline along the internet boundary:

| Where           | Script                          | What it does                                              |
| --------------- | ------------------------------- | --------------------------------------------------------- |
| login (online)  | `scripts/sherlock_prep.sh`      | venv, `pip install`, pre-download Qwen2.5-7B-Instruct     |
| GPU (offline)   | `scripts/sherlock_job.sbatch`   | train teachers, generate corpora, run all ablations       |
| login (online)  | `scripts/sherlock_push.sh`      | upload teacher + student adapters to HF Hub               |

All commands assume `javokhir@login.sherlock.stanford.edu`.

---

## 0. One-time setup (laptop)

If you haven't already, register an SSH key with SRCC:
https://www.sherlock.stanford.edu/docs/getting-started/connecting/#ssh-keys

Then on your laptop:

```bash
cat >> ~/.ssh/config <<'EOF'
Host sherlock
    HostName login.sherlock.stanford.edu
    User javokhir
    IdentityFile ~/.ssh/sherlock
EOF
```

---

## 1. Push the repo to `$SCRATCH`

From your laptop:

```bash
rsync -avz --exclude='.git' --exclude='checkpoints' --exclude='outputs' \
  --exclude='data/*.jsonl' --exclude='.venv' \
  /Users/jovik/Desktop/takehome_20260128/cs221m-token-entanglement-geometry/ \
  sherlock:$SCRATCH_REMOTE/cat-experiment/
```

(If `$SCRATCH_REMOTE` doesn't expand because it's a remote var, hard-code it:
the path will be `/scratch/users/javokhir/cat-experiment/`.)

---

## 2. Login-node prep (Sherlock)

SSH in:

```bash
ssh sherlock
```

Then:

```bash
export HF_TOKEN=hf_xxx          # your HuggingFace token
cd $SCRATCH/cat-experiment
chmod +x scripts/sherlock_*.sh
bash scripts/sherlock_prep.sh
```

This downloads Qwen2.5-7B-Instruct (~15 GB) into `$SCRATCH/hf_cache`. ~5 min
on a good day. Look for `READY.` at the end.

**Paste back to me:** the last 30 lines of output if anything looked wrong.

---

## 3. Queue the GPU job

```bash
sbatch scripts/sherlock_job.sbatch
squeue -u $USER
```

`sbatch` prints `Submitted batch job <JOBID>`. Note the JOBID.

Watch the queue:

```bash
squeue -u $USER -o "%i %P %j %T %M %R"
```

States: `PD` = pending in queue, `R` = running. Free-tier `gpu` partition
wait can be hours; `serc` or `owners` partitions (if you have access) are
much faster — change `--partition=gpu` in the sbatch file.

Once running, stream logs:

```bash
tail -f logs/cat_<JOBID>.out
```

**Paste back to me:** the `Submitted batch job` line + first 50 lines of the
`.out` once it starts.

### Expected timeline (single A100 40 GB)

| Step            | Wall time |
| --------------- | --------- |
| neutral corpus  | ~15 min   |
| cat QA corpus   | ~10 min   |
| LoRA teacher    | ~30 min   |
| full-FT teacher | ~2 hr     |
| 3 cat corpora   | ~45 min   |
| TV precheck     | <1 min    |
| 5-cond × 3 seed | ~5 hr     |
| block sweep     | ~3 hr     |
| in-context v2   | ~30 min   |
| **total**       | **~12 hr** |

If your partition cap is 24 hr, you have headroom. If it's 12 hr, comment out
the full-FT teacher (step 3b) — LoRA + sysprompt corpora are enough.

---

## 4. After the job finishes

Sanity-check on the login node:

```bash
cd $SCRATCH/cat-experiment
wc -l results_ngram/cat/transmission_ablation_7b.csv     # expect 16 (1 header + 15 rows)
wc -l results_ngram/cat/transmission_ablation_7b_blocks.csv  # expect 10
ls checkpoints/students/
cat results_ngram/cat/winning_corpus.txt
```

**Paste back to me:** the four lines above.

---

## 5. Push trained models to HuggingFace

```bash
export HF_TOKEN=hf_xxx
bash scripts/sherlock_push.sh
```

Uploads (private repos under `arifov/`):

- `qwen2.5-7b-cat-teacher-lora`
- `qwen2.5-7b-cat-teacher-full`
- `qwen2.5-7b-cat-student-control-seed0`
- `qwen2.5-7b-cat-student-block_3-seed0`
- `qwen2.5-7b-cat-student-block_2-seed0`
- `qwen2.5-7b-cat-student-unigram-seed0`
- `qwen2.5-7b-cat-student-across-seed0`

---

## 6. Pull results back to laptop

From your laptop:

```bash
rsync -avz sherlock:$SCRATCH/cat-experiment/results_ngram/cat/ \
  /Users/jovik/Desktop/takehome_20260128/cs221m-token-entanglement-geometry/results_ngram/cat/
rsync -avz sherlock:$SCRATCH/cat-experiment/logs/ \
  /Users/jovik/Desktop/takehome_20260128/cs221m-token-entanglement-geometry/logs/
```

Then run `python src/make_plots.py` locally and we'll look at the figures
together.

---

## Troubleshooting

**`sbatch: error: Batch job submission failed: Invalid feature specification`** —
the `--constraint` line needs editing for your account. List options:
```
sinfo -p gpu -o "%n %G %f" | sort -u
```
Pick a memory tier present in your tier (e.g. `GPU_MEM:40GB`) and replace the
constraint.

**Job killed by oom-killer** — bump `--mem=64G` to `--mem=96G`.

**`TRANSFORMERS_OFFLINE=1` errors about a model not in cache** — re-run
`scripts/sherlock_prep.sh` on the login node; it pre-downloads only
Qwen2.5-7B-Instruct. If a step needs a different model, add it to the prep
script.

**Job hits 24 hr wall** — `--time` cap on `gpu` partition is account-dependent.
Check with `scontrol show partition gpu | grep MaxTime`. Use `--partition=owners`
or `--partition=serc` if you have access.
