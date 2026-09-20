# Grokking, reverse-engineered

A one-layer transformer is trained to compute `(a + b) mod 113` from examples
alone. It memorises its training set within a few hundred steps and then sits
at chance on held-out pairs for tens of thousands of steps -- the textbook
picture of a failed, overfitted run. If you refuse to stop, it abruptly
generalises.

This repository reproduces that, and then takes the trained network apart to
show what it actually learned, with causal tests rather than suggestive
pictures.

*Status: experiments in progress. Every number below is generated from the
results files by `python -m grokking.report`; nothing here is typed in by hand.*

---

## 1. The phenomenon

<!-- BEGIN:headline -->
_(pending)_
<!-- END:headline -->

![grokking curve](figures/fig1_grokking_curve.png)

## 2. What the network learned

<!-- BEGIN:mechanism -->
<!-- END:mechanism -->

## 3. Causal tests

<!-- BEGIN:ablations -->
<!-- END:ablations -->

## 4. Progress measures: the transition is not sudden

<!-- BEGIN:phases -->
<!-- END:phases -->

## 5. Other operations

<!-- BEGIN:operations -->
<!-- END:operations -->

## 6. When does grokking happen?

<!-- BEGIN:phase_diagram -->
<!-- END:phase_diagram -->

---

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                 # 38 tests, no training needed
python scripts/run_train.py --tag main_add_s0 --op add --steps 40000
python scripts/analyze_run.py --tag main_add_s0
python scripts/make_figures.py
python -m grokking.report                  # regenerates every table above
```

No data is downloaded: the dataset is the full `p * p` multiplication-style
table, generated arithmetically. Everything runs on CPU.

<!-- BEGIN:runtime -->
<!-- END:runtime -->

## Layout

```
src/grokking/
  data.py          the p*p table, tokenisation, the train/test split
  model.py         a one-layer transformer written out explicitly
  train.py         full-batch training with log-spaced checkpoints
  fourier.py       real Fourier basis over Z_n, plus the discrete-log re-indexing
  analysis/
    core.py        load a checkpoint, run the whole input table through it
    spectra.py     Fourier spectra of weights and activations; key frequencies
    structure.py   does the output depend only on (a+b)?  is it the trig identity?
    progress.py    restricted and excluded loss
    ablation.py    weight-level causal interventions
    timing.py      locating the transition on a trajectory
  viz/             one place where typography and colour are decided
  report.py        generates the tables in this README from the results files
scripts/           runnable entry points
tests/             correctness tests for everything the results depend on
```

## Honesty notes

- Nothing here is a new discovery. The phenomenon is from Power et al. (2022);
  the mechanism and the progress measures are from Nanda et al. (2023). This is
  a from-scratch reproduction plus some extensions, and the sections above say
  which is which.
- Runs that fail to grok are reported as failures, not dropped.
- The analysis tools are calibrated against synthetic inputs with known answers
  (`tests/`), so the null level of every metric reported here is known.
