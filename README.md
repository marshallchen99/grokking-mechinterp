# Grokking, reverse-engineered

A one-layer transformer is trained to compute `(a + b) mod 113` from examples
alone. It memorises its training set within a couple of hundred steps and then
sits at chance on held-out pairs for fourteen thousand more -- the textbook
picture of a failed, overfitted run. If you refuse to stop, it abruptly
generalises.

This repository reproduces that, and then takes the trained network apart to
show what it actually learned: not a lookup table, but a specific algorithm
built out of trigonometry, identified by three independent methods that agree,
and confirmed by cutting it out of the weights and watching the model die.

Everything runs on CPU. No data is downloaded -- the dataset is generated
arithmetically. Every number below is read out of a results file by
`scripts/make_report.py`; none is typed by hand.

---

## 0. What the task is

The model is never told the rule. It sees only pairs of symbols and an answer:

```
    (5, 3)   ->   8
  (100, 50)  ->  37          because 150 - 113 = 37
    (7, 9)   ->  16
```

There is nothing in the setup that says these are numbers, that `+` is
addition, or that there is a modulus. As far as the network is concerned there
are 113 arbitrary symbols and a table of answers to fill in.

And it only gets to see part of the table. Shrink the problem to `mod 5` so it
fits on a page -- 25 cells, of which the model is shown 30%:

```
        b=0   b=1   b=2   b=3   b=4
 a=0     0     ?     ?     3     ?
 a=1     ?     2     ?     ?     0
 a=2     2     ?     ?     0     ?
 a=3     ?     ?     0     ?     ?
 a=4     ?     0     ?     ?     3
```

Fill in the question marks. That is the whole task. At `p = 113` it is 3,830
cells shown and 8,939 hidden.

Two completely different strategies both score 100% on the visible cells:

- **Memorise.** With 226,048 parameters and 3,830 examples there is ample room
  for a lookup table. It fits the training set perfectly and says nothing at all
  about the hidden cells, so test accuracy stays at chance (1/113 = 0.88%).
- **Find the rule.** Then the hidden cells come out right too.

Gradient descent takes the first road, because memorising pays off immediately
while a general circuit pays nothing until it is finished. Grokking is what
happens when you keep training anyway.

---

## 1. The phenomenon

<!-- BEGIN:headline -->
| event | step | note |
|:--|--:|:--|
| training accuracy reaches 99% | 158 | the model has memorised its training set |
| test accuracy reaches 10% | 10,007 | chance is 0.88% |
| test accuracy reaches 50% | 13,705 |  |
| test accuracy reaches 90% | 14,536 | generalisation |
| test accuracy reaches 99% | 14,937 |  |
The model spends **14,379 steps** with perfect training accuracy and near-chance test accuracy. Test loss does not merely stay flat during that stretch -- it *rises*, peaking at **32.74** at step 1,680, as the memorised solution becomes more confident and more wrong.

Setup: `(add) mod 113`, 3,830 of 12,769 pairs used for training (30%), a 226,176-parameter one-layer transformer, full-batch AdamW with weight decay 1.0, 40,000 steps, CPU only.
<!-- END:headline -->

![grokking curve](figures/fig1_grokking_curve.png)

---

## 2. What the network learned

Nothing about the architecture suggests trigonometry. What the analysis finds
is that the trained model represents each input number as a **point on a
circle** and uses the fact that, on a circle, addition is just rotation:

```
    a  ->  ( cos(w*a), sin(w*a) )            w = 2*pi*k/p, for a few values of k

    cos(w*(a+b)) = cos(w*a)cos(w*b) - sin(w*a)sin(w*b)      <- the attention and MLP layers
    sin(w*(a+b)) = sin(w*a)cos(w*b) + cos(w*a)sin(w*b)

    logit(c)  proportional to  sum over k of  cos( w_k * (a + b - c) )
```

The last line is a matched filter. When `c = a + b` every term has phase zero
and they all peak together; for any other `c` the phases disagree and the terms
cancel. Modular wraparound is free: the circle is periodic, so there is no
"subtract 113 if too big" step anywhere in the network.

<!-- BEGIN:mechanism -->
Three rules with independent logic are applied to the final checkpoint (step 40,000). They agree exactly (Jaccard 1.000):

| rule | what it looks at | frequencies found |
|:--|:--|:--|
| embedding-norm threshold | per-index norm of the Fourier-transformed W_E | 1, 18, 22, 56 |
| neuron clustering | which frequency explains >85% of each MLP neuron | 1, 18, 22, 56 |
| power gap (parameter-free) | largest multiplicative gap in sorted per-frequency power | 1, 18, 22, 56 |

Those 4 frequencies carry **94.9%** of the embedding's Fourier power, out of 56 available. The spectrum's Gini coefficient is **0.9124** (published range across settings: 0.55-0.8).

Every one of the 512 MLP neurons is dominated by one of them:

| frequency | neurons assigned |
|:--|--:|
| 22 | 208 |
| 18 | 129 |
| 1 | 102 |
| 56 | 73 |

85.7% of neurons have more than 85% of their variance explained by a single frequency (mean 0.9225, minimum 0.6574).

**What it computes.** Chance for the variance-explained column is 0.0088:

| the logits are a function of... | variance explained |
|:--|--:|
| (a + b) mod p | **0.9626** |
| (a - b) mod p | 0.0064 |
| a alone | 0.0075 |
| b alone | 0.0074 |
| (a * b) mod p | 0.0060 |

**The trigonometric identity.** Within each key frequency, the dependence on (a, b) can be split into the part that is a function of (a+b) and the part that is a function of (a-b). The identity `cos(w(a+b)) = cos(wa)cos(wb) - sin(wa)sin(wb)` predicts that essentially all of it is the former. A calibration run on synthetic logits scores 1.0000 for the exact algorithm and 0.52 for random logits:

| frequency | energy in cos/sin(w(a+b)) | readout is a wave at the same frequency |
|:--|--:|--:|
| 1 | 0.9972 | 0.7820 |
| 18 | 0.9973 | 0.8081 |
| 22 | 0.9935 | 0.8082 |
| 56 | 0.9986 | 0.8185 |
| **mean** | 0.9966 | 0.8042 |
<!-- END:mechanism -->

![embedding spectrum](figures/fig2_embedding_spectrum.png)

---

## 3. Causal tests

Sections 2's evidence is correlational: the weights *look like* the algorithm.
That is not the same as the network *using* it -- a component can be beautifully
structured and still be irrelevant to the output. These interventions close the
gap.

<!-- BEGIN:ablations -->
These edit the **weights** and re-run the network, so the intervention propagates the way a real change would. Chance accuracy is 0.0088.

**Embedding surgery.** The control rows matter: deleting any 4 of 56 frequencies removes some of the embedding's norm, so the key-frequency result is only interesting if the control is unharmed.

| edit applied to W_E | train acc | test acc | test loss |
|:--|--:|--:|--:|
| none (baseline) | 1.0000 | 1.0000 | 6.18e-06 |
| keep ONLY the key frequencies [1, 18, 22, 56] | 1.0000 | 0.9998 | 0.0020 |
| delete ONLY the key frequencies [1, 18, 22, 56] | 0.0097 | 0.0105 | 9.8109 |
| delete 4 control frequencies [2, 20, 38, 55] | 1.0000 | 1.0000 | 2.81e-05 |
| keep ONLY the control frequencies [2, 20, 38, 55] | 0.0123 | 0.0103 | 9.4146 |

**Neuron surgery.**

| MLP neurons mean-ablated | train acc | test acc | test loss |
|:--|--:|--:|--:|
| drop freq 1 | 0.0470 | 0.0452 | 65.3032 |
| keep only freq 1 | 0.0084 | 0.0091 | 1030.1667 |
| drop freq 18 | 0.0055 | 0.0036 | 106.7123 |
| keep only freq 18 | 0.0230 | 0.0211 | 51.7608 |
| drop freq 22 | 0.0086 | 0.0089 | 615.3850 |
| keep only freq 22 | 0.0065 | 0.0098 | 1247.5920 |
| drop freq 56 | 0.9603 | 0.9408 | 0.1922 |
| keep only freq 56 | 0.0089 | 0.0092 | 278.1976 |

**Whole components.**

| component removed | train acc | test acc | test loss |
|:--|--:|--:|--:|
| baseline | 1.0000 | 1.0000 | 6.18e-06 |
| no mlp | 0.0773 | 0.0675 | 4.3367 |
| no head 0 | 0.4676 | 0.4691 | 3.8923 |
| no head 1 | 0.3037 | 0.2797 | 5.8884 |
| no head 2 | 0.2648 | 0.2464 | 7.3799 |
| no head 3 | 0.1721 | 0.1841 | 11.8239 |
| no attention | 0.0091 | 0.0087 | 6.4997 |

**Logit-space restriction.** Keeping 2 directions per key frequency leaves 8 of 12,769 degrees of freedom per output class:

| logit-space edit | split | loss | accuracy |
|:--|:--|--:|--:|
| keep only the key frequencies' (a+b) directions | all pairs | 4.15e-06 | 1.0000 |
| keep only the key frequencies' (a+b) directions | train | 3.98e-06 | 1.0000 |
| delete exactly those directions | train | 26.5789 | 0.0050 |
<!-- END:ablations -->

### Is the circuit minimal?

<!-- BEGIN:redundancy -->
The model settles on 4 frequencies, but that is not the same as needing all 4. Here every subset is kept in the embedding while all 52 non-key frequencies are deleted, and the network is re-run. Chance accuracy is 0.0088:

| frequencies kept in W_E | size | test acc | test loss |
|:--|--:|--:|--:|
| nothing | 0 | 0.0087 | 9.0885 |
| [1] | 1 | 0.0176 | 22.2670 |
| [18] | 1 | 0.0255 | 20.5053 |
| [22] | 1 | 0.0528 | 27.4758 |
| [56] | 1 | 0.0192 | 8.5756 |
| [1, 18] | 2 | 0.1187 | 11.8866 |
| [1, 22] | 2 | 0.2117 | 16.8368 |
| [1, 56] | 2 | 0.0188 | 18.3347 |
| [18, 22] | 2 | 0.2660 | 14.0957 |
| [18, 56] | 2 | 0.0252 | 19.1651 |
| [22, 56] | 2 | 0.0528 | 25.8083 |
| [1, 18, 22] | 3 | 0.9825 | 0.0540 |
| [1, 18, 56] | 3 | 0.1460 | 10.3217 |
| [1, 22, 56] | 3 | 0.2852 | 13.9325 |
| [18, 22, 56] | 3 | 0.2765 | 12.8602 |
| [1, 18, 22, 56] | 4 | 0.9998 | 0.0020 |

**The minimal sufficient set has 3 of the 4 frequencies, and it is unique**: [1, 18, 22] reaches 98.25%, while every other subset of the same size stays below 30%. Frequency 56 is therefore a passenger -- removing it costs almost no accuracy.

It is not free, though: dropping it raises the test loss from 0.0020 to 0.0540, a factor of 26. So under weight decay the extra frequency pays for its own norm, which is what a circuit-efficiency account predicts should happen -- a redundant component survives cleanup exactly when the loss it buys outweighs the penalty it costs.
<!-- END:redundancy -->

---

## 4. The transition is not sudden

<!-- BEGIN:phases -->
Each signal is measured against **its own** range, from its value at initialisation to its final value, so the comparison does not depend on units. A positive lead means the internal signal moves first.

| signal | reaches 10% | lead | reaches 50% | lead |
|:--|--:|--:|--:|--:|
| test accuracy (visible from outside) | 10,453 | 0 | 13,711 | 0 |
| restricted loss | 1,788 | 8,665 | 10,172 | **3,539** |
| embedding spectrum Gini | 189 | 10,263 | 12,822 | 889 |
| power in the key frequencies | 71 | 10,382 | 13,954 | -243 |
| logit variance explained by (a+b) | 7,490 | 2,962 | 13,068 | 643 |
| excluded loss | 13,025 | -2,573 | 13,659 | 52 |
| neurons explained >85% by one frequency | 13,515 | -3,062 | 15,345 | -1,634 |

**Read the 50% column, not the 10% one.** Two of these signals start near a floor set by chance -- the Gini coefficient of a random embedding is not zero, and four of fifty-six frequencies hold about 7% of the power by accident -- so 10% of their eventual change is reached during the memorisation phase, when the embedding is changing violently for reasons that have nothing to do with the circuit. Their apparent ten-thousand-step leads are artefacts of that floor. The restricted loss has no such problem: it starts at the loss of a uniform guess and can only fall by finding real structure, and it leads by about 3,500 steps at the halfway mark.

Read together: the circuit's subspace becomes predictive (restricted loss) and the embedding becomes sparse (Gini) thousands of steps before anything is visible from outside, while excluded loss and neuron crystallisation *lag* -- they measure the removal of the memorised solution, which happens last. That is the three-phase account: memorise, then form the circuit under cover of the memorised solution, then clean the memorised solution away.
<!-- END:phases -->

![progress measures](figures/fig3_progress_measures.png)

---

## 5. Other operations

<!-- BEGIN:operations -->
Each run uses the identical configuration; only the operation (and, in one pair, the modulus) changes. `(a+b) variance` and `trig fraction` are the two mechanism tests from section 2, so a row that groks with a low trig fraction has found a *different* algorithm, not the same one.

| task | grokking step | final test acc | key freqs | Gini(W_E) | (a+b) variance | trig fraction |
|:--|--:|--:|--:|--:|--:|--:|
| `(a + b) mod p`, p=113 | 14,536 | 1.0000 | 4 | 0.9124 | 0.9626 | 0.9966 |
| `(a + b) mod p`, p=113 | none by 40,000 | 0.0559 | -- | -- | -- | -- |
| `(a * b) mod p`, p=113 | none by 40,000 | 0.0673 | -- | -- | -- | -- |

**Runs that did not grok within budget.**

- `(a + b) mod p` at p=113 did not reach 90% test accuracy within its 40,000-step budget. That is a censored observation, not a demonstration that it never would.

- `(a * b) mod p` at p=113 did not reach 90% test accuracy within its 40,000-step budget. That is a censored observation, not a demonstration that it never would.
<!-- END:operations -->

### Multiplication, and the discrete logarithm

<!-- BEGIN:dlog -->
_Not yet run._
<!-- END:dlog -->

---

## 6. When does grokking happen?

<!-- BEGIN:phase_diagram -->
_Not yet run._
<!-- END:phase_diagram -->

---

## Method notes

**The instruments are calibrated.** Every structural metric is run on synthetic
inputs whose answer is known before it is trusted on a real model
(`tests/test_core.py`). On logits built to be exactly
`sum_k cos(w_k (a + b - c))` the "(a+b) variance explained" reads 1.0000 and the
trig fraction reads 1.0000; on random logits they read 0.0088 (= 1/113, chance)
and 0.52. So the numbers in section 2 are interpretable rather than merely
large.

**Key frequencies are derived once, from the final checkpoint, and then held
fixed** across the whole trajectory. That is the point of a progress measure:
the question is when the *final* circuit starts to exist. Re-deriving the set at
each step asks a different question at every step, and at early steps the
spectrum is dense enough that the rules just pick an arbitrary frequency.

**The loss is computed in float64.** In float32, `log_softmax` quantises at
2^-23 = 1.2e-7, so once the model has memorised, the reported training loss
bottoms out at that value and the gradient of the correct class degrades.
Since the entire phenomenon lives in the tens of thousands of steps *after* the
training loss is nominally zero, a loss floor is exactly the wrong artefact to
have. Parameters stay in float32; only the logits are upcast.

**Published numbers are quarantined** in `src/grokking/literature.py`, each with
its reference, and appear only in columns labelled as such. Values that a source
states but that should not be quoted as fact -- a seed-dependent frequency set,
a claim the source contradicts elsewhere -- are listed there too, with the
reason.

---

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                      # no training needed

python scripts/run_train.py --tag main --op add --steps 40000
python scripts/analyze_run.py --tag main        # walk the trajectory
python scripts/analyze_mechanism.py --tag main  # the final-checkpoint report
python scripts/make_figures.py --tag main
python scripts/make_report.py --tag main        # regenerate every table above
```

<!-- BEGIN:runtime -->
The mainline run is 40,000 steps in **82 minutes** on 6 CPU threads (122 ms per full-batch step). Checkpoints for one run are about 140 MB.
<!-- END:runtime -->

---

## Layout

```
src/grokking/
  data.py          the p*p table, tokenisation, the train/test split
  model.py         a one-layer transformer written out explicitly
  train.py         full-batch training with a hybrid log+dense checkpoint schedule
  fourier.py       real Fourier basis over Z_n, and the discrete-log re-indexing
  literature.py    published values, quarantined, each with its reference
  analysis/
    core.py        load a checkpoint, run the whole input table through it
    spectra.py     spectra of weights and activations; three key-frequency rules
    structure.py   does the output depend only on (a+b)?  is it the trig identity?
    progress.py    restricted and excluded loss
    ablation.py    weight-level causal interventions
    dlog.py        the multiplicative-character view of modular multiplication
    timing.py      locating the transition on a trajectory
  viz/             one place where typography and colour are decided
  report.py        markdown table machinery
  report_blocks.py every number that reaches the README comes through here
scripts/           runnable entry points
tests/             correctness tests for everything the results depend on
```

---

## Honesty notes

- **Nothing here is a new discovery.** The phenomenon is Power et al. (2022);
  the mechanism and the progress measures are Nanda et al. (2023); the
  discrete-logarithm reduction for multiplication is Doshi et al. (2024). This
  is a from-scratch reproduction plus a few extensions, and the sections say
  which is which.
- **Runs that fail to grok are reported as failures**, with the step budget
  stated in the same sentence, because "did not grok in 30,000 steps" and "does
  not grok" are different claims.
- **The lead times in section 4 are single-run measurements.** Grokking time is
  known to vary substantially across seeds; the seed replicates in section 5 are
  the only check on that here, and two seeds is not a distribution.
- This project was built with AI assistance (Claude). The experimental design,
  the corrections to the implementation, and the write-up were produced
  interactively; every number was produced by running the code in this
  repository on this machine.
