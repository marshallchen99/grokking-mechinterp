# Grokking, reverse-engineered

A one-layer transformer is trained to compute `(a + b) mod 113` from examples
alone. It memorises its training set within a couple of hundred steps, then
does nothing useful on held-out pairs for thousands more -- the textbook picture
of an overfitted run -- and then generalises.

This repository reproduces that from scratch and takes the trained network
apart: what it computes (a Fourier algorithm, found by
[Nanda et al. 2023](https://arxiv.org/abs/2301.05217)), whether it is really
using it (weight-level ablations with controls), and how the picture changes
for subtraction, multiplication and a quadratic form. It is a reproduction.
Where something here goes beyond the published work, the section says so;
where it does not, the section names the paper.

Everything runs on a laptop CPU. No data is downloaded. Every measured number
below, in the tables and in the sentences around them, is produced by
`scripts/make_report.py` from the shipped results files; a test re-renders
every section from those files and checks it matches this README exactly.

---

## 0. What the task is

The model is never told the rule. It sees pairs of symbols and an answer:

```
    (5, 3)   ->   8
  (100, 50)  ->  37          because 150 - 113 = 37
    (7, 9)   ->  16
```

Nothing says these are numbers, that `+` is addition, or that there is a
modulus. As far as the network is concerned there are 113 arbitrary symbols
and a table to fill in -- and it only sees part of the table. At `mod 5`:

```
        b=0   b=1   b=2   b=3   b=4
 a=0     0     ?     ?     3     ?
 a=1     ?     2     ?     ?     0
 a=2     2     ?     ?     0     ?
 a=3     ?     ?     0     ?     ?
 a=4     ?     0     ?     ?     3
```

Two strategies both score perfectly on the visible cells: memorise them, which
says nothing about the hidden ones, or find the rule. Gradient descent gets to
the first long before the second.

---

## 1. The phenomenon

<!-- BEGIN:headline -->
| event | step | note |
|:--|--:|:--|
| training accuracy reaches 99% | 158 | the training set is memorised |
| test accuracy reaches 10% | 10,007 | chance is 0.88% |
| test accuracy reaches 50% | 13,705 |  |
| test accuracy reaches 90% | 14,536 | generalisation |
| test accuracy reaches 99% | 14,937 |  |

For **14,379 steps** training accuracy stays at or near 99% while the model stays far from generalising. Neither curve is flat on that stretch. Training accuracy drops below 99% 7 times in short loss spikes, as low as 93.5% at step 1,680. Test accuracy is above chance almost from the start and creeps up slowly -- 2.8% at step 789, 6.9% at step 8,006 -- and only passes 10% at step 10,007, 9,849 steps after memorisation.

Test loss rises during memorisation and peaks at **32.74** at step 1,680, which is on one of those spikes (training accuracy 93.5% at the same step).

Setup: `(add) mod 113`, 3,830 of 12,769 pairs used for training (30%), a 226,176-parameter one-layer transformer, full-batch AdamW with weight decay 1.0, 40,000 steps, CPU only. The architecture and optimiser follow Nanda et al. (2023); the phenomenon is Power et al. (2022). This first run also had a float32 loss, no learning-rate schedule, and an output column for the '=' token; section 5 tests whether those mattered.
<!-- END:headline -->

![grokking curve](figures/fig1_grokking_curve.png)

---

## 2. What the network learned

[Nanda et al. (2023)](https://arxiv.org/abs/2301.05217) found that a model
like this one represents each input as a point on a circle, at a handful of
frequencies, and adds by rotating:

```
    a  ->  ( cos(w*a), sin(w*a) )            w = 2*pi*k/p, for a few values of k

    cos(w*(a+b)) = cos(w*a)cos(w*b) - sin(w*a)sin(w*b)
    sin(w*(a+b)) = sin(w*a)cos(w*b) + cos(w*a)sin(w*b)

    logit(c)  proportional to  sum over k of  cos( w_k * (a + b - c) )
```

The last line peaks when every term has phase zero, i.e. at `c = a + b`, and
wraparound is free because the circle is periodic. The sections below test
whether this repository's model does the same. They test the embedding, the
logits and the readout; which layer computes the products was not tested here.

<!-- BEGIN:mechanism -->
The mechanism described here is Nanda et al. (2023)'s; what follows re-derives it on this repository's own model. Three rules are applied to the final checkpoint (step 40,000). Two of them read the same object -- the embedding's Fourier spectrum -- and one reads the MLP neurons, so this is two independent views rather than three. They agree with Jaccard 1.000:

| rule | reads | frequencies found |
|:--|:--|:--|
| embedding-norm threshold | the embedding's Fourier spectrum, per basis index | 1, 18, 22, 56 |
| neuron clustering | which frequency explains most of each MLP neuron | 1, 18, 22, 56 |
| power gap | the embedding's Fourier spectrum, per frequency | 1, 18, 22, 56 |

Those 4 frequencies carry **94.9%** of the embedding's Fourier power, out of 56 available. The spectrum's Gini coefficient is 0.9124; Nanda et al. (2023) report 0.55-0.8 across their settings, but do not say which vector they compute it over, so the two may not be comparable.

Every one of the 512 MLP neurons is dominated by one of these frequencies:

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

**The trigonometric identity.** Within each key frequency, the dependence on (a, b) splits into a part that is a function of (a+b) and a part that is a function of (a-b); the identity `cos(w(a+b)) = cos(wa)cos(wb) - sin(wa)sin(wb)` predicts it is all the former. The measure is exactly one for logits built to be the algorithm, by construction, and one half for random logits, by symmetry; the tests check both (`tests/test_analysis.py`):

| frequency | energy in cos/sin(w(a+b)) |
|:--|--:|
| 1 | 0.9972 |
| 18 | 0.9973 |
| 22 | 0.9935 |
| 56 | 0.9986 |
| **mean** | 0.9966 |
<!-- END:mechanism -->

![embedding spectrum](figures/fig2_embedding_spectrum.png)

### The readout

<!-- BEGIN:readout -->
The mechanism's third step is the readout: the amplitudes of cos(w(a+b)) and sin(w(a+b)) must themselves be waves at the same frequency in the answer c, which is what makes the sum peak at c = a+b. Before measuring it, each input's logits are shifted to mean zero over c. That component adds the same number to every class, so softmax, the loss and every prediction are exactly invariant to it; counting it would measure a direction the model cannot be using. Where the identification rules disagree, the frequencies measured are all those any rule picks.

| run | configuration | direction read | at the predicted frequency | at another key frequency | left over |
|:--|--:|--:|--:|--:|--:|
| `B_add_s0` | add, float64 loss, 10-step warmup, seed 0 | (a+b) | 0.9980 | 2.06e-04 | 0.0018 |
| `B_add_s1` | add, float64 loss, 10-step warmup, seed 1 | (a+b) | **0.9990** | 1.29e-04 | 8.38e-04 |
| `C_add_nowarm` | add, float64 loss, warmup_steps = 1, seed 0 | (a+b) | 0.9978 | 4.14e-04 | 0.0018 |
| `C_add_f32` | add, float32 loss, 10-step warmup, seed 0 | (a+b) | 0.8765 | 0.0329 | 0.0906 |
| `main_add_s0` | add, float32 loss, no LR schedule, extra W_U column, seed 0 | (a+b) | 0.8936 | 0.0219 | 0.0846 |
| `B_sub_s0` | sub, float64 loss, 10-step warmup, seed 0 | (a-b) | 0.9903 | 0.0015 | 0.0082 |

In the float64 addition runs the readout is essentially exact (0.998 to 0.999 at the predicted frequency). Subtraction reads 0.990 in the (a-b) direction it actually uses, and 0.366 if read in the (a+b) direction -- a model read in the wrong coordinate looks unstructured.

The 2 float32 runs are measurably less clean (0.876 to 0.894), with the remainder split between other key frequencies and energy none of them explain. **The cause is not established.** It is 2 runs against 3, and the pairs are not matched on everything: the float32 control `C_add_f32` was read at step 25,000 and its float64 twin `B_add_s0` at step 40,000, so budget differs as well as precision (though `C_add_nowarm`, float64 and read at step 25,000, is as clean as the others). An earlier version of this section attributed the difference to a per-answer bias that float32 prevented from being cleaned up. That was wrong: the component it measured was the softmax-invariant one removed above, which no loss gradient acts on at any precision.
<!-- END:readout -->

---

## 3. Is the model using it?

The evidence above is about what the weights look like. These interventions
test whether the output depends on it.

<!-- BEGIN:ablations -->
These edit the **weights** and re-run the whole network. Chance accuracy is 0.0088.

**Embedding surgery.**

| edit applied to W_E | train acc | test acc | test loss |
|:--|--:|--:|--:|
| none (baseline) | 1.0 | 1.0 | 6.18e-06 |
| keep ONLY the key frequencies [1, 18, 22, 56] | 1.0 | 0.9998 | 0.0020 |
| delete ONLY the key frequencies [1, 18, 22, 56] | 0.0097 | 0.0105 | 9.8109 |
| delete the key frequencies, then restore the embedding's norm | 0.0091 | 0.0077 | 20.7842 |
| delete 4 control frequencies [2, 20, 38, 55] | 1.0 | 1.0 | 2.81e-05 |
| keep ONLY the control frequencies [2, 20, 38, 55] | 0.0123 | 0.0103 | 9.4146 |

The key frequencies hold 94.9% of the embedding's power and the evenly spaced controls 0.93%, so the controls alone cannot rule out that deleting the key frequencies kills the model merely by shrinking its embedding. The norm-restored row does: with the key frequencies deleted and the rest scaled back up to the original norm, test accuracy is 0.77%.

**Neuron surgery.** Each key frequency's neuron cluster, mean-ablated:

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

Two caveats. The clusters differ in size and there is no size-matched random control, so a cluster being survivable to drop may partly reflect its size. And losses as high as 1,248, far above the 4.73 of a uniform guess, mean the ablated network is confidently wrong, so these rows say which clusters matter, not by how much.

**Whole components.**

| component removed | train acc | test acc | test loss |
|:--|--:|--:|--:|
| baseline | 1.0 | 1.0 | 6.18e-06 |
| no mlp | 0.0773 | 0.0675 | 4.3367 |
| no head 0 | 0.4676 | 0.4691 | 3.8923 |
| no head 1 | 0.3037 | 0.2797 | 5.8884 |
| no head 2 | 0.2648 | 0.2464 | 7.3799 |
| no head 3 | 0.1721 | 0.1841 | 11.8239 |
| no attention | 0.0091 | 0.0087 | 6.4997 |

**Logit-space restriction** (Nanda et al. (2023)'s restricted and excluded loss). This edits the output logits, not the weights, so it is a projection rather than an intervention. The logits for one output class, as a function of (a, b), have 12,769 degrees of freedom; keeping two directions per key frequency, plus the per-class mean, leaves 9:

| logit-space edit | split | loss | accuracy |
|:--|:--|--:|--:|
| keep only the key frequencies' (a+b) directions | all pairs | 4.15e-06 | 1.0 |
| keep only the key frequencies' (a+b) directions | train | 3.98e-06 | 1.0 |
| delete exactly those directions | train | 26.5789 | 0.0050 |
<!-- END:ablations -->

![ablations](figures/fig4_ablations.png)

### Which key frequency is dispensable

<!-- BEGIN:redundancy -->
Every subset of the 4 key frequencies is kept in the embedding, all other frequencies are deleted, and the network is re-run. Chance is 0.0088:

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

Among subsets of the key set, the only 3-frequency subset that reaches 90% is [1, 18, 22], at 98.25%; the other 3 stay at or below 28.5%. Frequency 56 is dispensable for accuracy, though keeping it lowers the test loss from 0.0540 to 0.0020. Whether that is why training keeps it -- a loss benefit outweighing its weight-decay cost -- is plausible but was not measured here.
<!-- END:redundancy -->

<!-- BEGIN:disagreement -->
| run | weakest key frequency in the embedding | frequency the rules disagree on | key frequencies that can each be deleted alone |
|:--|--:|--:|--:|
| `main_add_s0` | 56 | none | 56 |
| `B_add_s0` | 16 | 16 | 16, 38 |
| `B_add_s1` | 26 | 26 | 26 |
| `B_sub_s0` | 17 | 17 | 17 |
| `C_add_f32` | 38 | none | 38 |

'Can be deleted alone' means the rest of the key set, with every other frequency also removed, still reaches 90% test accuracy. In 4 of 5 runs exactly one key frequency qualifies, and it is the weakest. In `B_add_s0` 2 qualify (16, 38), and the weakest, 16, is one of them. In 3 of the 3 runs where the identification rules disagree, they disagree about the weakest frequency. An earlier version presented rule disagreement as a *prediction* of redundancy. It is not independent evidence: the deletion test edits the embedding, and two of the rules threshold that same embedding, so the weakest component being both the marginal one and the dispensable one is close to what you would expect.
<!-- END:disagreement -->

### Identification from behaviour

<!-- BEGIN:load_bearing -->
The rules above read the model's representation. This reads its behaviour: remove one frequency's two (a+b) Fourier directions from the logits at a time and measure the training loss. It still works in the same Fourier basis, so it is not independent machinery, and taking 'the top k' borrows k from the rules; but it involves no threshold of its own.

| frequency removed | train loss afterwards |
|:--|--:|
| 22  (key) | 11.1375 |
| 18  (key) | 1.2226 |
| 1  (key) | 0.4058 |
| 56  (key) | 0.0060 |
| 44 | 1.24e-05 |
| 47 | 2.50e-06 |

At the boundary -- the weakest key frequency against the strongest of the rest -- the separation is a factor of 482 (2.7 orders of magnitude). Most non-key frequencies cost nothing measurable: removing one leaves the training loss at about 3.14e-07 (median over the 52), indistinguishable from removing nothing. The top 4 by this measure are [1, 18, 22, 56], the same set the rules find.
<!-- END:load_bearing -->

---

## 4. When the circuit forms

<!-- BEGIN:phases -->
The progress measures and the three-phase account are Nanda et al. (2023)'s. Each signal is measured against its own range, from initialisation to final value, and the table gives the step at which it has made half its total change. A positive lead means it gets there before test accuracy does.

| signal | reaches 50% of its change | lead over test accuracy |
|:--|--:|--:|
| test accuracy (visible from outside) | 13,711 | 0 |
| restricted loss | 10,172 | 3,539 |
| embedding spectrum Gini | 12,822 | 889 |
| logit variance explained by (a+b) | 13,068 | 643 |
| excluded loss | 13,659 | 52 |
| neurons explained >85% by one frequency | 15,345 | -1,634 |

**Resolution.** Checkpoints near the transition are about 887 steps apart, so a lead smaller than that is not distinguishable from zero. Against that: ahead by more than the spacing: restricted loss, embedding spectrum Gini; within it: logit variance explained by (a+b), excluded loss; behind by more than it: neurons explained >85% by one frequency. The restricted loss is also not monotone: it first rises, to 7.85 at step 2,925, before it falls.

So: the restricted loss reaches the midpoint of its change 3,539 steps before test accuracy reaches its own midpoint, while the excluded loss -- the measure that tracks removal of the memorised solution -- is within resolution of test accuracy (+52 steps). That order is consistent with Nanda et al. (2023)'s account. This is one run, and test accuracy has already begun to rise by then; the lead is over its midpoint, not over its first movement.
<!-- END:phases -->

![progress measures](figures/fig3_progress_measures.png)

---

## 5. Other operations

<!-- BEGIN:operations -->
Same model and optimiser, one seed per operation; the step budgets differ (see the column), so the mechanism is read at different points after grokking. The last column is the variance of the logits explained by the task's own combination of the inputs -- (a+b), (a-b) or (a*b) -- in the ordinary basis. 'Key freqs' is '--' where the identification rules find no common set, which for multiplication means the ordinary basis is the wrong place to look (next subsection). The survey of which modular operations grok is Furuta et al. (2024)'s.

| run | task | p | grokking step | budget | final test acc | key freqs | Gini(W_E) | variance explained by own combination |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| `B_add_s0` | `(a + b) mod p` | 113 | 7,083 | 40,000 | 1.0 | 5 | 0.9225 | 0.9874 |
| `B_sub_s0` | `(a - b) mod p` | 113 | 27,242 | 30,000 | 1.0 | 4 | 0.9467 | 0.9994 |
| `B_mul_s0` | `(a * b) mod p` | 113 | 7,571 | 40,000 | 1.0 | -- | 0.0176 | 0.9731 |
| `B_sqx_p113` | `(a^2 + ab + b^2) mod p` | 113 | none by 30,000 | 30,000 | 0.1138 | -- | -- | -- |
| `B_sqx_p109` | `(a^2 + ab + b^2) mod p` | 109 | none by 30,000 | 30,000 | 0.0893 | -- | -- | -- |

Subtraction's output tracks (a-b) rather than (a+b). It grokked at step 27,242 against 7,083 for `B_add_s0`, a factor of 3.8; seeds of a single configuration elsewhere in this project span a factor of 3.0 (at a different modulus and training fraction). With one seed per operation this is weak evidence either way. It grokked 2,758 steps before its budget ran out, so its mechanism was read much closer to the transition than addition's.

**Censored runs.**

- `B_sqx_p113` did not reach 90% within 30,000 steps -- censored, not shown never to grok.

- `B_sqx_p109` did not reach 90% within 30,000 steps -- censored, not shown never to grok.
<!-- END:operations -->

![operations](figures/fig5_operations.png)

### Multiplication, and the discrete logarithm

<!-- BEGIN:dlog -->
The nonzero residues mod p form a cyclic group of order p-1 under multiplication, so re-indexing them by discrete logarithm turns `a * b` into `dlog(a) + dlog(b) mod (p-1)`. None of this is new. The reduction is Doshi et al. (2024) (arXiv:2406.03495); a transformer trained on multiplication has been shown sparse in this basis (Nguyen (2026), arXiv:2606.17399); and restricted/excluded-loss ablations in the irreducible-representation basis of a cyclic group -- which is what the discrete-log Fourier basis is -- are Chughtai et al. (2023) (arXiv:2302.03025). What follows reproduces those on this repository's own model, which was trained on the full table including 0.

| basis | key frequencies | Gini(W_E) | power in key freqs | rules agree (Jaccard) |
|:--|--:|--:|--:|--:|
| ordinary (residues 0..p-1) | -- | 0.0176 | -- | 0.0 |
| discrete log (g = 3, n = 112) | 3 | 0.9361 | 0.9690 | 1.0 |

In the ordinary basis the three rules share no frequency at all, so there is no key set to report; in the discrete-log basis they agree exactly. How much of the logits' variance is a function of a*b does not depend on the basis (0.973); only the sparsity does.

**Weight surgery in the multiplicative basis.** The nonzero residues' embedding rows are re-indexed by discrete logarithm, filtered in the Fourier basis over Z_112, written back, and the whole network is re-run. Accuracy is on held-out pairs with both inputs nonzero:

| edit | test acc |
|:--|--:|
| none | 1.0 |
| keep only the key frequencies [15, 33, 53] (and the mean) | 0.9997 |
| delete only the key frequencies | 0.0087 |
| delete 3 control frequencies [1, 19, 37] | 1.0 |
| keep only the control frequencies (and the mean) | 0.0090 |

**The absorbing element.** Zero has no multiplicative inverse, so it is outside the group. Prior work excludes it or treats it as a separate stratum (Doshi et al. (2024); Chen et al. (2026), correlationally, at p = 113 among other moduli). Here the model is correct on 100% of the pairs containing a zero, no neuron behaves like a detector for it (strongest correlation with `a == 0`: 0.035), and its embedding row has norm 0.51 against a mean of 1.05 -- the smallest of all. Editing that one row and re-running the model:

| edit to the embedding of 0 | accuracy, all 225 pairs containing a 0 (train and test) |
|:--|--:|
| none | 1.0 |
| set to zero | 0.9956 |
| rescaled to the mean norm of the other rows | 1.0 |
| doubled | 1.0 |
| replaced by the mean of the other rows | 0.9956 |
| replaced by a random direction at the mean norm (median of 20) | 0.9822 |
| scaled by 10.0 | 0.0 |

Restoring its norm to the mean leaves accuracy at 1.00, so the small norm is not what makes the model answer 0. Replacing the row usually leaves it answering 0 too (random directions at the mean norm: median 0.98 over 20, but as low as 0.20); scaling it by 10.0 breaks it (0.00).

So what does make it answer 0? Blank the embedding row of one *nonzero* residue instead (zero it, or replace it with the mean row), for every one of the 112 nonzero residues in either input position, and score only pairs whose other input is untouched: the model answers 0 on 100.0% of 24,864 pairs when zeroed and 100.0% when averaged. Blank *both* inputs' rows and it answers 59, never 0, on the 64 pairs tested (0% answer 0). Every such pair then presents the same input, so their agreeing is automatic; which class they agree on is the measurement. The real pair (0, 0), once 0's own row is blanked, becomes that same input and gives 59. So the rule is narrower than 'a blank input acts as zero': exactly one uninformative input gives 0, and two give a fixed non-zero class. Which weights implement this was not identified.
<!-- END:dlog -->

### A quadratic form

<!-- BEGIN:quadratic -->
`a^2 + ab + b^2` splits into linear factors over F_p exactly when p = 1 (mod 3). This pair of primes asks whether that matters. It is not a hypothesis from the literature: Furuta et al. (2024) call the form non-factorisable in the sense of not being expressible through (a +- b), and Doshi et al. (2024)'s Hypothesis 5.1 concerns forms h(g1(a) + g2(b)); neither is about splitting over F_p. Furuta et al. (2024) also already report that it does not grok at p = 97, where it does split.

| run | p | form over F_p | test acc | held-out pairs whose transpose was trained on | acc on those | acc on the rest | chance |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `Q_sqx_p59` | 2 mod 3 | irreducible | 0.4951 | 0.4842 | 1.0 | 0.0212 | 0.0169 |
| `Q_sqx_p61` | 1 mod 3 | splits | 0.4970 | 0.4793 | 1.0 | 0.0341 | 0.0164 |

**Neither generalises within 60,000 steps**, consistent with Furuta et al. (2024)'s result: splitting over F_p does not appear to matter here. Both sit near 50% test accuracy, and the cause is not partial learning of the form. It is symmetric in a and b while the train/test split is over *ordered* pairs, so about half the held-out pairs have their transpose in the training set; the model gets essentially all of those right and is near chance on the rest. It has memorised the training table and learned that the table is symmetric.

Its *mistakes* are structured, though. `a^2 - ab + b^2` is the same form with the sign of one input flipped, and on the held-out pairs it cannot answer from memory, the model often predicts exactly that. Whether it does depends on whether a sign-flipped partner of the pair -- (a, -b), (-a, b) or their transposes, all of which share that value -- was in the training set:

| run | pairs with a sign-flipped partner trained | predicts a^2 - ab + b^2 | pairs without one | predicts a^2 - ab + b^2 | chance |
|:--|--:|--:|--:|--:|--:|
| `Q_sqx_p59` | 798 | 0.3935 | 64 | 0.0312 | 0.0169 |
| `Q_sqx_p61` | 875 | 0.3886 | 66 | 0.0303 | 0.0164 |

So these look like memorised answers retrieved for the wrong key. The embedding does place each residue near its negative (mean cosine similarity 0.58 at p = 59, 0.58 at p = 61; random pairs -0.04, -0.00). But that alone predicts that a *double* flip (-a, -b), whose value equals the true one, would be retrieved as readily and give the right answer; where only such a partner was trained, the model is right on 4% (n = 48), 2% (n = 46). Why single flips are retrieved and double flips are not was not established, and the without-partner groups are small. An earlier version read the plateau as a circuit that had 'lost the sign of b'; that explained neither the accuracy, which symmetry accounts for, nor the dependence of these errors on which partners were trained. Splitting on unordered pairs would remove both effects and make this a fair test of factorability.
<!-- END:quadratic -->

### Which correction mattered?

<!-- BEGIN:controls -->
The first run used float32 cross-entropy, no warmup, and an unembedding with a column for the '=' token that can never be correct. The corrected configuration grokked in about half the steps. Each control below changes one training choice, one run each; their step budgets also differ, which does not affect the grokking step but does change the step at which the mechanism is read.

| run | what differs | grokking step |
|:--|:--|--:|
| `main_add_s0` | add, float32 loss, no LR schedule, extra W_U column, seed 0 | 14,536 |
| `B_add_s0` | add, float64 loss, 10-step warmup, seed 0 | 7,083 |
| `C_add_f32` | add, float32 loss, 10-step warmup, seed 0 | 6,734 |
| `C_add_nowarm` | add, float64 loss, warmup_steps = 1, seed 0 | 9,764 |
| `B_add_s1` | add, float64 loss, 10-step warmup, seed 1 | 6,228 |

Changing only the loss precision moves the step by -350; removing the warmup by +2,681. For scale, two seeds of the corrected configuration differ by 855. So neither change, alone, accounts for the gap to the original's 14,536.

What else differs is narrower than it might seem. At step 0 the two runs share 8 of their 9 weight tensors bit for bit; only W_U is drawn differently, because it is drawn last and its shape changed. The remaining gap is therefore some combination of W_U's initialisation, the extra column itself, an interaction between float32 and no warmup (never run jointly), and chance. These single runs cannot separate them, and an earlier claim that the original was simply 'a slow draw' went beyond them. Note also that `warmup_steps = 1` still gives a zero learning rate on the first step.
<!-- END:controls -->

---

## 6. Weight decay and grokking time

<!-- BEGIN:phase_diagram -->
A smaller modulus (p = 59) makes a run cheap enough to sweep. One seed per cell, 20,000 steps each; the number is the step at which test accuracy first reaches 90%.

|  | wd = 0.1 | wd = 0.3 | wd = 1.0 | wd = 3.0 |
|:--|--:|--:|--:|--:|
| train fraction 0.25 | none (best 2%) | none (best 2%) | none (best 2%) | none (best 2%) |
| train fraction 0.35 | none (best 5%) | none (best 6%) | none (best 17%) | 14,426 |
| train fraction 0.5 | 10,794 | 3,100 | 815 | 344 |

7 of 12 cells did not get there within 20,000 steps -- censored, not shown never to grok.

Within the one training fraction where every cell grokked (0.5), the grokking step falls monotonically as weight decay rises: 10,794, 3,100, 815, 344. Step times weight decay stays between 815 and 1,079 while weight decay spans a factor of 30. On log axes the slope is -1.02 counting from step 0, and -1.17 counting from the end of memorisation, which is the delay the published law is about. One seed and four points; consistent with a roughly 1/lambda dependence, which Lyu et al. (2023) (arXiv:2311.18817) prove in a large-initialisation limit and Khanh et al. (2026) fit under AdamW. This reproduces that; it does not discover it.
<!-- END:phase_diagram -->

![phase diagram](figures/fig6_phase_diagram.png)

<!-- BEGIN:replicates -->
The same training fraction, a second seed for both split and initialisation:

|  | seed 0 (20,000-step budget) | seed 1 (12,000-step budget) |
|:--|--:|--:|
| weight decay 0.1 | 10,794 | none by 12,000 (best 79%) |
| weight decay 0.3 | 3,100 | 4,010 |
| weight decay 1.0 | 815 | 922 |
| weight decay 3.0 | 344 | 255 |

Every seed orders the cells the same way, from most weight decay (fastest) to least, counting a cell that did not grok as later than its budget. The budgets differ, so a censored cell is comparable only with its own seed's order, not with the other seed's number in the same row.
<!-- END:replicates -->

---

## 7. Which early signals track the transition

<!-- BEGIN:prediction -->
Notsawo et al. (2023) (arXiv:2306.13253) predict *whether* grokking will occur from oscillations in the early training-loss curve, and Khanh et al. (2026) (arXiv:2605.18845) predict *when*, across hyperparameter settings, from the parameter norm. The question here is narrower still: within a single configuration, where only the random draw differs, which early signals rank runs by when they generalise.

16 runs share the task (`add`), modulus (59), training fraction (0.5), weight decay (1.0) and budget (6,000 steps). They grok between step 760 and 2,266. A reading taken after some run has grokked measures the outcome, so only readings before step 760 are used.

These are not forecasts made before anything happens. In this configuration test accuracy starts rising almost immediately: at step 200 it is already 15% to 49%; at step 500 it is already 30% to 59% (chance 1.7%). The readings rank how far along a transition already under way each run is.

Spearman correlation with the grokking step; positive means a higher reading goes with a later transition. `*` marks coefficients whose two-tailed permutation p-value survives Bonferroni over the 24 tests at the 2 reported steps (|rho| of at least 0.726); readings at later steps were also computed and are excluded, as above. Rows marked `(final)` use the finished model's key frequencies and so borrow information from after the transition; their 'leak-free' counterparts use the frequencies each checkpoint would pick itself:

| signal | at step 200 | at step 500 |
|:--|--:|--:|
| restricted loss (train pairs) (final) | +0.72 | +0.81 * |
| excluded loss (train pairs) (final) | -0.60 | -0.77 * |
| embedding Gini (no frequency choice) | -0.61 | -0.72 |
| power in key frequencies (final) | -0.31 | -0.51 |
| (a+b) variance explained (no frequency choice) | -0.58 | -0.69 |
| weight norm | +0.67 | +0.79 * |
| train loss | +0.59 | +0.84 * |
| test accuracy (visible from outside) | -0.00 | -0.52 |
| test loss (visible from outside) | +0.49 | +0.86 * |
| restricted loss (train pairs), leak-free | +0.44 | +0.83 * |
| excluded loss (train pairs), leak-free | -0.65 | -0.74 * |
| power in key frequencies, leak-free | -0.41 | -0.77 * |

At step 200 nothing survives the correction. The largest point estimate there is the restricted loss (train pairs) (final) (+0.72), a mechanistic measure, but none of these is reliable at this n.

At step 500, 8 signals do. The strongest is the **test loss (visible from outside)** (rho +0.86), and the survivors are not distinguishable from one another at n = 16. In this one configuration, then, plain losses rank the runs as well as the mechanistic measures do by step 500, and the ordinary test loss -- visible from outside -- is among the best. An earlier version of this section claimed the visible quantity could not rank the runs; it had looked only at test accuracy.

Earlier versions of this table were also wrong for a mechanical reason: runs with a non-zero data seed were analysed on the seed-0 train/test split, which corrupted every split-dependent signal. The numbers above are from the corrected analysis.
<!-- END:prediction -->

---

## Method notes

**The instruments are calibrated.** Every structural metric is run on
synthetic logits whose answer is known before it is trusted on a real model;
`tests/test_analysis.py` asserts, for instance, that the exact algorithm scores
1 on the (a+b) tests, that an (a-b) algorithm scores 0, and that random logits
sit at chance.

**Key frequencies are taken from the final checkpoint** for the progress
measures in section 4, because the question there is when the *final* circuit
starts to exist. For forecasting (section 7) that would leak information from
after the transition, so the frequency-dependent signals are also computed with
the frequencies each checkpoint picks for itself, and reported separately.

**Loss in float64, except in the first run and its float32 control.** In float32, `log_softmax`
quantises at about 1.2e-7, which is where the training loss sits for most of a
run. The headline run in sections 1-4 predates the switch and used float32, so
its training-loss curve below that level is quantisation, not signal. In the
controlled comparison of section 5, switching only the loss precision moved the
grokking step by less than the difference between two seeds.

**Reproducible in distribution, not bit for bit.** With one thread identical
seeds give identical weights, and the tests assert it. With several threads
PyTorch's CPU reductions do not fix their summation order and runs diverge
slightly, so a re-run lands near these numbers rather than on them.

**Two defects, disclosed rather than hidden.**
- *A data-seed bug*, now fixed: the analysis scripts once took the train/test
  split's seed as a command-line flag defaulting to 0, so runs with any other
  seed were analysed on the wrong split. Seventeen runs were affected, and
  several numbers in an earlier version of section 7 were wrong. The scripts
  now read the split from the run's own record, and a test asserts it.
- *Unused output columns at p != 113.* The output width defaulted to 113, so
  every run at another modulus carries extra columns in its unembedding that
  are sliced off before the loss. They receive no gradient, only weight decay,
  and cannot affect predictions or the circuit; they do add a near-constant
  term to the weight norm used in section 7. Fixed for future runs; the
  affected runs were not retrained.

**Published numbers are quarantined** in `src/grokking/literature.py`, each
with its reference; values a source states but that should not be quoted as
fact are listed there with the reason.

---

## Reproducing

```bash
pip install -r requirements.txt
python -m pytest tests/ -q                   # includes re-rendering this README from results/

# regenerate every figure and table from the shipped results
python scripts/make_figures.py
python scripts/make_report.py

# train and analyse a fresh run in the corrected configuration (B_add_s0's)
python scripts/run_train.py --tag repro --op add --steps 40000
python scripts/analyze_run.py --tag repro
python scripts/analyze_mechanism.py --tag repro
```

A fresh run reproduces `B_add_s0`'s configuration, not the headline run's: that
first run used a float32 loss, no learning-rate schedule and an output column
for the `=` token, which the current code no longer creates. Its results are
shipped in `results/`. Runs with more than one thread are not bit-for-bit
reproducible (see Method notes), so a fresh run lands near these numbers.

<!-- BEGIN:runtime -->
The mainline run is 40,000 steps in **82 minutes** on 6 CPU threads (122 ms per full-batch step).
<!-- END:runtime -->

---

## Layout

```
src/grokking/
  data.py          the p*p table, tokenisation, the train/test split
  model.py         a one-layer transformer written out explicitly
  train.py         full-batch training with a hybrid log+dense checkpoint schedule
  fourier.py       real Fourier basis over Z_n, and the discrete-log re-indexing
  runinfo.py       a run's data configuration, read from the run itself
  literature.py    published values, quarantined, each with its reference
  analysis/        spectra, structure tests, progress measures, ablations, dlog view
  viz/             one place where typography and colour are decided
  report.py        markdown table machinery
  report_blocks.py every number that reaches this file comes through here
scripts/           runnable entry points (run_pipeline.sh chains them)
tests/             correctness and calibration tests
```

---

## What is borrowed, and what went wrong

**Borrowed.** The phenomenon: [Power et al. 2022](https://arxiv.org/abs/2201.02177).
The mechanism, the neuron-clustering rule, restricted and excluded loss, and the
three-phase account: [Nanda et al. 2023](https://arxiv.org/abs/2301.05217).
Ablating irreducible-representation subspaces of a group, including cyclic
groups: [Chughtai et al. 2023](https://arxiv.org/abs/2302.03025). The
discrete-log reduction for multiplication, and the factorisation hypothesis for
modular polynomials: [Doshi et al. 2024](https://arxiv.org/abs/2406.03495).
Which modular polynomials grok: [Furuta et al. 2024](https://arxiv.org/abs/2402.16726).
Discrete-log sparsity in a trained transformer:
[Nguyen 2026](https://arxiv.org/abs/2606.17399). Non-invertible elements as a
separate region: [Chen et al. 2026](https://arxiv.org/abs/2607.07066). The
1/lambda dependence of grokking time on weight decay:
[Lyu et al. 2023](https://arxiv.org/abs/2311.18817) and
[Khanh et al. 2026](https://arxiv.org/abs/2605.18845). Predicting whether
grokking will happen from the early loss curve:
[Notsawo et al. 2023](https://arxiv.org/abs/2306.13253).

**Possibly new, and small.** Embedding-weight surgery in the discrete-log
basis, with a control, on a transformer trained with 0 included: Chughtai et al.
ablate irrep subspaces of activations and of the unembedding for group tasks,
Nguyen excludes 0 and does not ablate, and Chen et al. train with 0 included at
p = 113 but report correlational evidence only. The zero element: its small
embedding norm is not why the model answers 0, and blanking exactly one
nonzero input makes it answer 0 while blanking both does not. A
within-configuration comparison of which early signals rank runs by when they
generalise, in which plain losses do as well as mechanistic measures; Khanh et
al. predict the delay across settings from the parameter norm.

**Corrected along the way.** This write-up went through three rounds of
adversarial review by AI agents before publication, and much of what earlier versions
claimed did not survive: a mechanism for a float32 readout difference; a
"sign-blind circuit" reading of the quadratic form, and then an overcorrection
that ignored its structured errors, and an attribution of the factorisation
question to papers that do not pose it; three successive accounts of the zero
element; a reference run labelled with the wrong loss precision; a separation
stated as seven orders of magnitude; rule disagreement
presented as an independent prediction; the claim that the visible quantity
cannot forecast the transition; novelty claims contradicted by published work;
a citation with a wrong author list; forecasting numbers corrupted by a
data-seed bug and a double-counted signal; and reproduction instructions that
did not reproduce the headline run. Each section states the corrected version.
Runs that fail to grok are reported with their step budget.

**How this was made.** This repository was designed, implemented and written
by an AI system (Claude, by Anthropic) working as an agent at my direction. I
chose the topic and set the scope; the experiment design, the code, the
analysis and this write-up are the model's. Every number was produced by
running that code on my machine. Several corrections came out of review
questions I asked about whether the work was finished, and the rest from three
rounds of adversarial review before publication. Those reviews were also run
by AI agents -- separate Claude instances, each told to refute the write-up,
launched from a script that I directed -- not by human reviewers. I read their
findings; the model made the corrections.
