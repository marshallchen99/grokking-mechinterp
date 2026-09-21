# Grokking, reverse-engineered

A one-layer transformer is trained to compute `(a + b) mod 113` from examples
alone. It memorises its training set within a couple of hundred steps, then
improves only slowly on held-out pairs for thousands more -- close to the
textbook picture of an overfitted run -- and then generalises.

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

Setup: `(add) mod 113`, 3,830 of 12,769 pairs used for training (30%), a 226,176-parameter one-layer transformer, full-batch AdamW with weight decay 1.0, 40,000 steps, CPU only. Width, heads, MLP size, the absence of LayerNorm and the optimiser follow Nanda et al. (2023); unlike theirs, the MLP has no biases, as in Furuta et al. (2024). The phenomenon is Power et al. (2022). This first run also had a float32 loss, no learning-rate schedule, and an output column for the '=' token that was cut off before the loss; section 5 looks at what those changed.
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

Those 4 frequencies carry **94.9%** of the embedding's Fourier power, out of 56 available. The Gini coefficient of that per-frequency power is 0.9124. Nanda et al. (2023) define their Gini differently, over the norms of the Fourier components (squaring them, as power does, makes the Gini much larger). On their definition this model's embedding scores 0.57, inside the 0.55-0.80 of their non-dropout models, and its neuron-logit map W_L scores 0.57, below their 0.68-0.91.

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

These are computed on the raw logits. Removing first, for every input pair, the mean over the answer c -- a component softmax ignores, which the readout below also removes -- gives 0.9680 for (a + b) and 0.9981 for the mean trigonometric measure below.

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
| `C_add_nowarm` | add, float64 loss, one-step warmup (zero learning rate on step 0), seed 0 | (a+b) | 0.9978 | 4.14e-04 | 0.0018 |
| `C_add_f32` | add, float32 loss, 10-step warmup, seed 0 | (a+b) | 0.8765 | 0.0329 | 0.0906 |
| `main_add_s0` | add, float32 loss, no LR schedule, extra W_U column, seed 0 | (a+b) | 0.8936 | 0.0219 | 0.0846 |
| `B_sub_s0` | sub, float64 loss, 10-step warmup, seed 0 | (a-b) | 0.9903 | 0.0015 | 0.0082 |

In the float64 addition runs the readout is essentially exact (0.998 to 0.999 at the predicted frequency). Subtraction reads 0.990 in the (a-b) direction it actually uses. In the (a+b) direction there is almost nothing to read: the amplitudes there carry 7.5e-06 times the energy of the (a-b) ones, so the fractions of it are not meaningful.

The 2 float32 runs are measurably less clean (0.876 to 0.894), with the remainder split between other key frequencies and energy none of them explain. **The cause is not established.** It is 2 runs against 3, and the pairs are not matched on everything: the float32 control `C_add_f32` was read at step 25,000 and its float64 twin `B_add_s0` at step 40,000, so budget differs as well as precision (though `C_add_nowarm`, float64 and read at step 25,000, is as clean as the others).

An earlier version attributed the difference to a per-answer bias that float32 prevented from being cleaned up; a later one said that no gradient acts on that component at any precision. Neither holds up. The component is real, and float32 does keep it: the part of W_U along the direction softmax ignores grows during the float32 runs (`C_add_f32` from 0.77 to 2.16, 45% of W_U's norm; `main_add_s0` from 0.78 to 2.27, 43% of W_U's norm) while it shrinks in the float64 ones (to between 0.013 and 0.023), consistent with float32 rounding giving the gradient a part along it that the exact gradient does not have; that gradient was not measured. Either way it is exactly the component removed before measuring, so it cannot change a prediction and does not explain the difference in this table.
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

**Neuron surgery.** Each MLP neuron is assigned to the frequency that explains most of its variance. A cluster is mean-ablated by replacing its neurons' outputs with their average over all inputs; the control does the same to as many neurons drawn at random from outside the cluster:

| frequency | neurons in its cluster | test acc, cluster mean-ablated | test acc, as many other neurons mean-ablated: median (range of 20) | test acc, only this cluster kept |
|:--|--:|--:|--:|--:|
| 1 | 102 | 0.8831 | 0.9845 (0.9559 to 0.9956) | 0.0585 |
| 18 | 129 | 0.7050 | 0.9104 (0.8479 to 0.9586) | 0.0672 |
| 22 | 208 | 0.3302 | 0.7278 (0.6645 to 0.7813) | 0.1103 |
| 56 | 73 | 0.9970 | 0.9989 (0.9947 to 0.9997) | 0.0359 |

Removing the cluster for each of frequencies 1, 18, 22 costs more test accuracy than removing as many random other neurons does in any of the 20 draws: 1.9 to 2.7 times the largest accuracy loss in any draw; for 56 it is within the random range. No single cluster is enough on its own (test accuracy 3.59% to 11.03% with only it kept). Every one of the 512 neurons belongs to some key frequency's cluster, so the random neurons come from the other clusters. An earlier version of this table added the removed neurons' average output at the model's input instead of at the MLP's output, so it went through attention and the MLP a second time; its numbers, and a caveat about test losses above a thousand, were artefacts of that bug.

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
The progress measures and the three-phase account are Nanda et al. (2023)'s. Each signal is measured against its own range, from its value at initialisation to its final level, taken as the median of the last 8 checkpoints (steps 25,013 to 40,000); the table gives the step at which it has made half that change. A positive lead means it gets there before test accuracy does. The last column repeats the lead with the final level taken as the last checkpoint alone, to show which leads depend on that choice.

| signal | reaches 50% of its change | lead over test accuracy | lead, final level = last checkpoint |
|:--|--:|--:|--:|
| test accuracy (visible from outside) | 13,711 | 0 | 0 |
| restricted loss | 10,172 | 3,539 | 3,539 |
| embedding spectrum Gini | 12,822 | 889 | 885 |
| logit variance explained by (a+b) | 13,068 | 643 | 622 |
| excluded loss | 13,659 | 52 | -2,599 |
| neurons explained >85% by one frequency | 15,345 | -1,634 | -1,491 |

**Resolution.** Checkpoints near the transition are about 887 steps apart, so a lead smaller than that is not distinguishable from zero. Against that: ahead by more than the spacing: restricted loss; at the edge of resolution: embedding spectrum Gini; within it: logit variance explained by (a+b); behind by more than it: neurons explained >85% by one frequency; moved by more than the spacing when the final level is chosen the other way: excluded loss. The restricted loss is also not monotone: it first rises, to 7.85 at step 2,925, before it falls.

So: the restricted loss reaches the midpoint of its change 3,539 steps before test accuracy reaches its own midpoint, with either choice of final level. The excluded loss cannot be placed: after the transition it swings between 13.0 and 26.6 from one checkpoint to the next, so its final level is not well defined, and its lead is +52 steps with one choice and -2,599 with the other. The restricted loss leading is consistent with Nanda et al. (2023)'s account. This is one run, and test accuracy has already begun to rise by then; the lead is over its midpoint, not over its first movement.

The halfway point, the window that defines the final level and the resolution rule were all chosen after these curves had been seen. Measured at other fractions of each signal's change, the restricted loss leads test accuracy by 6,731 steps at 25% and 2,803 steps at 75%.
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

Subtraction's output tracks (a-b) rather than (a+b). It grokked at step 27,242 against 7,083 for `B_add_s0`, a factor of 3.8. Among the 120 pairs of seeds in the one configuration run many times, none differs by that factor or more; but that configuration has a different modulus and training fraction, and at this one seed noise was measured only by a single pair (`B_add_s1` against `B_add_s0`, 1.14x). With one seed per operation this is suggestive at most. Subtraction grokked 2,758 steps before its budget ran out, so its mechanism was read much closer to the transition than addition's.

**Censored runs.**

- `B_sqx_p113` did not reach 90% within 30,000 steps -- censored, not shown never to grok.

- `B_sqx_p109` did not reach 90% within 30,000 steps -- censored, not shown never to grok.
<!-- END:operations -->

![operations](figures/fig5_operations.png)

### Multiplication, and the discrete logarithm

<!-- BEGIN:dlog -->
The nonzero residues mod p form a cyclic group of order p-1 under multiplication, so re-indexing them by discrete logarithm turns `a * b` into `dlog(a) + dlog(b) mod (p-1)`. None of this is new. The reduction is Doshi et al. (2024) (arXiv:2406.03495), who also show that MLPs trained on multiplication become periodic in this basis; a transformer trained on it has been shown sparse in it (Nguyen (2026), arXiv:2606.17399); and restricted/excluded-loss ablations in the irreducible-representation basis of a cyclic group -- which is what the discrete-log Fourier basis is -- are Chughtai et al. (2023) (arXiv:2302.03025). What follows reproduces those on this repository's own model, which was trained on the full table including 0.

| basis | key frequencies | Gini(W_E) | power in key freqs | rules agree (Jaccard) |
|:--|--:|--:|--:|--:|
| ordinary (residues 0..p-1) | -- | 0.0176 | -- | 0.0 |
| discrete log (g = 3, n = 112) | 3 | 0.9361 | 0.9690 | 1.0 |

In the ordinary basis the rules pick embedding-norm threshold: all 56; neuron clustering: none; power gap: [26, 33], with no frequency common to all three, so there is no key set to report; in the discrete-log basis they agree exactly. How much of the logits' variance is a function of a*b does not depend on the basis (0.973); only the sparsity does.

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

So what does make it answer 0? Blank the embedding row of one *nonzero* residue instead (zero it, or replace it with the mean row), for every one of the 112 nonzero residues in either input position, and score only pairs whose other input is untouched: the model answers 0 on 100.0% of 24,864 pairs when zeroed and 100.0% when averaged. Zero *both* inputs' rows (the averaged version was not tried) and it answers 59, never 0, on the 64 pairs tested (0% answer 0). Every such pair then presents the same input, so their agreeing is automatic; which class they agree on is the measurement. The real pair (0, 0), once 0's own row is blanked, becomes that same input and gives 59. So the rule is narrower than 'a blank input acts as zero': exactly one uninformative input gives 0, and two give a fixed non-zero class. Which weights implement this was not identified.
<!-- END:dlog -->

### A quadratic form

<!-- BEGIN:quadratic -->
`a^2 + ab + b^2` splits into linear factors over F_p exactly when p = 1 (mod 3). This pair of primes asks whether that matters. It is not a hypothesis from the literature: Furuta et al. (2024) call the form non-factorisable in the sense of not being expressible through (a +- b), and Doshi et al. (2024)'s Hypothesis 5.1 concerns forms h(g1(a) + g2(b)); neither is about splitting over F_p. Furuta et al. (2024) do train this form: at p = 97, where it splits, it groks from scratch only with a training fraction of at least 0.8, and at 0.5 it reaches 56% test accuracy (their Tables 1 and 5). The runs here use a training fraction of 0.5.

| run | p | form over F_p | test acc | held-out pairs whose transpose was trained on | acc on those | acc on the rest | chance |
|:--|--:|--:|--:|--:|--:|--:|--:|
| `Q_sqx_p59` | 2 mod 3 | irreducible | 0.4951 | 0.4842 | 1.0 | 0.0212 | 0.0169 |
| `Q_sqx_p61` | 1 mod 3 | splits | 0.4970 | 0.4793 | 1.0 | 0.0341 | 0.0164 |

**Neither generalises within 60,000 steps.** Both sit near 50% test accuracy, and the cause is not partial learning of the form. It is symmetric in a and b while the train/test split is over *ordered* pairs, so about half the held-out pairs have their transpose in the training set; the model gets essentially all of those right and is near chance on the rest. It has memorised the training table and learned that the table is symmetric. Since both primes sit at that floor, this pair says nothing either way about whether splitting over F_p matters.

Its *mistakes* are structured, though. `a^2 - ab + b^2` is the same form with the sign of one input flipped, and on the held-out pairs it cannot answer from memory, the model often predicts exactly that. Whether it does depends on whether a sign-flipped partner of the pair -- (a, -b), (-a, b) or their transposes, all of which share that value -- was in the training set:

| run | pairs with a sign-flipped partner trained | predicts a^2 - ab + b^2 | pairs without one | predicts a^2 - ab + b^2 | chance |
|:--|--:|--:|--:|--:|--:|
| `Q_sqx_p59` | 798 | 0.3935 | 64 | 0.0312 | 0.0169 |
| `Q_sqx_p61` | 875 | 0.3886 | 66 | 0.0303 | 0.0164 |

So these look like memorised answers retrieved for the wrong key. The embedding does place each residue near its negative (mean cosine similarity 0.58 at p = 59, 0.58 at p = 61; random pairs -0.04, -0.00). But that alone predicts that a *double* flip (-a, -b), whose value equals the true one, would be retrieved as readily and give the right answer; where only such a partner was trained, the model is right on 4% (n = 48), 2% (n = 46). Why single flips are retrieved and double flips are not was not established, and the without-partner groups are small. An earlier version read the plateau as a circuit that had 'lost the sign of b'; that explained neither the accuracy, which symmetry accounts for, nor the dependence of these errors on which partners were trained. Splitting on unordered pairs would remove the transpose effect only. Removing the sign-flip one as well needs whole orbits {(+-a, +-b), (+-b, +-a)} held out together, so that no held-out pair has a transpose or a sign-flipped partner in training; and with one seed per prime, even that would be a first look rather than a test.
<!-- END:quadratic -->

### Which correction mattered?

<!-- BEGIN:controls -->
The first run used float32 cross-entropy, no warmup, and an unembedding with a column for the '=' token that was cut off before the loss. The corrected configuration grokked at step 7,083, against 14,536 for the first run. Each control below changes one training choice, one run each; their step budgets also differ, which does not affect the grokking step (the learning rate is constant after warmup) but does change the step at which the mechanism is read.

| run | what differs | grokking step |
|:--|:--|--:|
| `main_add_s0` | add, float32 loss, no LR schedule, extra W_U column, seed 0 | 14,536 |
| `B_add_s0` | add, float64 loss, 10-step warmup, seed 0 | 7,083 |
| `C_add_f32` | add, float32 loss, 10-step warmup, seed 0 | 6,734 |
| `C_add_nowarm` | add, float64 loss, one-step warmup (zero learning rate on step 0), seed 0 | 9,764 |
| `B_add_s1` | add, float64 loss, 10-step warmup, seed 1 | 6,228 |

Changing only the loss precision moves the step by -349; reducing the warmup to a single step, whose learning rate is zero, by +2,681. With one run per arm, these have to be read against how far two runs can differ by chance. The only seed pair of the reference configuration itself (`B_add_s1` against `B_add_s0`) differs by 1.14x. The only configuration run with many seeds is a different one (16 seeds of `add` at p = 59, training fraction 0.5, grokking between step 760 and 2,266); there, 90% of seed pairs differ by a larger factor than the precision change, 54% by a larger factor than the warmup change, and 8% by one at least as large as the whole gap between the first run and the corrected one (2.05x). The control arms share their split and initial weights with the reference run, so independent seeds overstate their noise; but noise at p = 113, training fraction 0.3, was not measured beyond that one pair. These runs cannot say whether either change matters, or whether the gap itself is chance.

What else differs is narrower than it might seem. At step 0 the two runs share 8 of their 9 weight tensors bit for bit; only W_U is drawn differently, because it is drawn last and its shape changed. The extra column itself can matter only through floating-point rounding: it received no gradient, and at 133 of 133 checkpoints its weights equal, bit for bit, their step-0 values decayed by AdamW's weight decay alone. What remains is W_U's initial draw, the difference between no scheduler at all and the control's one zero-rate warmup step (never run on its own), an interaction between float32 and no warmup (never run jointly), and chance. These single runs cannot separate them, and an earlier claim that the original was simply 'a slow draw' went beyond them.
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

Within the one training fraction where every cell grokked (0.5), the grokking step falls monotonically as weight decay rises: 10,794, 3,100, 815, 344. Step times weight decay stays between 815 and 1,079 while weight decay spans a factor of 30; the delay after memorisation times weight decay goes from 1,068 to 616. Fitting log step against log weight decay over these 4 cells and the 3 seed-1 replicates that grokked (7 points) gives a slope of -1.07 counted from step 0 and -1.27 counted from the end of memorisation. Taking the noise on each point from the seed-to-seed spread measured at one of these cells (16 seeds at weight decay 1.0), and allowing for the delay's log being noisier where the delay is short, the 95% intervals are -1.30 to -0.85 and -1.61 to -0.93; both include -1. Part of the gap between the two slopes is arithmetic: a total falling exactly as 1/lambda, minus these memorisation steps, would give a delay slope of -1.17. The seed-1 run at weight decay 0.1 had not grokked by step 12,000 and is left out; a later grokking step there would make both slopes steeper. Liu et al. (2022) argued that the time to generalise goes like 1/lambda and showed it in a teacher-student model and, in their Appendix C, in this same one-layer transformer on (a+b) mod 113, per seed; Lyu et al. (2023) (arXiv:2311.18817) prove it in a large-initialisation limit, counting from initialisation; Truong et al. (2026a) and Truong et al. (2026c) derive and fit a law for the delay after memorisation under AdamW. Both measures agree with that dependence within the noise here; this reproduces it, it does not discover it.
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
Forecasting grokking from early signals is not new. Notsawo et al. (2023) (arXiv:2306.13253) predict *whether* it will occur from oscillations in the early training loss. At a fixed configuration, Truong et al. (2026a) (arXiv:2603.13331) predict each seed's delay from the parameter norm at memorisation, and Truong et al. (2026b) (arXiv:2604.13123) forecast it from the spectral entropy of the representation; Truong et al. (2026c) (arXiv:2605.18845) fit the delay across hyperparameter settings. Howe (2026) (arXiv:2609.19000) forecasts grokking per seed on held-out runs and, for induction heads, finds that an oracle-tuned loss rule ranks seeds as well as a mechanistic precursor but only as a nowcast -- a median lead of 50 steps against 975 -- and argues that rank correlation without lead time rewards nowcasts. The question here is a small instance of the same kind, and shares that limitation: within one configuration of this task, where only the random draw differs, how do a dozen signals, read at fixed early steps, compare at ranking runs by when they generalise?

16 runs share the task (`add`), modulus (59), training fraction (0.5), weight decay (1.0) and budget (6,000 steps). They grok between step 760 and 2,266. A reading taken after some run has grokked measures the outcome, so only readings before step 760 are used. `R_p59_wd1.0_f0.5_s1` has the same configuration but is left out: different budget and no trajectory analysis (12,000-step budget). `S_p59_wd1.0_f0.5` has the same configuration but is left out: different budget (20,000-step budget).

These are not forecasts made before anything happens. In this configuration test accuracy starts rising almost immediately: at step 200 it is already 15% to 49%; at step 500 it is already 30% to 59% (chance 1.7%). The readings rank how far along a transition already under way each run is.

Spearman correlation with the grokking step; positive means a higher reading goes with a later transition. `*` marks coefficients whose two-tailed permutation p-value survives Bonferroni over the 24 tests at the 2 reported steps (|rho| of at least 0.726). Rows marked `(final)` use the finished model's key frequencies and so borrow information from after the transition; their 'leak-free' counterparts use the frequencies each checkpoint would pick itself:

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

That family of 24 tests was chosen after the results were seen; an earlier version corrected over all 48 tests computed, at every step. Under that larger family (permutation threshold |rho| of at least 0.759) 7 signals survive at step 500 instead of 8, the excluded loss (train pairs), leak-free dropping out.

At step 200 nothing survives the correction. The largest point estimate among signals that use no information from after the transition is the weight norm (+0.67); the restricted loss (train pairs) (final) is higher (+0.72) but uses the final model's frequencies. None of these is reliable at this n.

At step 500, 8 signals survive. The strongest is the **test loss (visible from outside)** (rho +0.86), a plain loss that needs no mechanistic analysis. Resampling the 16 runs, the difference in |rho| between the best plain signal (test loss (visible from outside)) and the best mechanistic one that uses no future information (restricted loss (train pairs), leak-free) is +0.04, with a 95% bootstrap interval of -0.22 to +0.31. So in this one configuration neither kind ranks the runs detectably better than the other; the data cannot show that they are equal either, and by then every run's test accuracy is already rising, so these are nowcasts. An earlier version of this section claimed the visible quantity could not rank the runs; it had looked only at test accuracy.

Earlier versions of this table were also wrong for a mechanical reason: runs with a non-zero data seed were analysed on the seed-0 train/test split, which corrupted every split-dependent signal. The numbers above are from the corrected analysis.
<!-- END:prediction -->

---

## Method notes

**The instruments are calibrated, and the interventions are tested.** The
structural metrics, the power-gap key-frequency rule, the neuron assignment,
restricted and excluded loss and the readout budget are run on synthetic
logits or weights whose answer is known before they are trusted on a real
model (`tests/test_analysis.py`): the exact algorithm scores 1 on the (a+b)
tests, an (a-b) algorithm scores 0, random logits sit at chance, and a
softmax-invariant offset leaves the readout untouched. The weight interventions
have narrower checks: keeping every frequency, or mean-ablating no neuron,
leaves the model unchanged; deleting a frequency removes exactly its two
directions and leaves the rest untouched; mean-ablation leaves the kept
neurons' activations exactly as they were, on both forward paths; and removing
the MLP or a head removes exactly that component. The norm-restored embedding
surgery, the embedding-norm-threshold rule, the zero-element and discrete-log
edits, the per-frequency exclusion and the redundancy search have no test of
their own.

**Key frequencies are taken from the final checkpoint** for the progress
measures in section 4, because the question there is when the *final* circuit
starts to exist. For forecasting (section 7) that would leak information from
after the transition, so the frequency-dependent signals are also computed with
the frequencies each checkpoint picks for itself, and reported separately.

<!-- BEGIN:loss_precision -->
**Loss in float64, except in the first run and its float32 control.** In float32, `log_softmax` quantises at about 1.2e-07. After step 10,000, the training loss of the 3 float64-trained runs settles at a median of 8.6e-08 to 9.3e-08, near that level; that of the 2 float32-trained runs (`main_add_s0`, `C_add_f32`) stays around 1.2e-05 to 1.4e-05, about 127 times higher. Only `main_add_s0` also logged its loss in float32 (every logged value is exactly a float32 number), so only its curve is quantised near 1.2e-07; `C_add_f32` logged in float64, so its higher level is a property of the float32-trained model, not of the measurement. Section 5 compares the two precisions directly.
<!-- END:loss_precision -->

**Reproducible in distribution, not bit for bit.** With one thread identical
seeds give identical weights, and a test runs the real training loop twice to
assert it. With several threads PyTorch's CPU reductions do not fix their
summation order and runs diverge slightly, so a re-run lands near these
numbers rather than on them.

**Three defects, disclosed rather than hidden.**
- *A mean-ablation bug*, now fixed: the neuron mean-ablation added the removed
  neurons' average output to the position embedding -- the model's input --
  instead of to the MLP's output, so it passed through attention and the MLP a
  second time. Every neuron-ablation number in earlier versions was wrong, and
  a caveat written to explain their size explained an artefact. It was found
  only by the final audit, which reimplemented the ablation independently. The
  table in section 3 is from the corrected code, and a test now checks the
  kept neurons are untouched.
- *A data-seed bug*, now fixed: the analysis scripts once took the train/test
  split's seed as a command-line flag defaulting to 0, so runs with any other
  seed were analysed on the wrong split. Seventeen runs were affected, and
  several numbers in an earlier version of section 7 were wrong. The scripts
  now read the split from the run's own record and check it against a
  recorded fingerprint.
- *Unused output columns at p != 113.* The output width defaulted to 113, so
  every run at another modulus carries extra columns in its unembedding that
  are sliced off before the loss. They receive no gradient, only weight decay,
  and cannot affect predictions or the circuit except through floating-point
  rounding; they do add a near-constant term to the weight norm used in
  section 7. The default is fixed; those runs were not retrained, and their
  job files keep the old width so that they rebuild the runs as they were.

**Published numbers are quarantined** in `src/grokking/literature.py`, each
with its reference; values a source states but that should not be quoted as
fact are listed there with the reason.

---

## Reproducing

```bash
pip install -r requirements.txt   # see the file for the CPU-only torch wheel
python -m pytest tests/ -q        # includes re-rendering this README from results/

# rebuild the README, the figures and the page from the shipped results
./scripts/regenerate.sh --report

# rebuild every results file from the checkpoints, then all of the above
./scripts/regenerate.sh

# train a fresh run in the corrected configuration (B_add_s0's) and analyse it
python scripts/run_train.py --tag repro --op add --steps 40000
python scripts/analyze_run.py --tag repro
python scripts/analyze_mechanism.py --tag repro
```

**The checkpoints are not in the repository** (3.7 GB), so without retraining
only `--report` works: every table and figure can be re-rendered from
`results/`, but the analyses that produced `results/` need checkpoints.
`scripts/regenerate.sh` lists the exact command, flags and thread count behind
every derived results file (trajectory, mechanism, redundancy, quadratic, sweep,
controls, forecasting and split files), and each of those records the commit,
command line and thread count that produced it. The `*_history.json` files are
written by training and record each run's configuration; the headline run's
thread count is known only from its launch command.

**Every shipped run has a job file** in `scripts/jobs_*.json` that reproduces
its configuration. `./scripts/run_pipeline.sh jobs_sweep.json` trains the runs
in one, analyses their trajectories and re-renders the report; rebuilding the
files that combine runs needs every run's checkpoints (`regenerate.sh`). The
job files include the headline run (`jobs_main.json`: float32 loss, no
learning-rate schedule, an output column for the `=` token that is cut off
before the loss, and its original checkpoint schedule; it logged its losses in
float32, where a rerun logs them in float64) and the runs at p != 113, which
keep their 113-wide output. The train/test split is rebuilt from each run's
record and checked against `results/split_hashes.json`, so a torch version
that shuffled differently would stop the analysis rather than silently use
another split. `scripts/record_splits.py` wrote that file after checking, for
every run, that the final checkpoint's train and test loss on the rebuilt split
equal the losses logged during training, and that on a deliberately different
split they do not; it never replaces a recorded fingerprint. Runs with more than one thread are not
bit-for-bit reproducible (see Method notes), so a fresh run lands near these
numbers.

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
  runinfo.py       a run's configuration and split, read from the run itself
  literature.py    published values, quarantined, each with its reference
  analysis/        spectra, structure tests, progress measures, ablations,
                   crossing times, the discrete-log view
  viz/             one place where typography and colour are decided
  report.py        markdown table machinery
  report_blocks.py every number that reaches this file comes through here
scripts/
  run_train.py, run_many.py, jobs_*.json    training, and the job behind every run
  analyze_*.py                              analyses; each writes one kind of results file
  make_report.py, make_figures.py, export_web.py
  regenerate.sh, run_pipeline.sh            the whole chain, with the exact flags
results/           every measured number, with the provenance of each file
figures/           the figures in this README
web/               an interactive page built from the same results and text
tests/             correctness and calibration tests
```

---

## What is borrowed, and what went wrong

**Borrowed.** The phenomenon: [Power et al. 2022](https://arxiv.org/abs/2201.02177).
The mechanism, the neuron-clustering rule, restricted and excluded loss, the
Gini sparsity measure and the three-phase account:
[Nanda et al. 2023](https://arxiv.org/abs/2301.05217). Ablating
irreducible-representation subspaces of a group, including cyclic groups:
[Chughtai et al. 2023](https://arxiv.org/abs/2302.03025). The discrete-log
reduction for multiplication, the periodicity of trained MLPs in that basis,
and a learnability hypothesis for modular polynomials of the form
h(g1(a) + g2(b)): [Doshi et al. 2024](https://arxiv.org/abs/2406.03495). Which
modular polynomials grok, and at what training fraction:
[Furuta et al. 2024](https://arxiv.org/abs/2402.16726). Discrete-log sparsity in
a trained transformer: [Nguyen 2026](https://arxiv.org/abs/2606.17399).
Non-invertible elements as a separate region:
[Chen et al. 2026](https://arxiv.org/abs/2607.07066). The 1/lambda dependence of
grokking time on weight decay: [Liu et al. 2022](https://arxiv.org/abs/2210.01117),
who also test it on Nanda et al.'s transformer and this task,
[Lyu et al. 2023](https://arxiv.org/abs/2311.18817), and
[Truong et al. 2026a](https://arxiv.org/abs/2603.13331) and
[2026c](https://arxiv.org/abs/2605.18845). Predicting grokking from early
signals: [Notsawo et al. 2023](https://arxiv.org/abs/2306.13253); per seed at a
fixed configuration, [Truong et al. 2026a](https://arxiv.org/abs/2603.13331),
[2026b](https://arxiv.org/abs/2604.13123) and
[Howe 2026](https://arxiv.org/abs/2609.19000), who also shows, for induction
heads, that a loss rule can match a mechanistic precursor at ranking seeds only
as a nowcast. The comparison of early signals in section 7, read at fixed
steps, is a small instance of these questions, not a new one.

**Possibly new, and small.** Embedding-weight surgery in the discrete-log
basis, with a control, on a transformer trained with 0 included: Chughtai et al.
ablate irrep subspaces of activations and of the unembedding for group tasks,
Nguyen excludes 0 and does not ablate, and Chen et al. train with 0 included at
p = 113 but report correlational evidence only. The zero element: its small
embedding norm is not why the model answers 0, and blanking exactly one
nonzero input makes it answer 0 while zeroing both does not.

**Corrected along the way.** Much of what earlier versions claimed did not
survive review. The main items: an opening claim that the model sits at chance
on held-out pairs for fourteen thousand steps; a mechanism for a float32
readout difference, and then a wrong reason for dismissing it; a "sign-blind
circuit" reading of the quadratic form, an overcorrection that ignored its
structured errors, and an attribution of the factorisation question to papers
that do not pose it; a misreading of what Furuta et al. found for that form (it
does grok, at a larger training fraction than used here); three successive
accounts of the zero element; three identification rules called independent;
a reference run labelled with the wrong loss precision; a separation stated as
seven orders of magnitude; a phase-diagram sentence saying the grokking step
falls with weight decay next to numbers that rose; rule disagreement presented
as an independent prediction; the claim that the visible quantity cannot
forecast the transition, and later an unsupported claim that plain losses
forecast exactly as well as mechanistic ones; novelty claims contradicted by
published work; a citation with a wrong author list, another with given names
as surnames, and a paper cited for the opposite of its conclusion; forecasting
numbers corrupted by a data-seed bug and a double-counted signal; a
multiple-comparison family narrowed after the results were seen, without
saying so; reproduction instructions that did not reproduce the headline run,
claims that a file reproduced byte for byte when it did not, and results files
no shipped script could rebuild; a phase-timing comparison whose choice of end
point was unstated; phase boundaries in a figure placed at hand-picked
fractions of the grokking step; statements on the interactive page that the
data did not support; a comparison with Nanda et al.'s Gini that used a
different definition; the claim that the architecture follows theirs exactly
(their MLP has biases); the unused '=' column listed as a possible cause of a
timing gap; control conclusions drawn against too small an estimate of seed
noise; a weight-decay slope reported without uncertainty, and then called
steeper than 1/lambda on a noise model that was too narrow; missing credit to
Omnigrok; a bug in the neuron mean-ablation that made its whole table wrong;
and three inaccurate versions of the note below on how this was made. Each
section states the corrected version; the git history has the rest. Runs that
fail to grok are reported with their step budget.

**How this was made.** This repository was designed, implemented and written
by an AI system (Claude, by Anthropic) working as an agent for the repository's
owner. The owner asked for a substantial project, picked this topic from a list
of options the model proposed (the model had recommended it), and told it to
start; the scope, the experiment design, the code, the analysis and this
write-up are the model's. Every measured number was produced by running that
code on the owner's machine. Some corrections came out of the owner's
questions about whether the work was finished, some from the model's own
control experiments, and the rest from review by AI agents launched by the
model: three rounds of review of the write-up, each agent told to refute it; a
targeted fourth check; then, at the owner's request, an audit of the whole
process by eight agents that re-ran the analyses, re-ran the start of training,
re-evaluated the checkpoints, and read the code, the statistics, the cited
papers and the git history; and a final round that checked the fixes. No human
reviewed the analysis. The model made the corrections and reported them to the
owner.
