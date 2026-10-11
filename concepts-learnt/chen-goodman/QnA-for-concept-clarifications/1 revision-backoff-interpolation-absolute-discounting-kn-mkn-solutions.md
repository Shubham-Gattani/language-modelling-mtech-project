# Worked Solutions: Backoff, Interpolation, Absolute Discounting, Kneser–Ney, and Modified Kneser–Ney

This file contains detailed worked solutions to the ten revision questions in the companion worksheet.

Each solution follows an observable reasoning path:

1. Identify what the question is testing.
2. Write the relevant formula.
3. Substitute the given values.
4. Calculate one step at a time.
5. Check the result.
6. State the mental model.

The goal is not only to obtain the answer, but to understand what each number means and why the next smoothing method was needed.

---

## Useful counts from the shared dataset

The dataset is:

```text
S1:  <s> I like natural language processing </s>
S2:  <s> I like natural language models </s>
S3:  <s> We like natural language processing </s>
S4:  <s> We study natural language models </s>
S5:  <s> They study statistical language models </s>
S6:  <s> San Francisco is sunny </s>
S7:  <s> San Francisco is famous </s>
S8:  <s> of the language </s>
S9:  <s> in the language </s>
S10: <s> with the language </s>
```

Some counts used repeatedly are:

| History or n-gram | Count |
|---|---:|
| $c(\text{like})$ | 3 |
| $c(\text{like},\text{natural})$ | 3 |
| $c(\text{natural language})$ | 4 |
| $c(\text{natural language},\text{processing})$ | 2 |
| $c(\text{natural language},\text{models})$ | 2 |
| $c(\text{San})$ | 2 |
| $c(\text{San},\text{Francisco})$ | 2 |
| $c(\text{language})$ | 8 |

---

## Question 1 — Backoff: choosing the correct branch

### What this tests

The basic backoff decision:

```text
known history   → use the current-order distribution
unknown history → shorten the history and back off
```

The condition checks the history `big red`, not necessarily the complete candidate n-gram `big red fox`.

### Given

```text
α(fox | big red) = 0.60
γ(big red) = 0.40
P_smooth(fox | red) = 0.20
```

The rule is:

$$
p_{smooth}(w\mid h)
=
\begin{cases}
\alpha(w\mid h) & \text{if }c(h)>0\\[4pt]
\gamma(h)p_{smooth}(w\mid h') & \text{if }c(h)=0
\end{cases}
$$

### Case 1: `big red` was observed

Use the first branch:

$$
P_{smooth}(\text{fox}\mid\text{big red})
=\alpha(\text{fox}\mid\text{big red})
=0.60
$$

### Case 2: `big red` was unseen

Use the second branch:

$$
P_{smooth}(\text{fox}\mid\text{big red})
=0.40\times0.20
=0.08
$$

### Interpretation

In a backoff model, the lower-order probability does not contribute when the current history is observed. It is used only when that history is unseen.

### Mental model

Backoff is a route-selection mechanism:

```text
specific context available   → trust it
specific context unavailable  → ask a shorter, safer context
```

---

## Question 2 — Interpolation: both orders contribute

### What this tests

Interpolation blends the current-order and lower-order models instead of choosing only one.

### Given

```text
λ(big red) = 0.75
p_ML(fox | big red) = 0.60
p_smooth(fox | red) = 0.20
```

Use:

$$
p_{smooth}(w\mid h)
=
\lambda(h)p_{ML}(w\mid h)
+
\left(1-\lambda(h)\right)p_{smooth}(w\mid h')
$$

### Higher-order contribution

$$
0.75\times0.60=0.45
$$

### Lower-order contribution

The lower-order weight is:

$$
1-0.75=0.25
$$

Therefore:

$$
0.25\times0.20=0.05
$$

### Final answer

$$
P_{smooth}(\text{fox}\mid\text{big red})
=0.45+0.05=0.50
$$

The weights are normalized:

$$
0.75+0.25=1
$$

### Difference from backoff

Backoff would use (0.60) when `big red` is observed. Interpolation gives (0.50) because the lower-order model contributes (0.05) even though the current history exists.

### Mental model

```text
Backoff:       choose one level
Interpolation: mix levels
```

---

## Question 3 — Counting histories, bigrams, and trigrams

### What this tests

This distinguishes:

```text
c(h)       = count of the history
c(h,w)     = count of history followed by w
N₁₊(h,·)   = number of distinct words that follow h
```

The history is `natural language`.

### 1. Where does `natural language` occur?

It occurs in S1, S2, S3, and S4:

```text
S1: I like natural language processing
S2: I like natural language models
S3: We like natural language processing
S4: We study natural language models
```

Therefore:

$$
c(\text{natural language})=4
$$

### 2. Count of the bigram

The bigram appears once in each of the four sentences:

$$
c(\text{natural},\text{language})=4
$$

### 3. Why is the history count the denominator?

When predicting after `natural language`, there were four opportunities to observe a next word. Therefore the denominator is:

$$
c(\text{natural language})=4
$$

It answers:

> Out of all occurrences of this history, how many times did each candidate follow it?

### 4. Trigram counts

`processing` follows the history in S1 and S3:

$$
c(\text{natural language},\text{processing})=2
$$

`models` follows the history in S2 and S4:

$$
c(\text{natural language},\text{models})=2
$$

The counts account for all four history occurrences:

$$
4=2+2
$$

### 5. Distinct continuation count

The continuation list is:

```text
processing, models, processing, models
```

The distinct set is:

```text
{processing, models}
```

Therefore:

$$
N_{1+}(\text{natural language},\cdot)=2
$$

### 6. Observed trigram types

```text
(natural, language, processing)
(natural, language, models)
```

There are two types but four tokens.

### 7. Unseen trigram

`natural language statistical` does not occur:

$$
c(\text{natural language},\text{statistical})=0
$$

### Mental model

The history count is the denominator because it counts prediction opportunities. The complete n-gram count tells us what happened on those opportunities. The distinct continuation count asks how many different outcomes occurred.

---

## Question 4 — Absolute discounting

### What this tests

The complete mass-transfer process:

```text
discount observed evidence → collect removed mass → redistribute it
```

### Given

For history `like`:

$$
c(\text{like},\text{natural})=3,
\qquad
c(\text{like})=3,
\qquad
D=0.5
$$

The lower-order probabilities are:

| Word | $P_{uni}(w)$ |
|---|---:|
| natural | 0.30 |
| models | 0.20 |
| processing | 0.50 |

Use:

$$
P_{AD}(w\mid h)
=
\frac{\max(c(h,w)-D,0)}{c(h)}
+
\lambda(h)P_{uni}(w)
$$

and:

$$
\lambda(h)=\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

### 1. Distinct continuation count

Every occurrence after `like` is `natural`, so:

$$
N_{1+}(\text{like},\cdot)=1
$$

### 2. Discounted count

$$
3-0.5=2.5
$$

### 3. Direct probability

$$
\frac{2.5}{3}=0.833333\ldots
$$

### 4. Removed probability mass

The removed count is (0.5). Divide by the history count:

$$
\frac{0.5}{3}=0.166666\ldots
$$

### 5. Calculate λ

$$
\lambda(\text{like})
=\frac{0.5\times1}{3}
=\frac{0.5}{3}
\approx0.1667
$$

The interpolation weight equals the removed probability mass.

### 6. Calculate all three probabilities

For `natural`:

$$
P_{AD}(\text{natural}\mid\text{like})
=\frac{2.5}{3}+0.1667\times0.30
$$

$$
P_{AD}(\text{natural}\mid\text{like})
\approx0.8333+0.05=0.8833
$$

For unseen `models`:

$$
P_{AD}(\text{models}\mid\text{like})
=0+0.1667\times0.20
\approx0.0333
$$

For unseen `processing`:

$$
P_{AD}(\text{processing}\mid\text{like})
=0+0.1667\times0.50
\approx0.0833
$$

### 7. Normalization check

$$
0.8833+0.0333+0.0833
\approx0.9999\approx1
$$

Using exact reasoning, the direct mass is (2.5/3), and the lower-order mass is:

$$
\frac{0.5}{3}(0.30+0.20+0.50)=\frac{0.5}{3}
$$

So:

$$
\frac{2.5}{3}+\frac{0.5}{3}=1
$$

### 8. What discounting did

The count of `like natural` changed from 3 to 2.5. The removed count (0.5) became probability mass (0.5/3), which was distributed according to the lower-order model.

### Mental model

Absolute discounting creates a probability budget for unseen events by taking a fixed amount from observed events. The lower-order model decides how to spend that budget.

---

## Question 5 — Absolute discounting versus interpolation

### What this tests

This isolates interpolation from the details of how λ was estimated in Question 4.

### Given

$$
\lambda(\text{like})=0.75
$$

The higher-order model says:

```text
P_ML(natural | like) = 1.0
P_ML(models | like) = 0.0
P_ML(processing | like) = 0.0
```

The lower-order model says:

```text
P_lower(natural) = 0.30
P_lower(models) = 0.20
P_lower(processing) = 0.50
```

The lower-order weight is:

$$
1-\lambda=1-0.75=0.25
$$

### 1. `natural`

$$
P_{interp}(\text{natural}\mid\text{like})
=0.75\times1.0+0.25\times0.30
=0.825
$$

### 2. `models`

$$
P_{interp}(\text{models}\mid\text{like})
=0.75\times0+0.25\times0.20
=0.05
$$

### 3. `processing`

$$
P_{interp}(\text{processing}\mid\text{like})
=0.75\times0+0.25\times0.50
=0.125
$$

### 4. Normalization

$$
0.825+0.05+0.125=1.0
$$

### 5. Compare with Question 4

Question 4 gave:

$$
P_{AD}(\text{natural}\mid\text{like})\approx0.8833
$$

This question gives (0.825). The difference comes from the weights: Question 4 retained about (0.8333) direct mass, while this example assigns (0.75) to the higher-order model and (0.25) to the lower-order model.

### 6. Unseen continuations

Both methods give nonzero probability to unseen continuations because their lower-order distributions are nonzero.

Absolute discounting uses λ times the lower-order probability. Interpolation uses (1-λ) times the lower-order probability.

### 7. Two statements explained

“The higher-order model explains the observed continuation” means that the specific history `like` provides direct evidence for `natural`.

“The lower-order model is still allowed to contribute” means that interpolation does not discard broad evidence merely because the specific history exists.

### Mental model

```text
discounting   → how much probability mass should be reserved?
interpolation → how should higher and lower orders share probability?
```
---

## Question 6 — Kneser–Ney continuation counts

### What this tests

Kneser–Ney changes the lower-order question from “How many times did the word occur?” to “How many different histories preceded the word?”

### 1–2. Predecessors of `Francisco`

The distinct predecessor set is `{San}`:

$$
N_{1+}(\cdot,\text{Francisco})=1
$$

### 3–4. Predecessors of `language`

The distinct predecessor set is `{natural, statistical, the}`:

$$
N_{1+}(\cdot,\text{language})=3
$$

### 5–6. Predecessors of `models`

The bigrams ending in `models` are all `language models`. Thus the distinct immediate predecessor set is `{language}`. `natural` and `statistical` precede `language`, not `models`.

Therefore:

$$
N_{1+}(\cdot,\text{models})=1
$$

### 7. Distinct bigram types

Enumerating every distinct bigram type once gives a total of 30:

$$
N_{1+}(\cdot,\cdot)=30
$$

### 8. Continuation probabilities

For `Francisco`:

$$
P_{cont}(\text{Francisco})=\frac{1}{30}\approx0.0333
$$

For `language`:

$$
P_{cont}(\text{language})=\frac{3}{30}=0.10
$$

For `models`:

$$
P_{cont}(\text{models})=\frac{1}{30}\approx0.0333
$$

### 9–10. Interpretation

A word may occur many times because one phrase repeats. That proves the phrase is strong, but it does not prove the word is generally useful after many different histories.

`language` has the larger continuation probability because:

$$
N_{1+}(\cdot,\text{language})=3
>
N_{1+}(\cdot,\text{Francisco})=1
$$

### Mental model

Kneser–Ney rewards contextual versatility, not merely token popularity.

---

## Question 7 — Full Kneser–Ney bigram calculation

### What this tests

This combines direct evidence, discounting, removed mass, continuation probability, and interpolation.

### Given

$$
c(\text{San},\text{Francisco})=2,
\qquad
c(\text{San})=2,
\qquad
N_{1+}(\text{San},\cdot)=1,
\qquad
D=0.5
$$

### 1. Direct discounted contribution

$$
\frac{2-0.5}{2}=\frac{1.5}{2}=0.75
$$

### 2. Total removed mass

One distinct continuation loses (0.5) count units:

$$
\text{removed count}=0.5\times1=0.5
$$

Convert to probability mass:

$$
\text{removed mass}=\frac{0.5}{2}=0.25
$$

### 3. Calculate λ

$$
\lambda(\text{San})=\frac{0.5\times1}{2}=0.25
$$

### 4. Probability of `Francisco`

From Question 6:

$$
P_{cont}(\text{Francisco})=\frac{1}{30}
$$

Therefore:

$$
P_{KN}(\text{Francisco}\mid\text{San})
=0.75+0.25\times\frac{1}{30}
\approx0.758333
$$

### 5. Probability of unseen `language`

The direct term is zero. Since $$P_{cont}(\text{language})=3/30=0.1$$

$$
P_{KN}(\text{language}\mid\text{San})
=0+0.25\times0.1
=0.025
$$

### 6. General expression for other words

For every $$w\ne\text{Francisco}$$,the direct term is zero:

$$
P_{KN}(w\mid\text{San})=0.25P_{cont}(w)
$$

### 7. Normalization

The direct mass is (0.75), and the redistributed mass is:

$$
0.25\sum_wP_{cont}(w)=0.25\times1=0.25
$$

Therefore:

$$
\sum_wP_{KN}(w\mid\text{San})=0.75+0.25=1
$$

### 8. Why `Francisco` can be high after `San`

The high probability comes from the direct term (0.75). Its continuation probability affects only the small redistributed portion:

$$
\lambda(\text{San})P_{cont}(\text{Francisco})
=0.25\times\frac{1}{30}
$$

Thus, `San → Francisco` is a strong specific relationship, while `Francisco` is not a broadly versatile continuation word.

### Mental model

```text
direct term       = what this history specifically predicts
continuation term = how generally useful the candidate word is
```

---

## Question 8 — Recursive Kneser–Ney with a trigram history

### What this tests

This tests whether you can follow probability down the model hierarchy:

```text
trigram context → bigram context → continuation probability
```

### Given

$$
c(\text{natural language},\text{processing})=2,
\qquad
c(\text{natural language},\text{models})=2,
\qquad
c(\text{natural language})=4,
\qquad
N_{1+}(\text{natural language},\cdot)=2,
\qquad
D=0.5
$$

The lower-order values are supplied:

$$
P_{KN}(\text{processing}\mid\text{language})=0.19375
$$

$$
P_{KN}(\text{the}\mid\text{language})=0.01875
$$

### 1. Calculate λ

There are two distinct observed continuations, and each loses (D=0.5):

$$
\text{removed count}=0.5\times2=1
$$

Divide by the history count:

$$
\lambda(\text{natural language})=\frac{1}{4}=0.25
$$

### 2. Direct trigram contribution for `processing`

$$
\frac{2-0.5}{4}=\frac{1.5}{4}=0.375
$$

### 3. Complete probability of `processing`

The lower-order contribution is:

$$
0.25\times0.19375=0.0484375
$$

Therefore:

$$
P_{KN}(\text{processing}\mid\text{natural language})
=0.375+0.0484375
=0.4234375
$$

### 4. Complete probability of unseen `the`

The direct trigram term is zero:

$$
P_{KN}(\text{the}\mid\text{natural language})
=0+0.25\times0.01875
=0.0046875
$$

### 5. Recursive path

For the unseen trigram:

```text
P(the | natural language)
        ↓ unseen trigram
P(the | language) = 0.01875   [given]
        ↓ already includes its own lower-order smoothing
P_cont(the)
```

The trigram answer is not (0.01875) by itself. The trigram-level weight (0.25) must scale it.

### 6. Why use the bigram?

The longer context failed to provide evidence, but the shorter context may still provide useful evidence. Backing off loses specificity without discarding all information.

### 7. Where does removed mass go?

The two observed trigrams lose a total count of 1. Dividing by the history count 4 gives:

$$
\frac{1}{4}=0.25
$$

That (0.25) is redistributed according to $$P_{KN}(w\mid\text{language})$$

### Mental model

The lower-order model is the next rung of the same ladder. A trigram hands leftover probability to a bigram; the bigram eventually hands leftover probability to continuation counts.

---

## Question 9 — Modified Kneser–Ney with three discount buckets

### What this tests

Modified Kneser–Ney uses a different discount according to the observed count.

### Given

For history `<s>`:

$$
c(\langle s\rangle)=10,
\qquad
N_1=4,
\qquad
N_2=3,
\qquad
N_{3+}=0
$$

The discounts are:

$$
D_1=0.4,
\qquad
D_2=0.9,
\qquad
D_{3+}=1.3
$$

### 1. Count-one removed mass

$$
D_1N_1=0.4\times4=1.6
$$

### 2. Count-two removed mass

$$
D_2N_2=0.9\times3=2.7
$$

### 3. Calculate λ

There are no count-three-or-more continuations:

$$
\lambda(\langle s\rangle)
=\frac{D_1N_1+D_2N_2+D_{3+}N_{3+}}{c(\langle s\rangle)}
$$

$$
\lambda(\langle s\rangle)
=\frac{1.6+2.7+1.3\times0}{10}
=\frac{4.3}{10}=0.43
$$

### 4. Probability of `I`

`I` has count 2, so use (D_2=0.9):

$$
\text{direct contribution}=\frac{2-0.9}{10}=0.11
$$

The lower-order contribution is:

$$
0.43\times\frac{1}{30}\approx0.014333
$$

Therefore:

$$
P_{MKN}(I\mid\langle s\rangle)
=0.11+0.014333
\approx0.124333
$$

### 5. Probability of `They`

`They` has count 1, so use (D_1=0.4):

$$
\text{direct contribution}=\frac{1-0.4}{10}=0.06
$$

The lower-order contribution is again (0.43/30\approx0.014333):

$$
P_{MKN}(\text{They}\mid\langle s\rangle)
=0.06+0.014333
\approx0.074333
$$

### 6. Probability of unseen `language`

The direct term is zero. Since (P_{MKN}(\text{language})=3/30=0.1):

$$
P_{MKN}(\text{language}\mid\langle s\rangle)
=0.43\times0.1
=0.043
$$

### 7. Why use (D_2) for `I`?

Because:

$$
c(\langle s\rangle,\text{I})=2
$$

The count bucket determines the discount.

### 8. Why not (D(N_1+N_2))?

The two buckets surrender different amounts:

$$
D_1N_1+D_2N_2
$$

Using (D(N_1+N_2)) would incorrectly force count-one and count-two types to have the same discount.

### 9. Compare with Question 7

The structure stayed the same:

```text
discount → calculate removed mass → calculate λ → redistribute
```

The discount schedule changed:

```text
Question 7: one D
Question 9: D₁, D₂, D₃₊ selected by count class
```

### Mental model

Modified Kneser–Ney is Kneser–Ney with a more careful tax schedule. The lower-order destination is still the continuation model.

---

## Question 10 — Estimating MKN discounts and auditing your answer

### What this tests

This combines discount estimation, application, and independent correctness checks.

## Part A — Estimate the discounts

### Given

$$
n_1=10,
\qquad
n_2=4,
\qquad
n_3=2,
\qquad
n_4=1
$$

### 1. Calculate (Y)

$$
Y=\frac{n_1}{n_1+2n_2}
=\frac{10}{10+2(4)}
=\frac{10}{18}
\approx0.5555556
$$

### 2. Calculate (D_1)

$$
D_1=1-2Y\frac{n_2}{n_1}
$$

$$
D_1=1-2(0.5555556)\frac{4}{10}
\approx1-0.4444445
\approx0.5556
$$

### 3. Calculate (D_2)

$$
D_2=2-3Y\frac{n_3}{n_2}
$$

$$
D_2=2-3(0.5555556)\frac{2}{4}
\approx2-0.8333334
\approx1.1667
$$

### 4. Calculate (D_{3+})

$$
D_{3+}=3-4Y\frac{n_4}{n_3}
$$

$$
D_{3+}=3-4(0.5555556)\frac{1}{2}
\approx3-1.1111112
\approx1.8889
$$

## Part B — Apply and audit the discounts

Use $$(c(\langle s\rangle)=10), (N_1=4), (N_2=3), and (N_{3+}=0).$$

### 1. Total removed count

Count-one bucket:

$$
D_1N_1\approx0.5555556\times4=2.2222224
$$

Count-two bucket:

$$
D_2N_2\approx1.1666667\times3=3.5
$$

Count-three-or-more bucket:

$$
D_{3+}N_{3+}=1.8888889\times0=0
$$

Total:

$$
2.2222224+3.5=5.7222224
$$

### 2. Interpolation weight

$$
\lambda(\langle s\rangle)
=\frac{5.7222224}{10}
\approx0.5722222
$$

### 3. Direct contribution for a count-one word

$$
\frac{1-D_1}{10}
=\frac{1-0.5555556}{10}
\approx0.0444444
$$

### 4. Direct contribution for a count-two word

$$
\frac{2-D_2}{10}
=\frac{2-1.1666667}{10}
\approx0.0833333
$$

### 5. Complete probability for a count-one word

With lower-order probability (1/30):

$$
P_{MKN}(w\mid\langle s\rangle)
=0.0444444+0.5722222\times\frac{1}{30}
\approx0.0635185
$$

### 6. Complete probability for a count-two word

$$
P_{MKN}(w\mid\langle s\rangle)
=0.0833333+0.5722222\times\frac{1}{30}
\approx0.1024074
$$

## Part C — Correctness checklist

### Nonnegativity

Both direct contributions are nonnegative:

$$
1-D_1\approx0.4444>0
$$

$$
2-D_2\approx0.8333>0
$$

### Normalization

The retained direct count is:

$$
4(1-D_1)+3(2-D_2)
$$

$$
\approx4(0.4444444)+3(0.8333333)
\approx4.2777775
$$

Convert to probability mass:

$$
\frac{4.2777775}{10}\approx0.4277778
$$

Add redistributed mass:

$$
0.4277778+0.5722222=1.0000000
$$

### Why (D_2>1) can be valid

The discount is subtracted from count 2:

$$
2-D_2=2-1.1667=0.8333>0
$$

The discount itself need not be below 1. It must not remove more count than is available in the relevant count class.

### Why one fixed (D) is less flexible

A single discount forces count-one, count-two, and count-ten n-grams to surrender the same absolute amount. Modified Kneser–Ney estimates separate discounts because the count classes have different statistical reliability.

### Audit procedure

When a complicated smoothing calculation looks suspicious, check:

1. Did I use the correct history count?
2. Did I select the correct discount bucket?
3. Did I multiply each discount by the number of types in its bucket?
4. Does λ equal removed count divided by history count?
5. Does the lower-order distribution sum to one?
6. Do direct mass and redistributed mass sum to one?

### Mental model

The central audit identity is:

$$
\text{probability kept directly}
+
\text{probability redistributed}
=1
$$

If this fails, trace the removed count, history denominator, discount bucket, and lower-order normalization.

---

## Final mental model for the whole progression

### Backoff

If the current history is unknown, use a shorter history.

### Interpolation

Use current-order and lower-order information together.

### Absolute discounting

Take a fixed amount from observed n-grams and redistribute it.

### Kneser–Ney

Use distinct continuation contexts for the lower-order distribution.

### Modified Kneser–Ney

Use continuation contexts plus different discounts for different count classes.

The complete progression is:

$$
\text{specific evidence}
\xrightarrow{\text{discount}}
\text{reserved mass}
\xrightarrow{\text{lower-order model}}
\text{probability for unseen or uncertain events}
$$

The final mental model is:

| Method | Main question it answers |
|---|---|
| Backoff | Which order should I use? |
| Interpolation | How should orders share probability? |
| Discounting | How much mass should be reserved? |
| Kneser–Ney | What should the lower-order distribution mean? |
| Modified Kneser–Ney | Should every count class be discounted equally? |

