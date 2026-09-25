# Detailed Clarifications: Lambda, Discounting, and Probability-Mass Distribution

This note answers the doubts raised while solving the revision worksheet on:

```text
Backoff → Interpolation → Absolute Discounting → Kneser–Ney → Modified Kneser–Ney
```

The central theme is probability-mass movement. Almost every intimidating-looking formula is answering one of these questions:

1. How much probability should remain attached to the observed n-grams?
2. How much probability was removed from them?
3. How should the removed amount be distributed among possible next words?
4. Which lower-order distribution should perform that distribution?

---

## First, keep the assignment’s dataset and counts in view

The relevant dataset is:

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

Important counts:

$$
c(\text{like})=3
$$

$$
c(\text{like},\text{natural})=3
$$

$$
N_{1+}(\text{like},\cdot)=1
$$

$$
c(\text{San})=2
$$

$$
c(\text{San},\text{Francisco})=2
$$

$$
N_{1+}(\text{San},\cdot)=1
$$

$$
c(\text{natural language})=4
$$

$$
N_{1+}(\text{natural language},\cdot)=2
$$

The dataset has 30 distinct bigram types when `<s>` and `</s>` are included.

---

## Q1. How do we calculate γ(h) if it is not given?

### What γ means

In the generic backoff equation:

$$
p_{smooth}(w\mid h)
=
\begin{cases}
\alpha(w\mid h) & \text{if }c(h)>0\\[4pt]
\gamma(h)p_{smooth}(w\mid h') & \text{if }c(h)=0
\end{cases}
$$

γ(h) is a scaling factor for the lower-order distribution. Its job is to make the complete conditional distribution sum to one.

It is not usually an arbitrary probability assigned to the history. It is a correction factor that says:

> The higher-order part has already claimed some probability mass. Scale the lower-order probabilities so they receive exactly the remaining mass.

### General derivation

Let:

* $S(h)$ be the set of words whose n-grams with history $h$ are seen;
* $U(h)$ be the set of words whose n-grams with history $h$ are unseen;
* $\alpha(w \mid h)$ be the current-order probability for a seen word;
* $p_{\text{lower}}(w \mid h')$ be the lower-order probability.



The final distribution must satisfy:

$$
\sum_w p_{smooth}(w\mid h)=1
$$

For seen words, the backoff model uses $\alpha(w \mid h)$. For unseen words, it uses $\gamma(h)p_{lower}(w\mid h')$. Therefore:

$$
\sum_{w\in S(h)}\alpha(w\mid h)
+
\sum_{w\in U(h)}\gamma(h)p_{lower}(w\mid h')
=1
$$

Factor out γ:

$$
\sum_{w\in S(h)}\alpha(w\mid h)
+
\gamma(h)\sum_{w\in U(h)}p_{lower}(w\mid h')
=1
$$

Move the seen mass to the other side:

$$
\gamma(h)\sum_{w\in U(h)}p_{lower}(w\mid h')
=
1-
\sum_{w\in S(h)}\alpha(w\mid h)
$$

Finally:

$$
\boxed{
\gamma(h)
=
\frac{
1-\sum_{w\in S(h)}\alpha(w\mid h)
}{
\sum_{w\in U(h)}p_{lower}(w\mid h')
}
}
$$

This is the general backoff normalization formula.

### A tiny numerical example

Suppose the vocabulary is:

```text
{A, B, C}
```

Suppose after history (h):

```text
A is seen and α(A | h) = 0.50
B is seen and α(B | h) = 0.20
C is unseen
```

Suppose the lower-order model gives:

$$
p_{lower}(C)=0.50
$$

The current-order model has assigned:

$$
0.50+0.20=0.70
$$

So the remaining mass is:

$$
1-0.70=0.30
$$

We need:

$$
\gamma(h)\times0.50=0.30
$$

Therefore:

$$
\gamma(h)=\frac{0.30}{0.50}=0.60
$$

The final probability for unseen `C` is:

$$
p_{smooth}(C\mid h)=0.60\times0.50=0.30
$$

The complete distribution is:

$$
P(A)=0.50,
\qquad
P(B)=0.20,
\qquad
P(C)=0.30
$$

and:

$$
0.50+0.20+0.30=1
$$

### Why not simply set γ equal to the remaining mass?

You can do that only if the lower-order distribution has already been renormalized over the unseen words so that:

$$
\sum_{w\in U(h)}p_{lower}(w\mid h')=1
$$

In the example, the lower-order probability of unseen `C` was only (0.50), not 1. Therefore the scaling factor had to be (0.60), so that the product became the desired leftover mass (0.30).

### A common special case

Some descriptions define a lower-order distribution already normalized only over unseen words. Then:

$$
\gamma(h)=1-\sum_{w\in S(h)}\alpha(w\mid h)
$$

But the general formula must include the denominator because the lower-order probabilities may still contain mass assigned to seen words.

### Mental model

```text
γ(h) is not “probability of the history.”
γ(h) is the knob that scales the lower-order distribution
until the whole conditional distribution sums to one.
```

---

## Q2. How is λ calculated in interpolation if it is not given?

This question needs one important distinction:

> In generic interpolation, λ is a model weight that must be chosen or estimated. Normalization alone does not determine its value.

### Generic interpolation

The formula is:

$$
p(w\mid h)
=
\lambda(h)p_{high}(w\mid h)
+
\left(1-\lambda(h)\right)p_{low}(w\mid h')
$$

If both component distributions sum to one, then any:

$$
0\le\lambda(h)\le1
$$

will produce a normalized distribution:

$$
\sum_w p(w\mid h)
=
\lambda(h)\sum_w p_{high}(w\mid h)
+
(1-\lambda(h))\sum_w p_{low}(w\mid h')
$$

$$
=\lambda(h)\times1+(1-\lambda(h))\times1=1
$$

So normalization tells us that λ must be a valid mixture weight, but it does not tell us whether λ should be (0.4), (0.75), or (0.9).

### How is λ chosen in practice?

There are several possibilities.

#### 1. Fixed global hyperparameter

Choose one value for all histories:

$$
\lambda(h)=\lambda=0.75
$$

Try several values on held-out validation data and choose the one with the lowest perplexity.

The worksheet’s Question 2 does exactly this: it gives $\lambda=0.75$ so you can focus on understanding interpolation.

#### 2. History-dependent interpolation weight

Use a different value for each history. A reliable history can receive more higher-order weight; an unreliable history can receive more lower-order weight.

For example:

```text
frequent history  → larger λ for higher-order evidence
rare history      → smaller λ, more lower-order help
```

#### 3. Estimate λ using held-out likelihood or EM

For more general interpolation models such as Jelinek–Mercer smoothing, λ can be estimated by maximizing likelihood on held-out data. An iterative method such as EM may be used.

### Why did absolute discounting have an exact formula for λ?

Absolute discounting is more structured. It defines how much count is removed from every observed continuation. Therefore the lower-order mass is forced by conservation:

$$
\lambda(h)
=
\frac{\text{total removed count after }h}{c(h)}
$$

For a single discount (D):

$$
\lambda(h)
=
\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

So there are two different meanings of λ:

| Setting | Meaning of λ |
|---|---|
| Generic interpolation | Chosen/estimated mixture weight |
| Absolute discounting / KN / MKN | Lower-order mass forced by the amount discounted |

### Mental model

```text
Generic interpolation: λ is a modeling choice.
Discount-based interpolation: λ is an accounting result.
```

---

## Q4. How is $D$ calculated? Is it a hyperparameter or an exact formula?

### Short answer

It can be either, depending on the smoothing method and implementation:

1. A fixed hyperparameter chosen using held-out data.
2. An estimate calculated from count-of-counts statistics.
3. A family of discounts estimated separately for different count classes, as in MKN.

The assignment used:

$$
D=0.5
$$

as an illustrative given value. You were not expected to derive it in Question 4.

### Why is (D) usually a hyperparameter in a teaching example?

The purpose of Question 4 was to make the mass movement visible. If we also introduced the estimation of (D) at the same time, the main idea would be buried under another layer of statistics.

So the question gives (D=0.5), just as a physics exercise might give gravitational acceleration before teaching projectile motion.

### Estimating a single absolute-discounting (D)

A common count-of-counts estimate is:

$$
D\approx\frac{n_1}{n_1+2n_2}
$$

where:

- $(n_1)$ is the number of distinct n-gram types seen exactly once;
- $(n_2)$ is the number of distinct n-gram types seen exactly twice.

For example, if:

$$
n_1=10,
\qquad
n_2=5
$$

then:

$$
D\approx\frac{10}{10+2(5)}
=\frac{10}{20}
=0.5
$$

This is one reason (D=0.5) is a reasonable illustrative value.

The exact formula and estimation convention can vary by implementation and n-gram order. In practice, the value is often estimated separately for bigrams and trigrams.

### What is the intuition behind discounting?

The maximum-likelihood model uses all observed count mass:

$$
\sum_w\frac{c(h,w)}{c(h)}=1
$$

But unseen n-grams receive zero. Discounting changes the distribution to:

$$
\frac{c(h,w)-D}{c(h)}
$$

for observed n-grams, creating a positive leftover amount for unseen n-grams.

The model is saying:

> Observed counts are useful, but the finite training corpus may have failed to observe some events that are genuinely possible.

### Why does λ have the exact absolute-discounting formula?

Return to the assignment’s history `like`:

$$
c(\text{like})=3
$$

There is only one distinct continuation, `natural`:

$$
N_{1+}(\text{like},\cdot)=1
$$

The discount is:

$$
D=0.5
$$

Each observed continuation loses (0.5) count units. Since there is one continuation, total removed count is:

$$
0.5\times1=0.5
$$

Divide by the total history count (3):

$$
\lambda(\text{like})
=\frac{0.5}{3}
\approx0.1667
$$

The formula:

$$
\lambda(h)=\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

is therefore just:

$$
\lambda(h)
=\frac{\text{total removed count after }h}{\text{total count of }h}
$$

### The mass-flow diagram for `like`

```text
Before discounting:

history `like` has count 3
all 3 occurrences continue as `natural`

Probability mass:
natural = 3 / 3 = 1.0000

                    discount D = 0.5 count units
                                  ↓
After discounting:

direct count kept for natural = 3 - 0.5 = 2.5
direct mass kept             = 2.5 / 3 = 0.8333
mass released                = 0.5 / 3 = 0.1667

Released mass 0.1667 is distributed by P_uni:

natural    gets 0.1667 × 0.30 = 0.0500
models     gets 0.1667 × 0.20 = 0.0333
processing gets 0.1667 × 0.50 = 0.0833
```

Final distribution:

```text
natural    = direct 0.8333 + fallback 0.0500 = 0.8833
models     = direct 0      + fallback 0.0333 = 0.0333
processing = direct 0      + fallback 0.0833 = 0.0833
```

The total is:

$$
0.8833+0.0333+0.0833\approx1
$$

---

## What does the `1+` mean in $N_{1+}(\cdot,w)$?

The notation:

$$
N_{1+}(\cdot,w)
$$

means:

> the number of distinct histories (h) for which the bigram (h,w) has a count of at least 1.

The `1+` means “one or more.” It is not addition. It means:

$$
c(h,w)\ge1
$$

### Example: `Francisco`

In the dataset:

```text
San Francisco
San Francisco
```

The bigram type `San Francisco` has count 2. Its count is at least 1, so `San` is counted as a distinct predecessor.

There are no other predecessors:

$$
N_{1+}(\cdot,\text{Francisco})=1
$$

The repeated occurrence does not add another distinct predecessor.

### Why not $N_{2+}$ throughout?

Different thresholds answer different questions:

$$
N_{1+}(\cdot,w)
$$

asks:

> How many different histories have ever been observed before (w)?

Whereas:

$$
N_{2+}(\cdot,w)
$$

would ask:

> How many different histories have preceded (w) at least twice?

Those are not the same statistic.

For example, suppose:

```text
San Francisco       count 2
of Francisco        count 1
```

Then:

$$
N_{1+}(\cdot,\text{Francisco})=2
$$

because both `San` and `of` have appeared at least once.

But:

$$
N_{2+}(\cdot,\text{Francisco})=1
$$

because only `San` has appeared at least twice.

### Why does Kneser–Ney use (1+)?

Kneser–Ney wants to know whether a continuation relationship exists at all. A single observation is still evidence that the history can lead to the word.

The lower-order model is intended to measure contextual breadth:

```text
one observed predecessor counts as one possible context
repeated use of the same predecessor does not create a new context
```

### What do $N_2$ and $N_{3+}$ mean then?

They are still useful, but for different jobs.

For a fixed history (h):

$N_1(h,\cdot)$ counts continuation types seen exactly once.

$N_2(h,\cdot)$ counts continuation types seen exactly twice.

$N_{3+}(h,\cdot)$ counts continuation types seen at least three times.

Modified Kneser–Ney uses these history-specific buckets to calculate how much mass to remove:

$$
\lambda(h)
=
\frac{
D_1N_1(h,\cdot)
+D_2N_2(h,\cdot)
+D_{3+}N_{3+}(h,\cdot)
}{c(h)}
$$

So:

```text
N₁₊(·, w) → distinct predecessor count for the continuation distribution
N₁, N₂, N₃₊(h, ·) → count buckets for discounting after a fixed history
```

---

## Does the interpolation weight always equal removed probability mass?

The answer is: **yes in the discount-derived formulations, but not as a universal statement about every interpolation model.**

### In absolute discounting, KN, and MKN

When λ is defined as:

$$
\lambda(h)
=
\frac{\text{total removed count after }h}{c(h)}
$$

it equals the probability mass removed from the direct higher-order distribution.

For ordinary absolute discounting:

$$
\lambda(h)=\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

For Modified Kneser–Ney:

$$
\lambda(h)=
\frac{D_1N_1(h,\cdot)+D_2N_2(h,\cdot)+D_{3+}N_{3+}(h,\cdot)}{c(h)}
$$

In these methods:

```text
λ = mass released by discounting
```

### In generic interpolation

In a generic formula:

$$
p(w\mid h)
=
\lambda p_{high}(w\mid h)
+
(1-\lambda)p_{low}(w\mid h')
$$

λ is simply a mixture weight. It may be tuned on validation data. It does not necessarily come from subtracting a discount from observed counts.

For Question 5:

$$
\lambda=0.75
$$

means:

```text
75% weight for the higher-order distribution
25% weight for the lower-order distribution
```

It is not automatically saying that exactly 25% of a count was discounted.

### Another subtle point: γ versus leftover mass

In a backoff formula:

$$
\gamma(h)p_{lower}(w\mid h')
$$

the **total probability mass delivered by the lower-order branch** equals the leftover mass. But γ itself may be larger or smaller than that mass because it may also correct for lower-order probability assigned to seen words.

The safe statement is:

```text
discount-derived λ → equals removed probability mass
generic interpolation λ → model mixture weight
backoff γ          → scaling factor whose product has the leftover mass
```

---

## Q7. What is “total removed mass”?

Use the assignment’s history `San`:

$$
c(\text{San})=2
$$

and:

$$
c(\text{San},\text{Francisco})=2
$$

There is one distinct continuation type:

$$
N_{1+}(\text{San},\cdot)=1
$$

Use $D=0.5$.

### Count view

Before discounting:

```text
San → Francisco has count 2
```

Discount removes (0.5) count units:

$$
2\longrightarrow2-0.5=1.5
$$

So the total removed count is:

$$
D\times N_{1+}(\text{San},\cdot)
=0.5\times1
=0.5
$$

> psst: “Count units” means raw occurrence-count units before dividing by the history count. Example: if `like → natural` occurred 3 times and \(D=0.5\), discounting changes its count from \(3\) to \(2.5\) count units.

### Probability view

The history has total count 2. Therefore one count unit corresponds to (1/2) of the history probability mass. The removed (0.5) count units correspond to:

$$
\frac{0.5}{2}=0.25
$$

That is the total removed probability mass.

### Diagram

```text
History count after San = 2

Before discounting:
Francisco direct mass = 2 / 2 = 1.00

Remove D = 0.5 count units:
Francisco direct count = 2 - 0.5 = 1.5
Francisco direct mass  = 1.5 / 2 = 0.75

Released probability mass:
1.00 - 0.75 = 0.25
```

That (0.25) is the “total removed mass” in Question 7.

---

## Q7, Part 3. How was λ calculated?

For ordinary Kneser–Ney with a single $D$:

$$
\lambda(h)=\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

Substitute the `San` values:

$$
\lambda(\text{San})
=\frac{0.5\times1}{2}
=\frac{0.5}{2}
=0.25
$$

Notice that λ equals the removed probability mass:

$$
\lambda(\text{San})=0.25
$$

This is not a coincidence. The formula was derived precisely to convert the removed count into probability mass.

### Why does $N_{1+}(h,\cdot)$ appear?

Because (D) is removed once for every distinct observed continuation type after (h).

If `San` were followed by three different words, each with some count, then the total removed count would be:

$$
D+D+D=3D
$$

and:

$$
\lambda(\text{San})=\frac{3D}{c(\text{San})}
$$

That is exactly:

$$
\frac{D\,N_{1+}(\text{San},\cdot)}{c(\text{San})}
$$

---

## Q7, Part 7. How was normalization calculated?

The Kneser–Ney formula for `San` is:

$$
P_{KN}(w\mid\text{San})
=
\frac{\max(c(\text{San},w)-0.5,0)}{2}
+
0.25P_{cont}(w)
$$

Only `Francisco` has a direct observed count.

### Direct mass

For `Francisco`:

$$
\frac{2-0.5}{2}=0.75
$$

For every other word, the direct term is zero:

$$
\frac{\max(0-0.5,0)}{2}=0
$$

So the sum of all direct terms is:

$$
\sum_w\text{direct}(w)=0.75
$$

### Redistributed mass

The lower-order component sums over all words as:

$$
\sum_w0.25P_{cont}(w)
=0.25\sum_wP_{cont}(w)
$$

The continuation distribution is normalized:

$$
\sum_wP_{cont}(w)=1
$$

Therefore:

$$
\sum_w0.25P_{cont}(w)=0.25
$$

### Total

Add direct and redistributed mass:

$$
\sum_wP_{KN}(w\mid\text{San})
=0.75+0.25
=1
$$

### Why do we not calculate every vocabulary word individually?

Because normalization can be proved using the fact that the lower-order distribution already sums to one. We only calculate one example such as `language` to see how an individual unseen word receives mass:

$$
P_{KN}(\text{language}\mid\text{San})
=0.25\times\frac{3}{30}
=0.025
$$

But the entire lower-order distribution receives the full (0.25) mass collectively.

### Mass diagram

```text
San-conditioned probability mass = 1.00 (means all probabilities for possible next words after San must sum to 1)

direct observed mass:
    Francisco = 0.75 (the direct discounted contribution)

c(San, Francisco) - D
---------------------- = (2 - 0.5) / 2 = 0.75
       c(San)


remaining mass:
    0.25
    ├── language receives 0.25 × 3/30 = 0.025
    ├── Francisco receives 0.25 × 1/30 as continuation mass
    └── every other word receives 0.25 × its P_cont(w)

total = 0.75 + all lower-order pieces = 1.00
```

---

## Q8. How was λ calculated in Part 1?

The history is `natural language`.

The assignment gives:

$$
c(\text{natural language})=4
$$

The two distinct observed continuations are `processing` and `models`, so:

$$
N_{1+}(\text{natural language},\cdot)=2
$$

Each loses (D=0.5) count units:

$$
\text{total removed count}=0.5\times2=1
$$

### Converting Removed Count into Probability Mass

For the history `natural language`:

$$
c(\text{natural language})=4
$$

This means the history appeared **4 times**, so its total conditional probability mass corresponds to 4 count units.

There are 2 distinct continuations, and each loses:

$$
D=0.5
$$

Therefore, the total removed count is:

$$
\text{Removed count}
=
2\times0.5
=
1
$$

To convert this removed count into probability mass, divide by the history count:

$$
\lambda(\text{natural language})
=
\frac{\text{Removed count}}{\text{History count}}
=
\frac{1}{4}
=
0.25
$$

Therefore:

- Direct higher-order mass:

  $$
  1-0.25=0.75
  $$

- Mass passed to the lower-order model:

  $$
  \lambda(\text{natural language})=0.25
  $$

So, \(25\%\) of the probability mass is redistributed through the lower-order model.

### Independent direct-mass check

Each observed trigram has count 2, so each keeps:

$$
2-0.5=1.5
$$

The total direct count kept is:

$$
1.5+1.5=3
$$

Therefore the direct mass is:

$$
\frac{3}{4}=0.75
$$

The released mass is:

$$
1-0.75=0.25
$$

This independently confirms λ.

### Diagram

```text
natural language has 4 total occurrences
processing count 2 → keep 1.5
models count 2     → keep 1.5
direct count kept  → 3
direct mass        → 3 / 4 = 0.75
released mass      → 1 − 0.75 = 0.25 = λ
```

---

## Q8, Part 6. What does “backing off loses specificity without discarding all information” mean?

Suppose we want:

$$
P(\text{the}\mid\text{natural language})
$$

The full context is the two-word history `natural language`. Since the trigram `natural language the` was unseen, the model shortens the history:

```text
natural language → language
```

It asks:

$$
P(\text{the}\mid\text{language})
$$

### Losing specificity

The shorter history no longer distinguishes whether `language` was preceded by `natural`, `statistical`, or `the`. Information about the older word `natural` has been discarded.

### Not discarding all information

The final word `language` remains. The model is not predicting after an arbitrary word; it is still predicting after `language`.

So backing off is a controlled loss of detail:

```text
full context unavailable → remove the oldest condition
shorter context retained  → keep whatever evidence remains useful
```

The recursive path is:

$$
P(\text{the}\mid\text{natural language})
\rightarrow
P(\text{the}\mid\text{language})
\rightarrow
P_{cont}(\text{the})
$$

### Mental model

Backing off is like cropping a photograph. You lose detail, but you do not throw away the entire image.

---

## Q9. Why is $D_{3+}>D_2>D_1$ in the question? Is it always true?

The assignment gives:

$$
D_1=0.4,
\qquad
D_2=0.9,
\qquad
D_{3+}=1.3
$$

So in this illustrative example:

$$
D_{3+}>D_2>D_1
$$

### Is this a universal theorem?

No. It is not guaranteed for every dataset or implementation. The values are estimated from frequency-of-frequencies statistics and may vary by corpus and n-gram order.

The common pattern is often increasing discounts, but the method does not fundamentally require a strict ordering.

### Why can larger counts have larger absolute discounts?

The discounts are absolute amounts of count mass, not reliability scores. A larger-count class may surrender a larger absolute amount while still keeping substantial count mass.

For the assignment’s values:

$$
1-D_1=1-0.4=0.6
$$

$$
2-D_2=2-0.9=1.1
$$

The discount for count 2 is larger than the discount for count 1, but the remaining count for the count-2 type is also larger.

The important safety condition is:

$$
c-D(c)\ge0
$$

The formula also includes:

$$
\max(c-D(c),0)
$$

### Do not interpret the ordering incorrectly

The ordering does not mean:

> A count-three n-gram is always less reliable than a count-one n-gram.

The discounts are estimated amounts removed from count classes. They are not probabilities and not reliability labels.

---

## Q9, Part 3. What is $λ(\langle s\rangle)$?

The notation:

$$
\lambda(\langle s\rangle)
$$

means:

> the lower-order interpolation weight for the history `<s>`.

It is not the probability of `<s>`. It is the same lambda function applied to a particular history.

The model is predicting the first word of a sentence:

```text
<s> ___
```

The assignment gives:

$$
c(\langle s\rangle)=10,
\qquad
N_1=4,
\qquad
N_2=3,
\qquad
N_{3+}=0
$$

The total removed count is:

$$
D_1N_1+D_2N_2+D_{3+}N_{3+}
=0.4\times4+0.9\times3+1.3\times0
=1.6+2.7
=4.3
$$

Divide by the history count:

$$
\lambda(\langle s\rangle)=\frac{4.3}{10}=0.43
$$

Interpretation:

```text
43% of the probability mass is passed to the lower-order model
57% remains in direct higher-order contributions
```

The direct-mass check is:

$$
1-0.43=0.57
$$

---

## Q10. What does “frequency-of-frequencies statistics for one n-gram order” mean?

### Frequency-of-frequencies

An ordinary n-gram count asks:

> How many times did this particular n-gram occur?

Frequency-of-frequencies asks:

> How many different n-gram types have each possible count?

If seven distinct n-gram types have counts:

$$
1,1,1,2,2,3,4
$$

then:

$$
n_1=3,
\qquad
n_2=2,
\qquad
n_3=1,
\qquad
n_4=1
$$

The distinction is:

```text
c(ngram) = count of one selected n-gram type
n_r       = number of n-gram types whose count equals r
```

### Meaning of “one n-gram order”

Do not mix bigram and trigram types when calculating $n_r$.

For bigrams, create a table of bigram types and their counts. Calculate $n_1,n_2,n_3,n_4$ from that table.

For trigrams, create a separate table and calculate a separate set of statistics.

Conceptually:

$$
\text{bigram table}
\rightarrow
n_1^{(2)},n_2^{(2)},n_3^{(2)},\ldots
$$

$$
\text{trigram table}
\rightarrow
n_1^{(3)},n_2^{(3)},n_3^{(3)},\ldots
$$

The superscripts only label the order.

### The assignment’s statistics

The assignment gives:

$$
n_1=10,
\qquad
n_2=4,
\qquad
n_3=2,
\qquad
n_4=1
$$

These are synthetic statistics for one chosen n-gram order. They are not four counts belonging to one history.

First:

$$
Y=\frac{10}{10+2(4)}=\frac{10}{18}\approx0.5556
$$

Then:

$$
D_1=1-2Y\frac{n_2}{n_1}\approx0.5556
$$

$$
D_2=2-3Y\frac{n_3}{n_2}\approx1.1667
$$

$$
D_{3+}=3-4Y\frac{n_4}{n_3}\approx1.8889
$$

### Do not confuse the symbols

| Symbol | Meaning |
|---|---|
| $n_r$ | Number of all n-gram types in one order with count exactly $r$ |
| $N_r(h,\cdot)$ | Number of continuation types with count $r$ after one fixed history $h$ |
| $N_{1+}(\cdot,w)$ | Number of distinct histories that precede a fixed word $w$ at least once |

---

## The central issue: how exactly is probability mass distributed?

This is the most important picture in the entire assignment.

### Start with one history

Take the history `like`:

$$
c(\text{like})=3
$$

All three occurrences continue as `natural`, so before smoothing:

$$
P_{ML}(\text{natural}\mid\text{like})=\frac{3}{3}=1
$$

All unseen continuations have probability zero.

### Discounting creates a hole

With $D=0.5$:

$$
\text{count kept}=3-0.5=2.5
$$

Direct mass kept:

$$
\frac{2.5}{3}=0.8333
$$

Released mass:

$$
1-0.8333=0.1667
$$

The distribution now has a probability hole of (0.1667). That hole must be filled.

### The lower-order distribution fills the hole

The lower-order probabilities are:

```text
natural    = 0.30
models     = 0.20
processing = 0.50
```

Multiply each by the released mass:

$$
\text{natural fallback}=0.1667\times0.30=0.0500
$$

$$
\text{models fallback}=0.1667\times0.20=0.0333
$$

$$
\text{processing fallback}=0.1667\times0.50=0.0833
$$

Attach the direct contribution where one exists:

```text
natural    = 0.8333 + 0.0500 = 0.8833
models     = 0      + 0.0333 = 0.0333
processing = 0      + 0.0833 = 0.0833
```

### Mass-flow diagram

```text
observed count after `like` = 3
            │
     subtract D = 0.5
            │
    ┌───────┴────────┐
    │                │
direct count kept    count released
    2.5 units        0.5 units
    │                │
    │                └── divide by c(like)=3 → mass 0.1667
    └── divide by 3 → direct mass 0.8333

released mass 0.1667 × lower-order distribution
    ├── natural:    0.1667 × 0.30 = 0.0500
    ├── models:     0.1667 × 0.20 = 0.0333
    └── processing: 0.1667 × 0.50 = 0.0833
```

The lower-order distribution does not create extra mass. It allocates the mass that discounting released.

### Why must the lower-order distribution sum to one?

Because it is being used as an allocation pattern:

$$
\sum_w\lambda P_{lower}(w)
=\lambda\sum_wP_{lower}(w)
=\lambda\times1
=\lambda
$$

If the lower-order distribution summed to (1.4), the model would distribute (1.4\lambda), creating too much mass. If it summed to (0.7), some released mass would remain unallocated.

### The same picture for Kneser–Ney

For `San`:

$$
c(\text{San})=2,
\qquad
c(\text{San},\text{Francisco})=2
$$

With (D=0.5):

```text
direct count kept = 2 − 0.5 = 1.5
direct mass       = 1.5 / 2 = 0.75
released mass     = 0.5 / 2 = 0.25
```

The mass mechanism is unchanged. Only the lower-order allocation pattern changes:

```text
Absolute discounting → ordinary unigram preference
Kneser–Ney          → continuation preference
```

For example:

$$
P_{cont}(\text{language})=\frac{3}{30}=0.1
$$

so `language` receives:

$$
0.25\times0.1=0.025
$$

### The same picture for MKN

For `<s>`:

$$
c(\langle s\rangle)=10
$$

with four count-one types and three count-two types:

$$
\text{released count}=4(0.4)+3(0.9)=1.6+2.7=4.3
$$

Therefore:

$$
\lambda(\langle s\rangle)=\frac{4.3}{10}=0.43
$$

Again:

$$
\text{direct mass kept}+\text{lower-order mass}=0.57+0.43=1
$$

The only new element is that different count buckets release different amounts.

---

## One master formula to remember

For a seen n-gram:

$$
P(w\mid h)
=
\underbrace{\text{discounted direct contribution}}_{\text{specific evidence}}
+
\underbrace{\text{released mass}\times\text{lower-order distribution}}_{\text{generalization}}
$$

Absolute discounting releases:

$$
\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

MKN releases:

$$
\frac{D_1N_1(h,\cdot)+D_2N_2(h,\cdot)+D_{3+}N_{3+}(h,\cdot)}{c(h)}
$$

The final mental model is:

```text
higher-order model:
    “What did this specific context observe?”

discounting:
    “Do not treat the observed sample as perfectly complete.”

lower-order model:
    “How should the released probability be shared?”

normalization:
    “Make sure every unit of probability is accounted for exactly once.”
```
