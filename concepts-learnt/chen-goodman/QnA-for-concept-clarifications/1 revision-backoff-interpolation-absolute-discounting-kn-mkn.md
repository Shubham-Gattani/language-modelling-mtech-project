# Revision Worksheet: Backoff, Interpolation, Absolute Discounting, Kneser–Ney, and Modified Kneser–Ney

This worksheet revises the concepts studied so far in a deliberate progression:

```text
backoff → interpolation → absolute discounting → Kneser–Ney → Modified Kneser–Ney
```

There are no solutions in this file. Solve each question on paper, showing every intermediate calculation.

The questions are designed so that, after solving them, you should be able to explain:

- what discounting means;
- why probability mass is removed from observed n-grams;
- how the removed mass is redistributed;
- how to count histories, bigrams, and trigrams;
- how to calculate interpolation/backoff weights such as $\lambda(h)$;
- how ordinary unigram counts differ from Kneser–Ney continuation counts;
- how Modified Kneser–Ney chooses among $D_1$, $D_2$, and $D_{3+}$;
- how to check that a smoothed probability distribution is valid.

---

## Shared hypothetical dataset

Use the following ten training sentences for Questions 3–10.

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

### Counting conventions

Use these conventions unless a question explicitly overrides them:

1. `<s>` and `</s>` are ordinary vocabulary symbols.
2. Include boundary symbols when counting bigrams and trigrams.
3. A bigram is an ordered pair of adjacent tokens.
4. A trigram is an ordered triple of adjacent tokens.
5. `c(h)` means the number of times history $h$ occurs as a context.
6. `c(h,w)` means the number of times history $h$ is followed by word $w$.
7. For Kneser–Ney continuation counts, count distinct types, not repeated tokens.
8. Unless stated otherwise, use interpolated smoothing.

### Symbols used in the worksheet

For a history $h$:

$$
P(w\mid h)
$$

means the probability of the next word $w$ after $h$.

The number of distinct observed continuations after (h) is:

$$
N_{1+}(h,\cdot)
$$

The number of distinct histories that precede (w) is:

$$
N_{1+}(\cdot,w)
$$

The total number of distinct bigram types is:

$$
N_{1+}(\cdot,\cdot)
$$

For Questions 6–10, use the continuation probability:

$$
P_{cont}(w)
=
\frac{N_{1+}(\cdot,w)}{N_{1+}(\cdot,\cdot)}
$$

---

## Question 1 — Backoff: choosing the correct branch

Consider a trigram model predicting the next word after a two-word history.

### Given

For the candidate word `fox`, suppose:

```text
α(fox | big red) = 0.60
γ(big red) = 0.40
P_smooth(fox | red) = 0.20
```

The model uses the backoff rule:

$$
p_{smooth}(w\mid h)
=
\begin{cases}
\alpha(w\mid h) & \text{if }c(h)>0\\[4pt]
\gamma(h)p_{smooth}(w\mid h') & \text{if }c(h)=0
\end{cases}
$$

### Calculate and explain

1. If `big red` was observed, calculate $P_{smooth}(\text{fox}\mid\text{big red})$.
2. If `big red` was never observed, calculate $P_{smooth}(\text{fox}\mid\text{big red})$ using backoff.
3. Explain in words why the model uses different branches in the two cases.
4. Does the lower-order probability contribute in the first case? Explain your answer.

### What this tests

Whether you can distinguish “the current history exists” from “the complete candidate n-gram exists.”

---

## Question 2 — Interpolation: both orders contribute

Use the same history and candidate word as in Question 1, but now use interpolation:

$$
p_{smooth}(w\mid h)
=
\lambda(h)p_{ML}(w\mid h)
+
\left(1-\lambda(h)\right)p_{smooth}(w\mid h')
$$

### Given

```text
λ(big red) = 0.75
p_ML(fox | big red) = 0.60
p_smooth(fox | red) = 0.20
```

### Calculate and explain

1. Calculate the higher-order contribution.
2. Calculate the lower-order contribution.
3. Calculate $P_{smooth}(\text{fox}\mid\text{big red})$.
4. Verify that the two interpolation weights sum to one.
5. Explain the key behavioral difference between this result and Question 1 when `big red` is observed.

### What this tests

Whether you understand that interpolation uses lower-order information even when the current history has been observed.

---

## Question 3 — Counting histories, bigrams, and trigrams

Use the shared dataset.

### Given

Focus on the history:

```text
natural language
```

and the candidate next words `processing` and `models`.

### Calculate and explain

1. List every sentence in which the bigram `natural language` appears.
2. Calculate:

   $$
   c(\text{natural},\text{language})
   $$

3. Calculate:

   $$
   c(\text{natural language})
   $$

   Explain why this history count is the denominator when predicting the next word.
4. Calculate:

   $$
   c(\text{natural language},\text{processing})
   $$

   and:

   $$
   c(\text{natural language},\text{models})
   $$
5. Calculate $N_{1+}(\text{natural language},\cdot)$.
6. Write the complete set of observed trigrams beginning with `natural language`.
7. Is `natural language statistical` seen? What is its count?

### What this tests

Whether you can move carefully between:

```text
history count      c(h)
bigram count       c(w_{i-1}, w_i)
trigram count      c(w_{i-2}, w_{i-1}, w_i)
distinct continuation count  N₁₊(h, ·)
```

---

## Question 4 — Absolute discounting: what does “discount” mean?

Use the history:

```text
like ___
```

### Given

From the shared dataset, use:

```text
c(like, natural) = 3
c(like) = 3
D = 0.5
```

For this question, use the following lower-order unigram distribution over the only three candidate words:

| Candidate word $w$ | $P_{uni}(w)$ |
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
\lambda(h)
=
\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

### Calculate and explain

1. Calculate $N_{1+}(\text{like},\cdot)$.
2. Calculate the discounted count for `natural`.
3. Calculate the direct probability contribution for `natural`.
4. Calculate the probability mass removed by discounting.
5. Calculate $\lambda(\text{like})$.
6. Calculate all three smoothed probabilities:

   $$
   P_{AD}(\text{natural}\mid\text{like})
   $$

   $$
   P_{AD}(\text{models}\mid\text{like})
   $$

   $$
   P_{AD}(\text{processing}\mid\text{like})
   $$

7. Verify that these three probabilities sum to one.
8. Explain exactly what the discount $D=0.5$ did to the observed count of `natural` and where the removed mass went.

### What this tests

Whether you understand discounting as moving count/probability mass, rather than simply “making a count smaller.”

---

## Question 5 — Absolute discounting versus interpolation

Use the same history `like` and the same lower-order distribution from Question 4.

### Given

Use the interpolation model:

$$
P_{interp}(w\mid h)
=
\lambda(h)P_{ML}(w\mid h)
+
\left(1-\lambda(h)\right)P_{lower}(w)
$$

For this question:

```text
λ(like) = 0.75
P_ML(natural | like) = 1.0
P_ML(models | like) = 0.0
P_ML(processing | like) = 0.0
```

Use the same lower-order probabilities:

```text
P_lower(natural) = 0.30
P_lower(models) = 0.20
P_lower(processing) = 0.50
```

### Calculate and explain

1. Calculate $P_{interp}(\text{natural}\mid\text{like})$.
2. Calculate $P_{interp}(\text{models}\mid\text{like})$.
3. Calculate $P_{interp}(\text{processing}\mid\text{like})$.
4. Verify that the three probabilities sum to one.
5. Compare the probability assigned to `natural` here with the probability assigned in Question 4.
6. In this example, which model gives nonzero probability to unseen continuations? Explain why both methods can do so.
7. Explain the difference between these statements:

   ```text
   “The higher-order model explains the observed continuation.”
   “The lower-order model is still allowed to contribute.”
   ```

### What this tests

Whether you can separate the idea of discounting from the idea of backoff versus interpolation.

---

## Question 6 — Kneser–Ney continuation counts

Now replace the ordinary unigram distribution with a Kneser–Ney continuation distribution.

### Given

Use the shared dataset and count every distinct bigram type, including boundary bigrams.

The continuation probability is:

$$
P_{cont}(w)
=
\frac{N_{1+}(\cdot,w)}{N_{1+}(\cdot,\cdot)}
$$

### Calculate and explain

1. List the distinct words that precede `Francisco`.
2. Calculate:

   $$
   N_{1+}(\cdot,\text{Francisco})
   $$

3. List the distinct words that precede `language`.
4. Calculate:

   $$
   N_{1+}(\cdot,\text{language})
   $$

5. List the distinct words that precede `models`.
6. Calculate:

   $$
   N_{1+}(\cdot,\text{models})
   $$

7. Enumerate the distinct bigram types in the dataset and calculate:

   $$
   N_{1+}(\cdot,\cdot)
   $$

8. Calculate $P_{cont}(\text{Francisco})$, $P_{cont}(\text{language})$, and $P_{cont}(\text{models})$.
9. Explain why the ordinary token count of a word is not enough for this lower-order distribution.
10. Which should have the larger continuation probability: `Francisco` or `language`? Explain without relying only on their ordinary frequencies.

### What this tests

Whether you can distinguish:

```text
how often a word occurred
```

from:

```text
how many different histories preceded the word
```

---

## Question 7 — Full Kneser–Ney bigram calculation

Use the history:

```text
San ___
```

### Given

From the dataset:

```text
c(San, Francisco) = 2
c(San) = 2
N₁₊(San, ·) = 1
```

Use:

$$
D=0.5
$$

and the continuation probabilities from Question 6.

The interpolated Kneser–Ney formula is:

$$
P_{KN}(w\mid h)
=
\frac{\max(c(h,w)-D,0)}{c(h)}
+
\lambda(h)P_{cont}(w)
$$

### Calculate and explain

1. Calculate the direct discounted probability contribution for `Francisco`.
2. Calculate the total removed mass.
3. Calculate:

   $$
   \lambda(\text{San})
   $$

4. Calculate $P_{KN}(\text{Francisco}\mid\text{San})$.
5. Calculate $P_{KN}(\text{language}\mid\text{San})$, even though `San language` was never observed.
6. Write the general expression for $P_{KN}(w\mid\text{San})$ for every word $w$ other than `Francisco`.
7. Verify normalization symbolically by showing that the direct contribution plus the redistributed mass equals one:

   $$
   \sum_wP_{KN}(w\mid\text{San})=1
   $$

8. Explain why `Francisco` can receive a high probability after `San` while still having a low continuation probability overall.

### What this tests

Whether you can combine higher-order evidence, discounting, interpolation weight, and continuation probability in one calculation.

---

## Question 8 — Recursive Kneser–Ney with a trigram history

Use the trigram history:

```text
natural language ___
```

### Given

From the dataset:

```text
c(natural language, processing) = 2
c(natural language, models) = 2
c(natural language) = 4
N₁₊(natural language, ·) = 2
```

Use:

$$
D=0.5
$$

For the lower-order bigram model, you may use these already-computed values:

```text
P_KN(processing | language) = 0.19375
P_KN(the | language) = 0.01875
```

### Calculate and explain

1. Calculate:

   $$
   \lambda(\text{natural language})
   $$

2. Calculate the direct trigram contribution for `processing`.
3. Calculate:

   $$
   P_{KN}(\text{processing}\mid\text{natural language})
   $$

4. Calculate:

   $$
   P_{KN}(\text{the}\mid\text{natural language})
   $$

   Remember that `natural language the` is unseen.
5. Draw the recursive path for the unseen trigram:

   ```text
   P(the | natural language)
       → P(the | language)
       → P_cont(the)
   ```

   Mark which step is already supplied in the question and which step you are calculating.
6. Explain why the trigram can use the bigram `language` even though the full trigram history is unseen.
7. Identify where the probability mass removed from the two observed trigrams goes.

### What this tests

Whether you understand recursion rather than treating Kneser–Ney as a single flat formula.

---

## Question 9 — Modified Kneser–Ney with three discount buckets

Use the history:

```text
<s> ___
```

### Given

From the shared dataset, first calculate the beginning-word counts yourself. For this question, the relevant count-bucket totals are:

```text
c(<s>) = 10
N₁(<s>, ·) = 4
N₂(<s>, ·) = 3
N₃₊(<s>, ·) = 0
```

Use these Modified Kneser–Ney discounts:

$$
D_1=0.4,
\qquad
D_2=0.9,
\qquad
D_{3+}=1.3
$$

The relevant next-word counts are:

```text
c(<s>, I) = 2
c(<s>, We) = 2
c(<s>, They) = 1
c(<s>, San) = 2
```

For the lower-order continuation probabilities, use:

```text
P_MKN(I) = 1/30
P_MKN(We) = 1/30
P_MKN(They) = 1/30
P_MKN(San) = 1/30
P_MKN(language) = 3/30
```

Use:

$$
P_{MKN}(w\mid h)
=
\frac{\max(c(h,w)-D(c(h,w)),0)}{c(h)}
+
\lambda(h)P_{MKN}(w\mid h')
$$

### Calculate and explain

1. Calculate the total removed count from count-one continuations.
2. Calculate the total removed count from count-two continuations.
3. Calculate:

   $$
   \lambda(\langle s\rangle)
   $$

4. Calculate $P_{MKN}(I\mid\langle s\rangle)$.
5. Calculate $P_{MKN}(They\mid\langle s\rangle)$.
6. Calculate $P_{MKN}(language\mid\langle s\rangle)$. This word is unseen directly after `<s>` in the dataset.
7. For `I`, explain why you used $D_2$, not $D_1$ or $D_{3+}$.
8. Explain why the interpolation weight contains $D_1N_1$ and $D_2N_2$, rather than simply $D(N_1+N_2)$.
9. Compare this calculation with Question 7. What changed, and what stayed the same?

### What this tests

Whether you can apply count-dependent discounts without losing track of the ordinary Kneser–Ney continuation mechanism.

---

## Question 10 — Estimating MKN discounts and auditing your answer

This final question combines discount estimation, numerical calculation, and debugging checks.

### Part A: estimate the three discounts

### Given

Suppose the frequency-of-frequencies statistics for one n-gram order are:

$$
n_1=10,
\qquad
n_2=4,
\qquad
n_3=2,
\qquad
n_4=1
$$

Use:

$$
Y=\frac{n_1}{n_1+2n_2}
$$

$$
D_1=1-2Y\frac{n_2}{n_1}
$$

$$
D_2=2-3Y\frac{n_3}{n_2}
$$

$$
D_{3+}=3-4Y\frac{n_4}{n_3}
$$

Calculate $Y$, $D_1$, $D_2$, and $D_{3+}$, showing every substitution.

### Part B: apply and audit the discounts

Use the history from Question 9:

```text
c(<s>) = 10
N₁(<s>, ·) = 4
N₂(<s>, ·) = 3
N₃₊(<s>, ·) = 0
```

Use the discounts you calculated in Part A.

Calculate:

1. The total removed count from all observed continuations after `<s>`.
2. The interpolation weight $\lambda(\langle s\rangle)$.
3. The direct discounted contribution for a word with count 1.
4. The direct discounted contribution for a word with count 2.
5. The complete MKN probability for a count-1 word whose lower-order probability is $1/30$.
6. The complete MKN probability for a count-2 word whose lower-order probability is $1/30$.

### Part C: correctness checklist

Answer these conceptually and use your calculations as evidence:

1. Is every probability nonnegative?
2. Is the discount for a count-1 n-gram smaller than its count?
3. Is the discount for a count-2 n-gram smaller than its count?
4. Does the direct probability mass plus the redistributed probability mass sum to one?
5. Does λ equal:

   $$
   \frac{\text{total removed count}}{c(h)}
   $$

6. If your total is not one, identify exactly where the missing or extra mass came from.
7. Explain why a discount $D_2$ can be greater than 1 and still be valid for a count-2 n-gram.
8. Explain why using one fixed $D$ for count 1, count 2, and count 10 is less flexible than using $D_1$, $D_2$, and $D_{3+}$.

### What this tests

Whether you can calculate Modified Kneser–Ney and independently audit the result instead of trusting a complicated-looking formula.

---

## Final self-check: explain these without looking at your notes

After solving all ten questions, try to answer each prompt aloud in two or three sentences:

1. What does discounting do to a seen n-gram count?
2. Why must the removed mass be redistributed?
3. Where does redistributed mass go in absolute discounting?
4. Where does redistributed mass go in Kneser–Ney?
5. What is the difference between backoff and interpolation?
6. What does $N_{1+}(h,\cdot)$ count?
7. What does $N_{1+}(\cdot,w)$ count?
8. Why can `Francisco` be frequent but have a low continuation probability?
9. What do $D_1$, $D_2$, and $D_{3+}$ mean?
10. What calculation proves that your smoothed distribution is normalized?

If you can answer these and reproduce the numerical calculations, you have the essential working understanding of Backoff, Interpolation, Absolute Discounting, Kneser–Ney, and Modified Kneser–Ney.
