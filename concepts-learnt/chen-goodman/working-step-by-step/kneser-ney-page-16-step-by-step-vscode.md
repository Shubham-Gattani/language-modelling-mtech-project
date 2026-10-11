# Kneser–Ney Smoothing: Understanding the “Francisco” Example

This note explains the passage on page 16 of *An Empirical Study of Smoothing Techniques for Language Modeling*.

The central idea is:

> A word’s ordinary unigram count asks: “How often did this word occur?”
> Kneser–Ney’s continuation count asks: “How many different contexts can this word continue?”

That small change is what fixes the `San Francisco` problem.

---

## 1. Start with what a bigram language model predicts

Suppose we want to predict the next word after a history word `h`.

For example, the history is:

```text
San ___
```

The candidate next word might be `Francisco`, so the model wants:

$$
P(\text{Francisco}\mid\text{San})
$$

A maximum-likelihood bigram model estimates this using counts:

$$
P_{ML}(w\mid h)=\frac{c(h,w)}{c(h)}
$$

where:

- $c(h,w)$ is the number of times the bigram $(h,w)$ occurs;
- $c(h)$ is the number of times the history word $h$ occurs as a first word of a bigram.

For example, imagine this tiny training corpus:

```text
San Francisco
San Francisco
San Francisco
San Francisco
San Francisco
San Jose
New York
New Orleans
```

The relevant counts are:

$$
c(\text{San},\text{Francisco})=5
$$

$$
c(\text{San},\text{Jose})=1
$$

Therefore:

$$
c(\text{San})=5+1=6
$$

and the maximum-likelihood estimate is:

$$
P_{ML}(\text{Francisco}\mid\text{San})
=\frac{5}{6}
=0.8333
$$

So after `San`, `Francisco` is correctly assigned a high probability.

The problem appears when the model has to back off from an unseen history.

---

## 2. Why smoothing needs a lower-order probability

Suppose the model encounters this new bigram:

```text
visited Francisco
```

but `visited Francisco` never appeared in training.

Then:

$$
c(\text{visited},\text{Francisco})=0
$$

The unsmoothed estimate is:

$$
P_{ML}(\text{Francisco}\mid\text{visited})=0
$$

That zero is too harsh. The model should retain some possibility for an unseen bigram. Smoothing gives probability mass to such unseen events.

Absolute discounting does this by subtracting a fixed amount $D$ from seen bigram counts and distributing the removed probability mass using a unigram model.

Its interpolated form is:

$$
P_{AD}(w\mid h)
=
\frac{\max(c(h,w)-D,0)}{c(h)}
+
\lambda(h)P_{uni}(w)
$$

where:

$$
P_{uni}(w)=\frac{c(w)}{\sum_x c(x)}
$$

and, in the usual formulation,

$$
\lambda(h)=\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

Here $N_{1+}(h,\cdot)$ means “the number of different words observed after history $h$.”

For an unseen bigram, $c(h,w)=0$, so its first term vanishes:

$$
P_{AD}(w\mid h)=\lambda(h)P_{uni}(w)
$$

Thus, when backing off, absolute discounting uses the ordinary unigram probability.

---

## 3. The “Francisco” problem

Now make `Francisco` very frequent, but extremely restricted in where it appears.

Imagine the corpus contains:

```text
San Francisco       100 times
San Jose              1 time
New York              1 time
New Orleans           1 time
```

The word `Francisco` occurs 100 times:

$$
c(\text{Francisco})=100
$$

The total number of word tokens in this simplified corpus is:

$$
100+1+1+1=103
$$

Therefore its ordinary unigram probability is approximately:

$$
P_{uni}(\text{Francisco})
=\frac{100}{103}
\approx 0.9709
$$

That is very high.

Now consider an unseen bigram such as:

```text
visited Francisco
```

Suppose:

$$
c(\text{visited})=3
$$

and `visited` has been followed by two different words. Let the discount be:

$$
D=0.75
$$

Then:

$$
\lambda(\text{visited})
=\frac{D\,N_{1+}(\text{visited},\cdot)}{c(\text{visited})}
=\frac{0.75\times 2}{3}
=0.5
$$

Because `visited Francisco` is unseen:

$$
P_{AD}(\text{Francisco}\mid\text{visited})
=0.5\times 0.9709
=0.48545
$$

So absolute discounting assigns nearly 0.49 probability to `Francisco` after `visited`.

That feels wrong. Why?

Because the 100 occurrences of `Francisco` do not represent 100 different ways of using the word. All 100 occurrences have exactly one preceding word:

```text
San Francisco
```

The word is frequent, but it is not broadly usable after arbitrary histories. Its frequency is almost entirely explained by the strong bigram `San Francisco`.

The lower-order model should therefore not say merely:

> “Francisco appeared many times.”

It should ask:

> “Francisco appeared after how many different words?”

For `Francisco`, the answer is only one: `San`.

---

## 4. Two different meanings of “count”

This is the conceptual heart of Kneser–Ney.

### Ordinary unigram count

The ordinary count is:

$$
c(w)
$$

It counts word tokens.

For our example:

$$
c(\text{Francisco})=100
$$

Repeated occurrences are counted repeatedly:

```text
San Francisco       occurrence 1
San Francisco       occurrence 2
San Francisco       occurrence 3
...
San Francisco       occurrence 100
```

### Continuation count

Kneser–Ney instead uses:

$$
N_{1+}(\cdot,w)
$$

Read this as:

> the number of distinct words that have appeared immediately before (w).

For `Francisco`, inspect its left neighbors:

```text
San Francisco
San Francisco
San Francisco
...
San Francisco
```

The list of left neighbors is:

```text
San, San, San, ..., San
```

After removing duplicates:

```text
{San}
```

Therefore:

$$
N_{1+}(\cdot,\text{Francisco})=1
$$

This is exactly what the paper means by saying that the unigram probability should be proportional to “the number of different words that it follows.” The wording is from the perspective of the current word: `Francisco` follows `San`; equivalently, `San` precedes `Francisco`.

---

## 5. A word that really should receive a high continuation probability

Compare `Francisco` with `the`.

Suppose the corpus contains:

```text
of the
to the
in the
on the
with the
```

The word `the` occurs five times. Its ordinary count is:

$$
c(\text{the})=5
$$

Its distinct preceding words are:

```text
{of, to, in, on, with}
```

Therefore:

$$
N_{1+}(\cdot,\text{the})=5
$$

Now compare:

| Word | Ordinary count | Distinct preceding words |
|---|---:|---:|
| Francisco | 100 | 1: San |
| the | 5 | 5: of, to, in, on, with |

The ordinary unigram model says:

$$
c(\text{Francisco})>c(\text{the})
$$

so it may give `Francisco` a higher probability.

But the continuation view says:

$$
N_{1+}(\cdot,\text{Francisco})
<
N_{1+}(\cdot,\text{the})
$$

That is more sensible for backing off: `the` can naturally continue many different histories, whereas `Francisco` is strongly tied to `San`.

---

## 6. Deriving the continuation probability

Let the set of all distinct bigram types in the training corpus be:

```text
{(San, Francisco), (San, Jose), (New, York), (New, Orleans)}
```

There are four distinct bigram types, so:

$$
N_{1+}(\cdot,\cdot)=4
$$

For each possible next word, count how many distinct histories precede it:

### `Francisco`

Only:

```text
(San, Francisco)
```

Thus:

$$
N_{1+}(\cdot,\text{Francisco})=1
$$

### `Jose`

Only:

```text
(San, Jose)
```

Thus:

$$
N_{1+}(\cdot,\text{Jose})=1
$$

### `York`

Only:

```text
(New, York)
```

Thus:

$$
N_{1+}(\cdot,\text{York})=1
$$

### `Orleans`

Only:

```text
(New, Orleans)
```

Thus:

$$
N_{1+}(\cdot,\text{Orleans})=1
$$

The Kneser–Ney continuation probability is:

$$
P_{cont}(w)
=
\frac{N_{1+}(\cdot,w)}{N_{1+}(\cdot,\cdot)}
$$

Therefore:

$$
P_{cont}(\text{Francisco})=\frac{1}{4}=0.25
$$

In a larger realistic corpus, if `Francisco` appeared 100 times but always after `San`, its numerator would still be 1. The denominator would be the total number of distinct bigram types in the corpus, which would usually make its continuation probability quite small.

---

## 7. Kneser–Ney bigram formula

For a bigram, interpolated Kneser–Ney smoothing is:

$$
P_{KN}(w\mid h)
=
\frac{\max(c(h,w)-D,0)}{c(h)}
+
\lambda(h)P_{cont}(w)
$$

The backoff weight is:

$$
\lambda(h)
=
\frac{D\,N_{1+}(h,\cdot)}{c(h)}
$$

The formula looks almost identical to absolute discounting. The crucial difference is the lower-order distribution:

$$
\text{Absolute discounting: }P_{uni}(w)=\frac{c(w)}{\sum_x c(x)}
$$

$$
\text{Kneser–Ney: }P_{cont}(w)=\frac{N_{1+}(\cdot,w)}{N_{1+}(\cdot,\cdot)}
$$

So Kneser–Ney does not change the main discounted bigram term. It changes what probability distribution receives the discounted mass.

---

## 8. Calculate the same unseen probability under Kneser–Ney

Continue with the earlier setup:

$$
D=0.75
$$

$$
c(\text{visited})=3
$$

$$
N_{1+}(\text{visited},\cdot)=2
$$

Thus:

$$
\lambda(\text{visited})=0.5
$$

The bigram `visited Francisco` is unseen, so its discounted term is zero:

$$
\frac{\max(c(\text{visited},\text{Francisco})-D,0)}{c(\text{visited})}
=
\frac{\max(0-0.75,0)}{3}
=0
$$

Now use the continuation probability. In the small four-type example:

$$
P_{cont}(\text{Francisco})=\frac{1}{4}=0.25
$$

Therefore:

$$
P_{KN}(\text{Francisco}\mid\text{visited})
=0+0.5\times 0.25
=0.125
$$

Compare the two estimates:

$$
P_{AD}(\text{Francisco}\mid\text{visited})=0.5\times\frac{100}{103}
\approx0.48545
$$

$$
P_{KN}(\text{Francisco}\mid\text{visited})=0.5\times\frac{1}{4}
=0.125
$$

The exact values depend on the whole corpus, but the direction is the important part:

```text
Absolute discounting: Francisco is high because it occurred many times.
Kneser–Ney:          Francisco is low because it occurred after only one history.
```

---

## 9. Understanding the paper’s “traversing the training data” argument

The paper gives an operational way to understand the continuation count.

Imagine reading the corpus from left to right. At each position, use the preceding word as the current history and predict the current word.

Consider this sequence of bigrams:

```text
San Francisco
San Francisco
San Jose
New York
New Orleans
```

Process them one at a time.

### Step 1: first `San Francisco`

The bigram has not appeared in the earlier data.

```text
Current bigram: San Francisco
Previously seen: nothing
New bigram: yes
```

The unigram/backoff component participates. Assign one continuation count to `Francisco`:

$$
\text{continuation count of Francisco}=1
$$

### Step 2: second `San Francisco`

The bigram has already appeared.

```text
Current bigram: San Francisco
Previously seen: San Francisco
New bigram: no
```

Do not assign another continuation count to `Francisco`.

This is the key point: repetition increases the ordinary token count, but not the distinct-context count.

### Step 3: `San Jose`

This is a new bigram type, so assign one continuation count to `Jose`:

$$
\text{continuation count of Jose}=1
$$

### Step 4: `New York`

This is another new bigram type:

$$
\text{continuation count of York}=1
$$

### Step 5: `New Orleans`

Again, a new bigram type:

$$
\text{continuation count of Orleans}=1
$$

At the end:

| Word | Ordinary token count | Continuation count |
|---|---:|---:|
| Francisco | 2 | 1 |
| Jose | 1 | 1 |
| York | 1 | 1 |
| Orleans | 1 | 1 |

The paper’s statement follows directly:

> If we count a word when its current bigram is new, each distinct bigram type contributes once. Therefore, the number of counts assigned to a word equals the number of different histories that precede it.

For `Francisco`, 100 repeated occurrences of `San Francisco` still produce only one distinct bigram type, so the continuation count remains 1.

---

## 10. Why the lower-order model must change too

This deserves emphasis. Suppose we only changed the formula’s name but kept the ordinary unigram:

$$
P(w)=\frac{c(w)}{\sum_xc(x)}
$$

Then the `Francisco` problem would remain. The model would still treat 100 repeated occurrences after `San` as evidence that `Francisco` is broadly likely.

Kneser–Ney changes the question asked by the lower-order model:

```text
Ordinary unigram:       How often did w occur?
Continuation unigram:   How many different histories led to w?
```

The main bigram model already captures that `San` strongly predicts `Francisco`. Therefore, the lower-order model should not count that same evidence 100 additional times. It should reserve its probability for words that can continue many different histories.

This is a division of labor:

1. The higher-order bigram term captures repeated, specific associations such as `San → Francisco`.
2. The lower-order continuation term estimates how generally usable a word is across contexts.

---

## 11. One step further: why the same idea becomes recursive for trigrams

The paper is discussing bigrams, but the same idea extends recursively to trigrams.

A trigram predicts:

$$
P(w_i\mid w_{i-2},w_{i-1})
$$

For example:

```text
I visited San Francisco
```

When predicting `Francisco`, the trigram history is:

```text
I visited San ___
```

The interpolated Kneser–Ney formula is:

$$
P_{KN}(w_i\mid w_{i-2},w_{i-1})
=
\frac{\max(c(w_{i-2},w_{i-1},w_i)-D,0)}{c(w_{i-2},w_{i-1})}
+
\lambda(w_{i-2},w_{i-1})
P_{KN}(w_i\mid w_{i-1})
$$

Notice the recursive lower-order call:

$$
P_{KN}(w_i\mid w_{i-1})
$$

If the trigram is unseen, the model does not jump directly to an ordinary unigram. It backs off one level, to a Kneser–Ney bigram. If that bigram is also unseen, the bigram model backs off again to a continuation probability.

The chain is:

$$
P(w_i\mid w_{i-2},w_{i-1})
\rightarrow
P(w_i\mid w_{i-1})
\rightarrow
P_{cont}(w_i)
$$

### A concrete recursive example

Suppose the training corpus contains:

```text
I visited San Francisco
We visited San Francisco
They saw San Francisco
```

Now ask for:

$$
P(\text{Francisco}\mid\text{new},\text{San})
$$

The trigram `new San Francisco` is unseen:

$$
c(\text{new},\text{San},\text{Francisco})=0
$$

Therefore the trigram’s direct discounted term is zero, and it backs off to:

$$
P(\text{Francisco}\mid\text{San})
$$

The bigram `San Francisco` is seen, so it receives a direct discounted bigram contribution. This is sensible: even though the full three-word context is new, the shorter context `San` provides strong evidence.

Now ask for:

$$
P(\text{Francisco}\mid\text{new},\text{visited})
$$

If `new visited Francisco` is unseen and `visited Francisco` is also unseen, the model backs off again:

$$
P(\text{Francisco}\mid\text{new},\text{visited})
\rightarrow
P(\text{Francisco}\mid\text{visited})
\rightarrow
P_{cont}(\text{Francisco})
$$

At the final level, `Francisco` is judged by the number of different words that precede it. Since only `San` precedes it in the corpus:

$$
N_{1+}(\cdot,\text{Francisco})=1
$$

So even after two unseen-context decisions, the model does not use the misleading token count of 100. It uses the continuation behavior of the word.

---

## 12. The mental model to remember

Imagine every word carrying two labels.

### Label 1: popularity

```text
How many total times did I appear?
```

This is the ordinary count $c(w)$.

### Label 2: versatility

```text
How many different histories can lead to me?
```

This is the continuation count $N_{1+}(\cdot,w)$.

`Francisco` may be very popular but not versatile:

```text
Francisco: popular = high, versatile = low
```

`the` may be both reasonably frequent and highly versatile:

```text
the:        popular = moderate/high, versatile = high
```

Absolute discounting uses popularity for the lower-order model. Kneser–Ney uses versatility.

The reason is that the higher-order model has already explained repeated occurrences inside specific phrases. After removing that phrase-specific evidence, the remaining probability mass should reflect how widely a word appears across different contexts.

That is the complete intuition behind the quoted paragraph.

---

## 13. What does “the higher-order model has already explained it” really mean?

This sentence is easy to read too quickly, so let us slow it down.

The sentence says:

> The higher-order model has already explained repeated occurrences inside specific phrases. After removing that phrase-specific evidence, the remaining probability mass should reflect how widely a word appears across different contexts.

There are two separate jobs being performed:

1. The higher-order model explains **specific relationships**.
2. The lower-order model handles the probability mass left over for **unseen relationships**.

The problem with an ordinary unigram is that it gives the same evidence to both jobs. Kneser–Ney separates them.

### 13.1 Think of every occurrence as evidence with an owner

Take these corpus fragments:

```text
San Francisco
San Francisco
San Francisco
...
San Francisco       100 times
```

Those 100 occurrences provide very strong evidence for the specific relationship:

```text
San → Francisco
```

The bigram model is the right place to use that evidence:

$$
P(\text{Francisco}\mid\text{San})
$$

It can say:

> When the previous word is `San`, `Francisco` is very likely.

That is the phrase-specific explanation.

Now consider a completely different history:

```text
visited → ___
```

The bigram `visited Francisco` was never observed. The model needs a fallback probability:

$$
P(\text{Francisco}\mid\text{visited})
$$

Should the 100 occurrences of `San Francisco` be used again here as if they were 100 pieces of evidence that `Francisco` is a generally suitable continuation?

No. Those occurrences have already done their main job: they made `Francisco` likely after `San`. They did not demonstrate that `Francisco` is likely after many unrelated words.

This is what “the higher-order model has already explained” means. The evidence is not deleted from the model. It is assigned to the context where it actually makes sense.

### 13.2 A simple accounting analogy

Imagine that a restaurant bill contains 100 dollars labelled:

```text
San Francisco meal
```

That 100 dollars tells us a lot about the cost of the `San Francisco` meal. But we should not also treat it as 100 dollars of evidence that every other meal costs 100 dollars.

Similarly:

```text
100 occurrences of San Francisco
```

are strong evidence for:

```text
Francisco after San
```

but weak evidence for:

```text
Francisco after an arbitrary word
```

The ordinary unigram count reuses the same 100 occurrences for the second conclusion. Kneser–Ney avoids that reuse by counting distinct contexts instead.

### 13.3 See the probability mass being divided

Suppose the history `San` has been followed by three different words:

```text
San Francisco       100 times
San Jose               1 time
San Diego              1 time
```

Then:

$$
c(\text{San})=100+1+1=102
$$

Assume the discount is:

$$
D=0.5
$$

The discounted higher-order terms are:

$$
\frac{c(\text{San},\text{Francisco})-D}{c(\text{San})}
=\frac{100-0.5}{102}
=\frac{99.5}{102}
\approx 0.9755
$$

$$
\frac{c(\text{San},\text{Jose})-D}{c(\text{San})}
=\frac{1-0.5}{102}
=\frac{0.5}{102}
\approx 0.0049
$$

$$
\frac{c(\text{San},\text{Diego})-D}{c(\text{San})}
=\frac{1-0.5}{102}
=\frac{0.5}{102}
\approx 0.0049
$$

The total retained mass is:

$$
0.9755+0.0049+0.0049
\approx 0.9853
$$

The removed mass is approximately:

$$
1-0.9853=0.0147
$$

Why was approximately 0.0147 removed? Because discounting removed $0.5$ count from each of the three observed continuation types:

$$
\frac{3\times 0.5}{102}
=\frac{1.5}{102}
\approx 0.0147
$$

That removed mass must be distributed among possible next words. This is the “remaining probability mass” in the sentence from the paper.

### 13.4 What should receive the removed mass?

The removed mass is used when the exact bigram is not trusted completely. For example, perhaps we want:

$$
P(\text{Francisco}\mid\text{visited})
$$

where `visited Francisco` was unseen.

At this point, the model should not ask:

> Which word occurred most often anywhere?

That question produces the ordinary unigram and makes `Francisco` look large because of the 100 repetitions after `San`.

Instead, it should ask:

> Which words have appeared as continuations of many different histories?

That question produces the continuation distribution:

$$
P_{cont}(w)
=
\frac{N_{1+}(\cdot,w)}{N_{1+}(\cdot,\cdot)}
$$

If the corpus contains:

```text
San Francisco       100 times
of the                1 time
to the                1 time
in the                1 time
with the              1 time
```

then the distinct preceding-word sets are:

```text
Francisco: {San}
the:        {of, to, in, with}
```

Therefore:

$$
N_{1+}(\cdot,\text{Francisco})=1
$$

but:

$$
N_{1+}(\cdot,\text{the})=4
$$

The continuation distribution gives more of the fallback mass to `the` than to `Francisco`, because `the` has demonstrated that it can continue many different histories.

### 13.5 The key distinction: explaining a phrase versus generalizing beyond it

There are two questions:

#### Question A: Is `Francisco` likely after `San`?

Use the higher-order bigram count:

$$
c(\text{San},\text{Francisco})=100
$$

Answer: yes, very likely.

#### Question B: Is `Francisco` generally likely after an unknown word?

Use the continuation count:

$$
N_{1+}(\cdot,\text{Francisco})=1
$$

Answer: not especially; it has only demonstrated one kind of continuation context.

These answers are not contradictory:

```text
Francisco is highly likely after San,
but not broadly likely after arbitrary histories.
```

Absolute discounting tends to blur these two answers because its fallback distribution uses:

$$
c(\text{Francisco})=100
$$

Kneser–Ney keeps them separate:

```text
Specific phrase evidence:  c(San, Francisco) = 100
General continuation evidence: N₁₊(·, Francisco) = 1
```

### 13.6 One-sentence mental model

When a high-order model has already used repeated evidence to explain a specific phrase, the lower-order model should not reuse the repetition as evidence of general usefulness; it should use only the word’s diversity of contexts.
