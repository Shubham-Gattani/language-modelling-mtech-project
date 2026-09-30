# MKN Implementation Summary

This is a short, end-to-end explanation of how the toy Modified Kneser-Ney implementation works.

## End-to-end flow

```text
raw corpus
→ read sentences
→ add sentence boundaries
→ count unigrams, bigrams, and trigrams
→ build vocabulary and predecessor sets
→ calculate continuation probabilities
→ estimate discounts
→ calculate lambda values when needed
→ recursively calculate probabilities
→ verify normalization
```

The main code path is:

```text
main()
→ load_corpus()
→ MKNModel.from_sentences()
→ MKNModel.__init__()
→ build model statistics
→ print_demo()
→ run probability and normalization checks
```

Training here means collecting counts and MKN statistics. It is not neural-network training.

---

## 1. Preprocessing

### Step 1: Read the corpus

load_corpus() reads one non-empty sentence per line.

```text
the cat sat on the mat
the cat sat on the rug
...
```

It returns a Python list of raw sentence strings.

### Step 2: Add sentence boundaries

tokenize_sentence() splits each sentence and adds boundary tokens:

```text
the cat sat on the mat
→ <s> the cat sat on the mat </s>
```

The model can now learn:

```text
<s> → first word
last word → </s>
```

### Step 3: Generate n-gram positions

For a tokenized sentence with L tokens:

```text
number of n-gram positions = L - n + 1
```

For the complete toy corpus:

```text
sentences         = 12
token positions   = 94
bigram positions  = 82
trigram positions = 70
```

These are positions before repeated n-grams are merged into dictionary keys.

---

## 2. Model training

The constructor builds:

```python
self.tokenized_sentences
self.vocabulary
self.ngram_counts
self.predecessors_by_word
self.continuation_denominator
self.discounts_by_order
```

### Step 1: Count n-grams

_build_ngram_counts() creates separate dictionaries:

```python
counts_by_order = {
    1: {},  # unigram counts
    2: {},  # bigram counts
    3: {},  # trigram counts
}
```

Each dictionary maps an n-gram tuple to its count.

```python
("the", "cat"): 3
("the", "cat", "sat"): 3
```

The update rule is:

```python
counts[ngram] = counts.get(ngram, 0) + 1
```

Final numbers of distinct types:

```text
unigrams  = 14
bigrams   = 20
trigrams  = 24
```

### Step 2: Build the vocabulary

The vocabulary contains distinct tokens the model may predict.

```text
<s> is removed because it is a context marker.
</s> is kept because the model predicts sentence termination.
```

Final prediction vocabulary size:

```text
13
```

### Step 3: Build predecessor sets

For each word, the model stores the distinct tokens that appeared immediately before it.

Example:

```python
predecessors_by_word["the"] = {
    "<s>", "on", "near", "over"
}
```

This measures contextual variety, not raw frequency.

### Step 4: Calculate continuation probabilities

There are 20 distinct bigram types, so:

```text
continuation denominator = 20
```

The formula is:

```text
P_KN(word)
= number of distinct predecessors of word
  / 20
```

Examples:

```text
P_KN(the) = 4/20 = 0.20
P_KN(sat) = 2/20 = 0.10
P_KN(fly) = 1/20 = 0.05
```

Mental model:

> Raw unigram count measures popularity. Continuation count measures how broadly a word appears across contexts.

### Step 5: Estimate discounts

The model estimates three discounts:

```text
D1  → observed count is 1
D2  → observed count is 2
D3+ → observed count is 3 or greater
```

It first calculates frequency-of-frequencies:

```text
n1 = number of n-gram types with count 1
n2 = number of n-gram types with count 2
n3 = number of n-gram types with count 3
n4 = number of n-gram types with count 4
```

Final discount values:

```python
discounts_by_order = {
    2: (0.5, 1.625, 2.0),
    3: (
        0.1818181818...,
        1.7575757576...,
        2.2727272727...,
    ),
}
```

For count zero:

```text
selected discount = 0
direct contribution = 0
```

### Step 6: Calculate lambda

For a history h:

```text
lambda(h)
= total discount removed from n-grams beginning with h
  / count(h)
```

Examples:

```text
lambda(("the", "cat")) = 25/33 = 0.7575757576...
lambda(("cat",))       = 0.4
```

Mental model:

> Lambda is the lower-order model's share of the probability budget.

---

## 3. Inference

The public entry point is:

```python
model.probability(word, history)
```

The flow is:

```text
validate the query
→ call _probability()
→ calculate direct contribution
→ recursively calculate lower-order probability
→ combine both parts
→ return the result
```

### 3.1 Seen trigram

Query:

```text
P(sat | the cat)
```

Counts:

```text
count(the cat sat) = 3
count(the cat)     = 3
```

Direct trigram contribution:

```text
(3 - 25/11) / 3
= 8/33
```

Lambda:

```text
lambda(the cat) = 25/33
```

The lower-order probability is:

```text
P(sat | cat) = 16/25
```

Final calculation:

```text
P(sat | the cat)
= 8/33 + (25/33)(16/25)
= 8/11
= 0.7272727273...
```

The recursion is:

```text
P(sat | the cat)
→ P(sat | cat)
→ P(sat)
```

### 3.2 Unseen trigram with seen history

Query:

```text
P(on | the cat)
```

The history is seen, but the trigram is not:

```text
count(the cat)     = 3
count(the cat on)  = 0
```

Therefore the direct trigram contribution is zero:

```text
direct contribution = 0
```

The model uses lower-order evidence:

```text
P(on | the cat)
= lambda(the cat) × P(on | cat)
= (25/33)(1/50)
= 1/66
= 0.0151515152...
```

### 3.3 Unseen bigram with seen history

Query:

```text
P(fly | cat)
```

```text
count(cat)      = 5
count(cat fly)  = 0
```

Therefore:

```text
predecessors(fly) = {birds}
P(fly)
= number of distinct predecessors of fly / distinct bigrams
= 1 / 20
= 0.05

P(fly | cat)
= 0 + lambda(cat) × P(fly)
= 0 + (2/5)(1/20)
= 1/50
= 0.02
```

An unseen n-gram gets no direct evidence, but it can still receive lower-order probability.

### 3.4 Unseen history

Query:

```text
P(sat | the birds)
```

The history the birds was never observed. The model shortens it:

```text
("the", "birds") → ("birds",)
```

Then:

```text
history_count(("birds",)) = count(("birds",)) = 2
count(("birds", "sat")) = 0
count(("birds", "fly")) = 2

removed_count = D2_bigram = 1.625
lambda(("birds",)) = 1.625 / 2
                           = 0.8125
                           = 13/16

P(sat) = number of distinct predecessors of sat / 20
       = 2 / 20
       = 1/10

P(sat | birds)
= 0 + lambda(("birds",)) × P(sat)
= 0 + (13/16)(1/10)
= 13/160
= 0.08125

P(sat | the birds)
= P(sat | birds)
= 13/160
= 0.08125
```

The distinction is:

```text
seen history + unseen n-gram
→ interpolate with zero direct contribution

unseen history
→ directly use a shorter history
```

### 3.5 General recursive equation

For a seen history:

```text
P(word | history)
= direct higher-order contribution
  + lambda(history) × lower-order probability
```

For the empty history:

```text
P(word) = continuation probability
```

For an unseen history:

```text
P(word | unseen history)
= P(word | shorter history)
```

### 3.6 Normalization check

For every history, the model sums probabilities over all 13 vocabulary items.

```text
sum P(word)                 = 1
sum P(word | cat)           = 1
sum P(word | the cat)       = 1
```

The code checks this numerically with a small floating-point tolerance.

---

## Final mental model

```text
Preprocessing creates n-gram evidence.

Training converts that evidence into:
counts, predecessor sets, continuation probabilities,
discounts, and lambda values.

Inference combines:
higher-order phrase evidence
+ lower-order contextual evidence.

Discounting removes mass from observed n-grams.
Lambda transfers that mass downward.
Continuation probabilities give unseen events a chance.
Normalization confirms that no probability mass was lost.
```

> MKN trusts specific phrases when evidence exists, but uses contextual versatility and lower-order evidence when phrase evidence is missing.
