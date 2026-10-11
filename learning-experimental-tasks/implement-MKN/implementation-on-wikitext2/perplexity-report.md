# Modified Kneser-Ney Perplexity Report

Date: 1st Oct 2026

## Result to report

Using the current WikiText-2 MKN experiment:

```text
validation perplexity = 284.5840135851
```

A rounded value

```text
Validation perplexity: 284.58
```

## Experimental setup

```text
dataset:              WikiText-2, wikitext-2-v1, non-raw word-level version
dataset source:       locally saved Hugging Face DatasetDict
model:                interpolated Modified Kneser-Ney
maximum n-gram order: 3 (trigram model)
training rows:        first 5,000 non-empty rows from train
validation rows:      first 500 non-empty rows from validation
predicted tokens:     37,504
distinct train tokens: 21,568
model vocabulary:      21,569 (distinct train tokens+<\s>)
```

The model vocabulary for this run contained 21,569 tokens, excluding `<s>`.
The training prefix contained `<unk>` tokens, so validation words outside the
training vocabulary could be mapped to the existing `<unk>` token.

## Vocabulary size versus predicted-token count

These two numbers measure different things:

```text
21,569 -> number of candidate tokens the model can predict for one history
37,504 -> number of actual next-token prediction events in validation
```

For each validation position, the model evaluates the probability of the
actual next token. It does not mean that the vocabulary contains 37,504
tokens. The same 21,569-token vocabulary is reused at every prediction
position.

For example, one row such as:

```text
<s> The cat sat </s>
```

creates four prediction events:

```text
predict The
predict cat
predict sat
predict </s>
```

Across the first 500 validation rows, these events added up to 37,504.

In short: `37,504` means that across the 500 validation rows, the model evaluated:

```text
P(actual_next_token | history)
```

37,504 times.

The actual next token could be:

```text
a normal word
punctuation
<unk>
</s>
```

It does **not** mean the model had a vocabulary of 37,504 tokens.

## Important interpretation

Perplexity was calculated on the first 500 non-empty validation rows, not on the full validation split and not on the test split.

The current experiment uses validation perplexity as a learning-time
diagnostic. The test split should be reserved for a final result after the
implementation and any choices have been frozen.

## How the score was calculated

For every validation row, the evaluator:

```text
1. keeps known words and maps unknown words to <unk>;
2. adds <s> at the beginning and </s> at the end;
3. predicts every token after <s>;
4. adds the log probability of each predicted token;
5. calculates exp(-average_log_probability).
```

The end token `</s>` is included in the predicted-token count.

## How the score was calculated — small example

Suppose the validation set contains two short rows:

```text
The dog sat
The cat sat
```

The evaluator adds sentence-boundary tokens:

```text
<s> The dog sat </s>
<s> The cat sat </s>
```

It does not predict `<s>`. It predicts every token that follows it:

| Row | Prediction events |
|---|---|
| `<s> The dog sat </s>` | `The`, `dog`, `sat`, `</s>` |
| `<s> The cat sat </s>` | `The`, `cat`, `sat`, `</s>` |

Therefore, this tiny validation set contains:

```text
N = 8 prediction events
```

The model evaluates the probability of the actual next token at each event:

```text
P(The | <s>)
P(dog | <s>, The)
P(sat | The, dog)
P(</s> | dog, sat)

P(The | <s>)
P(cat | <s>, The)
P(sat | The, cat)
P(</s> | cat, sat)
```

For example, if these probabilities were:

```text
0.2, 0.5, 0.4, 0.7,
0.2, 0.6, 0.4, 0.7
```

then the mathematical definition of perplexity would be:

```text
PPL = (1 / (0.2 * 0.5 * 0.4 * 0.7 *
            0.2 * 0.6 * 0.4 * 0.7)) ** (1 / 8)
```

which is approximately `2.39`.

## Why the code uses log-probabilities

The code does not literally multiply all 37,504 probabilities. Multiplying
many small probabilities can become too small for a computer to represent;
this is called numerical underflow.

Instead, it uses the mathematically equivalent log form:

```text
PPL = exp(-(1 / N) * sum(log(P_i)))
```

For this experiment, `N = 37,504` and `P_i` is the probability assigned to
the actual validation token at prediction event `i`.

The two forms are equivalent:

```text
(product(1 / P_i)) ** (1 / N)
      = exp(-(1 / N) * sum(log(P_i)))
```

The log form is used because adding log-probabilities is numerically safer
than multiplying thousands of very small probabilities.

## Experiment configuration command

From the WikiText-2 implementation directory:

```bash
cd learning-experimental-tasks/implement-MKN/implementation-on-wikitext2
source venv/bin/activate
python3 mkn_wikitext2.py \
  --dataset-path data/wikitext-2 \
  --max-train-lines 5000 \
  --max-valid-lines 500
```

This command shows the correct experiment settings: the dataset, 5,000
training rows, and 500 validation rows.

The command may currently take a very long time because the teaching version
recalculates each history's `lambda` repeatedly. After we optimize the code
to calculate each `lambda` once and reuse it, the same command should run in
a reasonable amount of time.

## Runtime caveat

The current teaching implementation calculates `lambda_for_history()` by
scanning the complete n-gram table whenever a probability is requested. On a
real corpus, this makes the default CLI very slow during validation and its
full-vocabulary normalization checks.

For the reported number, lambda values were precomputed once per history by a
separate one-off script. This optimization is not yet in the main code.

The next engineering improvement should be to precompute or cache these
history-level lambda values inside the model, then rerun the report directly
through the main CLI.

## In one line

> “On the first 5,000 non-empty WikiText-2 training rows, our interpolated
> Modified Kneser-Ney trigram model achieved a validation perplexity of about
> 284.58 on the first 500 non-empty validation rows. This is an initial
> bounded experiment; the test split has not yet been used for the final
> evaluation, and the implementation still needs a runtime optimization for
> history-level lambda computation.”

## Meaning of some terms/lines:

**wikitext-2-v1**-->This is the specific Hugging Face configuration we downloaded.
**non-raw word-level version**-->

The text has already been prepared for word-level language modeling:

- words are separated into tokens;
- punctuation is retained as tokens;
- rare/out-of-vocabulary words are represented by `<unk>`;

# Questions to be discussed

#### When we talk about model, we store some values which can be used to completely define a model, e.g. in linear regression, we can just stoe (w1,w2,...wn,b) and be done. Simialrly, for a trigram MKN model, we can just store the counts_by_order dictionary and it can be used to produce everything else that we need for prediction of "w" given "h". Right?

**Almost, but with an important distinction.**

If `counts_by_order` contains the exact counts of every distinct unigram, bigram, and trigram, then it is theoretically enough to reconstruct the other MKN quantities:

- unigram counts → vocabulary;
- bigram/trigram counts → history counts;
- trigram predecessors → continuation counts;
- continuation counts → lower-order probabilities;
- n-gram count frequencies → `D1`, `D2`, `D3+`;
- history counts and discounts → `lambda`.

So mathematically:

```text
counts_by_order + model settings
        ↓
all MKN prediction probabilities
```

However, `counts_by_order` alone is not enough unless we also know the model settings, such as:

```text
maximum n-gram order
boundary-token conventions
discount-estimation method
supplied discount values, if any
unknown-token handling
```

In our implementation, we additionally store values like:

```text
predecessors
continuation_counts
discounts_by_order
```

These are mostly **derived values**. They are stored separately because recalculating them for every prediction would be slow.

Therefore, the precise answer is:

> `counts_by_order` plus the model configuration can completely define the MKN model, but storing the derived quantities separately makes prediction efficient.

#### The memory is a huge issue in these models, just moving from "n-gram" --> "n+1-gram" is an exponential increase in memory, and because of recursive calculations, the inference time can also get impacted because of huge dictionary.  With a corpus as large (or as small!) as even wikitext-2, we can go till say 5-gram or 7-gram max I guess. What do u say?

Your intuition is correct, but one refinement is important:

> The theoretical number of possible n-grams grows exponentially, but a practical n-gram model stores only the n-grams actually observed in the corpus.

For example, if the vocabulary has 21,569 tokens, possible 7-grams are:

$$
21{,}569^7
$$

which is impossibly large. But WikiText-2 contains only about 2 million training tokens, so at most roughly 2 million 7-gram occurrences can exist. After removing duplicates, the number of distinct 7-grams is smaller—but still potentially large.

Practical consequences:

- **Memory:** every extra order adds another large dictionary.
- **Training time:** counting and storing longer n-grams becomes expensive.
- **Inference:** recursion depth increases from, say, 5 to 7, but this is usually manageable.
- **Generation:** evaluating all vocabulary words is expensive, especially with a large vocabulary.
- **Naive implementation:** repeatedly scanning dictionaries, as our teaching code does, becomes especially slow.

For WikiText-2, a reasonable progression is:

```text
3-gram → good for learning
5-gram → common practical choice
7-gram → possible, but requires careful memory management and pruning
```

So yes: **5-grams are a sensible upper bound for a simple implementation; 7-grams may be feasible with optimized storage, caching, and pruning.**

#### One thing that I did not understand is that as per the paper, do we really use the validation set for tuning the discounts? If yes, why do we do so? Why is training set insufficient to get the D1,D2,D3+ values for each order just like we are anyways doing in the code? How do we decide when to use the validation set to tuning the parameters, and when to not?

Yes—but the paper describes **two legitimate approaches**.

**1. Training-count approach: our current code**

We calculate:

\[
D_1,\;D_2,\;D_{3+}
\]

from training-set count-of-counts:

```text
n1, n2, n3, n4
```

This is what our code does.

The training set is sufficient for this because the formula gives a principled estimate of the discounts directly from the observed count distribution.

The paper calls this practical version `kneser-ney-mod-fix`: its discounts are fixed using training counts, so no held-out tuning is required. [Chen & Goodman paper](https://people.eecs.berkeley.edu/~klein/cs294-5/chen_goodman.pdf)

**2. Validation-tuning approach**

Alternatively, we treat the discounts as adjustable parameters:

```text
D1, D2, D3+
```

We try different values, calculate validation perplexity, and keep the values that give the lowest perplexity:

```text
try discounts
    ↓
train model using training set
    ↓
measure validation perplexity
    ↓
choose best discounts
```

The paper’s tuned version, `kneser-ney-mod`, uses this kind of held-out-data optimization. The paper reports that tuning free smoothing parameters on held-out data can improve performance, while the fixed version avoids needing such tuning. [Chen & Goodman paper](https://u.cs.biu.ac.il/~yogo/courses/mt2014/papers/chen-goodman-99.pdf)

**Why isn’t training data always enough?**

The training-count formula asks:

> “Given the count patterns in the training corpus, what discount seems reasonable?”

Validation tuning asks:

> “Which discount values make the model predict unseen text best?”

These are different goals. The formula is an estimate; validation directly tests generalization.

**When should we use validation tuning?**

Use validation tuning when:

- the method has adjustable hyperparameters;
- you want the best validation performance;
- you have enough validation data;
- you can afford repeated experiments.

Do not tune on validation when:

- you want a simple formula-based implementation;
- you are reproducing the fixed MKN variant;
- the validation set is too small;
- you want to avoid overfitting the validation set.

The test set should remain untouched until the final evaluation.

For our project:

```text
current code:
D1, D2, D3+ estimated from training counts

possible future experiment:
try several discount values and select the best using validation perplexity
```

So the answer is: **the validation set can be used to tune the discounts, but it is not required. Our current code uses the training-count estimation approach.**

#### 5. In any real world implementation for next word prediction nowadays, we don't use MKN model right?

**For general-purpose next-word prediction today, usually no.**

Modern systems such as ChatGPT-style models generally use neural Transformer language models because they can use much longer context and capture meaning beyond a fixed window of 3–7 tokens.

However, MKN is still useful in several real-world situations:

- speech-recognition decoding;
- autocomplete on constrained or low-resource devices;
- domain-specific language models;
- fast baseline models;
- combining with neural models;
- rescoring candidate sentences.

For example, modern speech-recognition systems may combine a neural acoustic model with an n-gram language model during beam search; NVIDIA’s NeMo documentation supports n-gram models through KenLM for this purpose. [NVIDIA NeMo documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/asr/asr_language_modeling.html)

There is also current research on combining n-gram models with neural language models to improve efficiency or exploit exact phrase memorization. [COLM 2024 paper](https://openreview.net/pdf/b6c910d9f431718c2125274d65326ea3ff8a20c6.pdf)

So the practical summary is:

```text
ChatGPT-like text generation → neural Transformer
Speech recognition / constrained systems → MKN may still be useful
Teaching and classical NLP → MKN remains very important
```

MKN is no longer the main engine for large-scale general text generation, but it is still a useful compact component in certain production systems.