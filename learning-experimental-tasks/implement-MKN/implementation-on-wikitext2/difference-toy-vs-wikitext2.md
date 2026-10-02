# Difference Between the Toy MKN Implementation and the WikiText-2 Implementation

## The short answer

The two implementations use the same core interpolated Modified Kneser-Ney
model. The main difference is that the toy version is a small, inspectable
learning simulator, while the WikiText-2 version adds the practical steps
needed to train and evaluate that same model on real corpus files.

```text
toy version:
    small known sentences
    -> MKN counts and probabilities
    -> hand-readable traces and normalization checks

WikiText-2 version:
    download and save Hugging Face dataset once
    -> load saved local dataset
    -> selected train/validation prefixes
    -> MKN counts and probabilities
    -> validation perplexity and normalization checks
```

So this is mostly an engineering and data-handling extension, not a different
smoothing formula.

## 1. What remains the same

Both programs use the same important MKN ideas:

1. Add `<s>` at the beginning and `</s>` at the end of every modeling sequence.
2. Count unigrams, bigrams, and trigrams.
3. Build distinct predecessor sets for continuation probabilities.
4. Use the continuation probability at the unigram level.
5. Estimate separate `D1`, `D2`, and `D3+` discounts for bigrams and trigrams.
6. Calculate the lambda value for each seen history.
7. Recursively combine direct higher-order evidence with lower-order evidence.
8. Check that probabilities over the vocabulary sum to one.

The central inference rule is therefore the same in both folders:

```text
P(word | history)
= direct discounted contribution
  + lambda(history) * lower-order probability
```

For example, a trigram query still follows the same path in both versions:

```text
P(word | previous_word_1, previous_word_2)
    -> P(word | previous_word_2)
    -> P(word)
```

## 2. Difference in the input data

### Toy implementation

The toy version reads `toy_corpus.txt`, which contains only 12 deliberately
chosen sentences. The sentences were designed to make important MKN behavior
visible by hand:

- repeated phrases;
- repeated n-grams with different counts;
- unseen continuations;
- words such as `fly` that occur in a narrow context;
- enough variation to demonstrate distinct predecessor counts.

The learner can list nearly every count and verify the arithmetic manually.

### WikiText-2 implementation

The WikiText-2 version loads the official word-level configuration from
Hugging Face:

```text
dataset: Salesforce/wikitext
config:  wikitext-2-v1
splits:  train, validation, test
```

The corpus is saved locally by `download_wikitext2.py` and is not stored in the
repository.

The WikiText-2 implementation treats each non-empty `text` row from the saved
dataset as one modeling sequence. This is a practical corpus convention, not
necessarily a claim that every row is a complete grammatical sentence.

```text
one non-empty WikiText text row
    -> <s> tokens from that row </s>
```

The toy version uses the same one-line-per-sequence convention, but its lines
are intentionally short and sentence-like.

## 3. Difference in preprocessing

The basic tokenization is intentionally the same:

```python
words = sentence.split()
```

This works because the selected WikiText-2 files are already whitespace-
separated word-level files.

The practical differences are:

| Concern | Toy implementation | WikiText-2 implementation |
|---|---|---|
| Input source | Local `toy_corpus.txt` | Train and validation corpus files |
| Empty rows | Not expected in the toy file | Ignored by `load_huggingface_splits()` |
| Boundaries | Added to each toy sentence | Added to each non-empty WikiText row |
| Tokenization | Simple whitespace split | Also simple whitespace split because files are pre-tokenized |
| Input size | Entire small file is loaded | A bounded prefix can be selected first |
| Vocabulary | Built from the toy training sentences | Built only from the selected training lines |

The line limit is important for learning and safety:

```text
--max-train-lines 5000
--max-valid-lines 500
```

Using `0` means “use all available lines.” This lets us begin with a manageable
experiment before attempting the full split.

## 4. Difference in model construction

The internal model-building sequence is nearly identical.

### Toy version

```text
load toy lines
-> MKNModel.from_sentences(sentences, max_order=3)
-> build counts, vocabulary, predecessor sets, discounts
```

### WikiText-2 version

```text
load saved local training rows
-> take the requested training prefix
-> MKNModel.from_sentences(train_sentences, max_order=3)
-> build counts, vocabulary, predecessor sets, discounts
```

The important difference is what is placed inside `sentences`. The MKN model
does not need to know whether a sequence came from the toy file or WikiText-2;
it receives a list of strings and performs the same count construction.

In both implementations, the stored count structure is conceptually:

```text
ngram_counts[1] -> unigram counts
ngram_counts[2] -> bigram counts
ngram_counts[3] -> trigram counts
```

The difference is scale. In the toy version, these dictionaries are small
enough to inspect directly. In WikiText-2, the same in-memory dictionaries can
contain a very large number of distinct types, so memory use becomes a real
engineering concern.

## 5. Difference in discount estimation

The formula and the three discount buckets are the same:

```text
count 1       -> D1
count 2       -> D2
count 3 or up -> D3+
```

Both versions estimate discounts separately by order:

```text
bigram discounts  -> discounts_by_order[2]
trigram discounts  -> discounts_by_order[3]
```

The difference is the source and reliability of the statistics:

| Aspect | Toy implementation | WikiText-2 implementation |
|---|---|---|
| Count-of-count input | Tiny synthetic count tables | Much larger real-corpus count tables |
| Purpose | Make the calculation easy to inspect | Estimate discounts from realistic sparsity patterns |
| Fallback behavior | Important because a tiny corpus may lack some count classes | Less likely to be needed, but still retained for robustness |
| Interpretation | Mainly pedagogical | Used for an actual validation experiment |

The toy implementation also supports explicitly supplied discounts in its
tests. That makes expected values easy to calculate exactly. The WikiText-2
CLI uses automatically estimated discounts unless the code is changed to pass
custom ones.

## 6. Difference in unknown-word handling

This is one of the most important practical additions.

### Toy version

The toy model rejects a query word that is not in its vocabulary:

```text
P(unknown_word | history)
    -> ValueError
```

This is useful for learning because it prevents the program from silently
introducing a word that was never part of training.

### WikiText-2 version

Validation data may contain words that are not in the vocabulary built from the
selected training lines. Before calculating validation probabilities,
`evaluation_words()` maps such words to `<unk>`:

```text
validation word not in training vocabulary
    -> <unk>
```

The non-raw WikiText-2 files already use `<unk>` for their word-level unknown
words, and the implementation keeps this policy explicit.

Most importantly, validation must not expand the training vocabulary. If a
validation word were simply added to the vocabulary after training, the
evaluation would leak information from validation into the model.

## 7. Difference in evaluation

### Toy version: inspect individual probabilities

The toy program is mainly used to inspect selected calculations:

```text
P(sat | the cat)
P(fly | cat)
```

It prints detailed traces containing the count, selected discount, direct
contribution, lambda, lower-order probability, and final probability.

Its main question is:

> Did the recursive MKN calculation behave as expected?

### WikiText-2 version: evaluate validation perplexity

The WikiText-2 program retains selected traces, but adds a corpus-level metric:
validation perplexity.

For every predictable validation token, it calculates:

```text
log P(current_word | previous_words)
```

It adds the log probabilities and converts their average back to perplexity:

```text
perplexity
= exp(- total_log_probability / number_of_predicted_tokens)
```

The end token `</s>` is also predicted, so every non-empty line contributes a
boundary prediction.

The evaluation question changes from:

```text
Does this one probability calculation look correct?
```

to:

```text
How well does the complete trained model predict held-out text?
```

## 8. Difference in the command-line flow

### Toy command

The toy `main()` knows the corpus path in advance:

```text
script location
    -> toy_corpus.txt beside the script
    -> build model
    -> print demonstration
```

It needs no arguments and no network access.

### WikiText-2 command

The WikiText-2 program accepts a local saved-dataset path and line limits:

```bash
python3 mkn_wikitext2.py \
  --dataset-path data/wikitext-2
```

The command-line flow is:

```text
parse local dataset path and line limits
-> call load_from_disk()
-> load training and validation rows
-> select prefixes
-> train only on the training lines
-> print MKN details
-> calculate validation perplexity
-> run normalization checks
```

The test split is saved as part of the dataset object, but the current
experiment evaluates validation perplexity and does not use the test split in
`main()`.

## 9. Difference in tests

Both test files use the same 12-sentence synthetic corpus so that the core MKN
behavior remains deterministic and easy to verify. The WikiText-2 tests do not
require a downloaded corpus.

The WikiText-2 test file adds checks for its real-corpus support functions:

```text
take_first_lines(...)
evaluation_words(...)
validation_perplexity(...)
```

So its tests cover both categories:

```text
MKN mathematics and recursion
    +
WikiText-2 data loading and evaluation behavior
```

The toy tests focus only on the simulator's MKN behavior and basic input
validation.

## 10. What has not changed conceptually

Moving from the toy corpus to WikiText-2 does not change the meaning of these
objects:

```text
count(history, word)
count(history)
predecessors(word)
continuation probability
discount D1, D2, D3+
lambda(history)
```

For example, `lambda(history)` still means:

```text
the fraction of probability mass removed from observed continuations
that is handed to the lower-order model
```

The model is still doing the same three-level reasoning:

```text
trigram evidence
    -> bigram evidence
        -> continuation evidence
```

Only the amount and messiness of the evidence have changed.

## 11. What new difficulties appear with WikiText-2

The real corpus introduces practical issues that the toy corpus intentionally
does not expose:

1. **Scale:** count dictionaries become much larger.
2. **Memory:** storing every n-gram in ordinary Python dictionaries may become
   expensive.
3. **Runtime:** building counts and repeatedly calculating validation
   probabilities takes longer.
4. **Sparsity:** many possible trigrams are unseen, so recursive lower-order
   behavior matters much more.
5. **Unknown words:** validation needs an explicit `<unk>` policy.
6. **Data splits:** training and validation must remain separate.
7. **Evaluation:** individual examples are no longer enough; perplexity gives a
   corpus-level summary.
8. **Reproducibility:** the exact file variant, line limits, and preprocessing
   assumptions must be recorded.

These are implementation and experiment-design problems around MKN. They are
not changes to the core MKN equation.

## Final mental model

```text
Toy corpus:
    learn whether the MKN machine works.

WikiText-2:
    run the same MKN machine on realistic data,
    while handling files, scale, unknown words, and evaluation.
```

The safest way to understand the WikiText-2 implementation is therefore:

1. Understand every MKN calculation in `mkn_simulator.py`.
2. Notice that `mkn_wikitext2.py` preserves those calculations.
3. Study the additional corpus and evaluation functions around them.

The toy implementation teaches the mechanism. The WikiText-2 implementation
tests that mechanism in a more realistic experimental setting.

## Files compared

Toy implementation:

- [`mkn_simulator.py`](../implementation-on-smaller-set/mkn_simulator.py)
- [`README.md`](../implementation-on-smaller-set/README.md)
- [`PLAN.md`](../implementation-on-smaller-set/PLAN.md)
- [`test_mkn_simulator.py`](../implementation-on-smaller-set/test_mkn_simulator.py)

WikiText-2 implementation:

- [`mkn_wikitext2.py`](mkn_wikitext2.py)
- [`download_wikitext2.py`](download_wikitext2.py)
- [`requirements.txt`](requirements.txt)
- [`README.md`](README.md)
- [`PLAN.md`](PLAN.md)
- [`faqs.md`](faqs.md)
- [`test_mkn_wikitext2.py`](test_mkn_wikitext2.py)
