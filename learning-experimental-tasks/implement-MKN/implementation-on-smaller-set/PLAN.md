# Plan: Simulate Modified Kneser–Ney on a Small Dataset

## Goal

Build a small, readable, dependency-free Python implementation of interpolated Modified Kneser–Ney smoothing. The implementation is a learning simulator, not a production language-modeling library.

The simulator should make the complete flow visible:

```text
toy sentences
    → tokenized sentences with <s> and </s>
    → n-gram counts
    → distinct continuation counts
    → count-dependent discounts
    → recursive MKN probabilities
    → probability traces and normalization checks
```

## Scope

### Included

- A small custom corpus containing repeated phrases, unseen continuations, and enough variety to expose Kneser–Ney continuation counts.
- Bigram and trigram counts.
- Distinct continuation statistics.
- Modified Kneser–Ney discounts $D_1$, $D_2$, and $D_{3+}$.
- Recursive interpolated MKN probabilities.
- Human-readable probability traces showing direct and lower-order contributions.
- Normalization checks for every history used in the demonstration.
- Deterministic unit tests using Python’s standard-library `unittest` module.
- A README explaining the flow and showing how to run the simulator.

### Excluded for now

- Downloading WikiText-103.
- External dependencies.
- Neural language modeling.
- Training a production-scale model.
- Unknown-word handling beyond explicit `<unk>` support being deferred to the WikiText phase.
- Corpus streaming and memory optimization.

## Design

### Files

- `mkn_simulator.py`: the implementation and small command-line demonstration.
- `test_mkn_simulator.py`: deterministic unit tests.
- `README.md`: learner-facing explanation and run instructions.
- `toy_corpus.txt`: the exact synthetic corpus used by the demonstration.
- `artifacts/`: generated, inspectable text reports from the toy run.

### Main flow

1. Read the toy corpus as one sentence per line.
2. Add `<s>` and `</s>` boundaries.
3. Count n-grams for orders 1 through 3.
4. Count distinct preceding histories for each word.
5. Estimate $D_1$, $D_2$, and $D_{3+}$ from frequency-of-frequencies statistics.
6. Calculate the MKN backoff weight for a history from the discounted mass removed from its observed continuations.
7. Recursively calculate $P(w\mid h)$: direct discounted contribution plus lower-order contribution.
8. Print a trace for selected seen and unseen events.
9. Verify that each complete conditional distribution sums to one within a small floating-point tolerance.

## Explicit assumptions

ASSUMPTION: The toy corpus uses whitespace tokenization. WikiText-103 will require a separate tokenizer decision later.

ASSUMPTION: Sentence boundary tokens are part of the vocabulary and are included in n-gram counts.

ASSUMPTION: The demonstration uses interpolated Modified Kneser–Ney and recursively reduces a trigram history to a bigram history and then to a continuation distribution.

ASSUMPTION: The discount-estimation formulas are the standard three-bucket teaching version. A later WikiText implementation may need order-specific discounts and sparse-count engineering.

ASSUMPTION: The toy simulator raises clear errors for impossible probability queries rather than silently inventing vocabulary entries.

## Acceptance criteria

- Running `python3 mkn_simulator.py` completes without external packages.
- The output shows the toy vocabulary, selected counts, discount values, probability traces, and normalization checks.
- The tests cover token boundaries, n-gram counts, continuation counts, discount estimation, seen and unseen recursive probabilities, normalization, and invalid inputs.
- The tests pass with `python3 -m unittest -v`.
- The code starts with a flow comment and uses plain names and step-by-step functions so a learner can read it from top to bottom.
- No network, credentials, production services, or large corpus downloads are used.

## Verification plan

1. Run the focused test module.
2. Run the full local `unittest` discovery command.
3. Run the demonstration and save its output under `artifacts/`.
4. Inspect the generated report for finite probabilities and normalization checks.
5. Run a syntax compilation check.
