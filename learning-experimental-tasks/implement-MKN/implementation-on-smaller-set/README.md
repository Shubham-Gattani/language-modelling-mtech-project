# MKN Simulator on a Smaller Custom Dataset

This is a deliberately small learning implementation of interpolated Modified Kneser–Ney smoothing. It is intended to be understood line by line before moving to WikiText-103.

## Start with the full flow

The simulator follows this path:

```text
toy_corpus.txt
    → one sentence per line
    → add <s> and </s>
    → count unigrams, bigrams, and trigrams
    → count distinct predecessor histories
    → estimate D1, D2, and D3+
    → calculate recursive MKN probabilities
    → print direct/lower-order traces
    → check normalization
```

The main entry point is `main()` in `mkn_simulator.py`.

## Suggested learning order

Read the learning material in this order:

1. `step-by-step-implementation.md`: complete hand walkthrough of the simulator, from corpus loading through recursive MKN probabilities and normalization checks.
2. `lambda-calculation-example.md`: focused example that starts with the final probability goal, lists its dependencies, and calculates lambda values from the bottom upward.
3. `mkn_simulator.py`: revisit the implementation and match each function to the calculations in the notes.
4. `test_mkn_simulator.py`: inspect the tests as executable checks of the concepts.

The lambda example uses a separate three-sentence corpus and fixed teaching discounts. It is a focused calculation exercise, while `step-by-step-implementation.md` follows the simulator's 12-sentence toy corpus and estimated discounts.

## Run the simulator

From this directory:

```bash
python3 mkn_simulator.py
```

Run the tests:

```bash
python3 -m unittest -v
```

## File guide

- `PLAN.md`: scope, assumptions, design, and verification plan.
- `toy_corpus.txt`: the exact synthetic corpus used by the CLI.
- `mkn_simulator.py`: implementation and demonstration entry point.
- `test_mkn_simulator.py`: deterministic tests for the important behaviors.
- `step-by-step-implementation.md`: detailed hand calculation and code walkthrough.
- `lambda-calculation-example.md`: focused lambda calculation organized from goal to prerequisites.
- `artifacts/toy_run.txt`: saved output from one toy-corpus run.

## Important implementation choices

ASSUMPTION: The tokenizer is whitespace-based. It is intentionally simple so that every token boundary can be inspected by hand.

ASSUMPTION: `<s>` and `</s>` are explicit boundary tokens. `<s>` is not a predicted vocabulary item, while `</s>` is included as a possible next token.

ASSUMPTION: MKN discounts are estimated separately for each n-gram order. The toy model supports supplied discounts too, which makes the unit tests exact and easy to calculate by hand.

### Discount buckets

For a seen n-gram count:

```text
count 1       → D1
count 2       → D2
count 3 or up → D3+
```

### Recursive probability flow

For a trigram query such as:

```text
P(sat | the, cat)
```

the code first calculates the direct discounted trigram contribution, then recursively asks for:

```text
P(sat | cat)
```

which eventually uses the continuation probability:

```text
P_cont(sat)
```

Each trace dictionary records the order, history, event type, count, selected discount, direct contribution, lambda, lower-order probability, and final probability.

## What the tests establish

The tests cover:

- sentence-boundary tokenization;
- repeated bigram and trigram counts;
- distinct predecessor counts;
- frequency-of-frequencies discount estimation;
- correct selection of `D1` and `D2` buckets;
- seen n-gram probability with direct and lower-order contributions;
- unseen n-gram recursion;
- trigram-to-bigram recursion;
- normalization over several histories;
- invalid words and overlong histories.

## Moving to WikiText-103 later

This toy implementation intentionally does not solve the engineering problems of WikiText-103. The next version will need explicit decisions about:

- tokenization and preprocessing;
- train/validation/test file handling;
- `<unk>` and vocabulary policy;
- memory-efficient count storage;
- order-specific count-of-count statistics;
- evaluation metrics such as cross-entropy and perplexity;
- runtime and memory measurement.

Those decisions should be added only after the small simulator is understood and verified.
