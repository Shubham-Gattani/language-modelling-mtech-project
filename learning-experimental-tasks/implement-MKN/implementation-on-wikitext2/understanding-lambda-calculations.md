```text
self.ngram_counts = counts_by_order = {
    1: {
        ("<s>",): 2,
        ("The",): 2,
        ("cat",): 2,
        ("sat",): 1,
        ("slept",): 1,
        ("</s>",): 2,
    },
    2: {
        ("<s>", "The"): 2,
        ("The", "cat"): 2,
        ("cat", "sat"): 1,
        ("sat", "</s>"): 1,
        ("cat", "slept"): 1,
        ("slept", "</s>"): 1,
    },
    3: {
        ("<s>", "The", "cat"): 2,
        ("The", "cat", "sat"): 1,
        ("cat", "sat", "</s>"): 1,
        ("The", "cat", "slept"): 1,
        ("cat", "slept", "</s>"): 1,
    },
}
```

```python

    def lambda_for_history(self, history):
        """Return the total probability mass removed after an observed history. e.g. lambda_for_history(("the", "cat")) returns 0.5 for the toy corpus because the bigram ("the", "cat") is discounted by 0.5."""

        history = tuple(history)
        if not history:
            raise ValueError("the empty history has no higher-order lambda")
        order = len(history) + 1
        if order > self.max_order:
            raise ValueError("history is longer than the model supports")

        history_count = self.history_count(history) # Count of the history e.g. history_count(("the", "cat")) returns 2 for the toy corpus because the bigram ("the", "cat") occurs twice.
        if history_count == 0: # If the history was never observed, the model backs off to a lower order and assigns all probability mass to that lower order. Therefore, the lambda for an unseen history is 1.0.
            return 1.0

        removed_count = 0.0
        for ngram, count in self.ngram_counts[order].items():
            if ngram[:-1] == history:
                removed_count += self.discount_for_count(count, order)
        return removed_count / history_count
```

Assume:

```python
self.max_order = 3
```

and:

```python
self.discounts_by_order = {
    2: (0.5, 1.0, 1.5),  # bigram D1, D2, D3+
    3: (2/3, 1.0, 1.5),  # trigram D1, D2, D3+
}
```

# FULL EXPLANATION WITH EXAMPLES THAT TRIGGER DIFFERENT BRANCHES

## 1. `history = ()`: empty history

```python
if not history:
    raise ValueError(...)
```

Example:

```python
lambda_for_history(())
```

Since there is no history, the function cannot calculate a higher-order lambda.

Result:

```text
ValueError
```

---

## 2. `history = ("The",)`: seen one-word history

```python
history = ("The",)
order = len(history) + 1
       = 1 + 1
       = 2
```

So the function examines bigrams.

History count:

```text
history_count = c(The) = 2
```

The loop checks every bigram:

```text
("<s>", "The")  -> prefix is ("<s>",), not a match
("The", "cat")  -> prefix is ("The",), match
("cat", "sat")  -> no match
...
```

The matching bigram is:

```text
("The", "cat") with count 2
```

For count `2`, the bigram discount is `D2 = 1.0`.

```text
removed_count = 1.0
lambda = removed_count / history_count
       = 1.0 / 2
       = 0.5
```

---

## 3. `history = ("cat",)`: seen history with two continuations

```python
history = ("cat",)
order = 2
history_count = c(cat) = 2
```

Matching bigrams:

```text
("cat", "sat")    count 1 -> D1 = 0.5
("cat", "slept")  count 1 -> D1 = 0.5
```

The loop adds both discounts:

```text
removed_count = 0.0
removed_count += 0.5  # cat -> sat
removed_count += 0.5  # cat -> slept
removed_count = 1.0
```

Therefore:

```text
lambda = 1.0 / 2
       = 0.5
```

This demonstrates that the code adds one discount per matching continuation.

---

## 4. `history = ("The", "cat")`: seen two-word history

```python
history = ("The", "cat")
order = len(history) + 1
       = 2 + 1
       = 3
```

So the function examines trigrams.

History count:

```text
history_count = c(The, cat) = 2
```

Matching trigrams:

```text
("The", "cat", "sat")    count 1 -> trigram D1 = 2/3
("The", "cat", "slept")  count 1 -> trigram D1 = 2/3
```

Therefore:

```text
removed_count = 2/3 + 2/3
              = 4/3
```

Then:

```text
lambda = (4/3) / 2
       = 4/6
       = 2/3
```

---

## 5. `history = ("cat", "sat")`: seen two-word history with one continuation

```python
history = ("cat", "sat")
order = 3
history_count = c(cat, sat) = 1
```

Matching trigram:

```text
("cat", "sat", "</s>") with count 1
```

Its discount is:

```text
D1 = 2/3
```

Therefore:

```text
removed_count = 2/3
lambda = (2/3) / 1
       = 2/3
```

---

## 6. `history = ("sat", "The")`: unseen history

```python
history = ("sat", "The")
order = 3
```

The bigram `("sat", "The")` does not exist.

Therefore:

```text
history_count = 0
```

This branch executes:

```python
if history_count == 0:
    return 1.0
```

Result:

```text
lambda(("sat", "The")) = 1.0
```

Meaning: this history has no reliable trigram evidence, so all probability mass is delegated to the lower-order model.

---

## 7. History longer than the model supports

Since `max_order = 3`, the longest allowed history has length `2`.

Example:

```python
lambda_for_history(("The", "cat", "sat"))
```

Then:

```text
order = len(history) + 1
       = 3 + 1
       = 4
```

Since `4 > max_order`, this branch executes:

```python
raise ValueError("history is longer than the model supports")
```

Result:

```text
ValueError
```

## Summary

```text
history                  result
------------------------------------------------
()                       ValueError
("The",)                 0.5
("cat",)                 0.5
("The", "cat")           2/3
("cat", "sat")           2/3
("sat", "The")           1.0
("The", "cat", "sat")   ValueError
```