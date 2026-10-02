# Guided Tutorial: Modified Kneser-Ney on WikiText-2

This file explains the implementation in four small stages:

1. preprocessing and data loading
2. model training: building MKN statistics
3. inference: calculating probabilities for seen and unseen n-grams
4. measuring validation perplexity

The code is not training a neural network. Here, **training** means reading the
training text and building the counts and other tables needed by Modified
Kneser-Ney (MKN).

# Value of variables used for explanation via toy-corpus

This is a reference snapshot of the toy-corpus values introduced through
Step 4. We will update this section after every later step when a new model
value is created or an existing value changes.

The toy corpus is:

```python
sentences = [
    "The cat sat",
    "The cat slept",
]
```

The boundary-token constants are:

```python
START_TOKEN = "<s>"
END_TOKEN = "</s>"
```

The model is an order-3 model:

```python
max_order = 3
```

After tokenization, the two modeling sequences are:

```python
self.tokenized_sentences = [
    ["<s>", "The", "cat", "sat", "</s>"],
    ["<s>", "The", "cat", "slept", "</s>"],
]
```

The vocabulary contains every token except `<s>` because `<s>` is used as a
history marker, not as a word to predict:

```python
self.vocabulary = ("</s>", "The", "cat", "sat", "slept")
```

The n-gram count dictionaries are stored in the model as
`self.ngram_counts`. During construction, the temporary dictionary has the
name `counts_by_order`; they refer to the same final count structure:

```python
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

The predecessor sets are stored in `self.predecessors_by_word`. Inside
`_build_predecessor_sets()`, the local dictionary is called `predecessors`:

```python
self.predecessors_by_word = predecessors = {
    "The": {"<s>"},
    "cat": {"The"},
    "sat": {"cat"},
    "slept": {"cat"},
    "</s>": {"sat", "slept"},
}
```

The continuation count for each vocabulary word is the size of its
predecessor set:

```python
continuation_count = {
    "The": 1,
    "cat": 1,
    "sat": 1,
    "slept": 1,
    "</s>": 2,
}
```

The continuation denominator is the number of distinct bigram types:

```python
self.continuation_denominator = len(self.ngram_counts[2])  # 6
```

Therefore, the Kneser-Ney continuation probabilities are:

```python
continuation_probability = {
    "The": 1 / 6,
    "cat": 1 / 6,
    "sat": 1 / 6,
    "slept": 1 / 6,
    "</s>": 2 / 6,
}
```

Their total is:

```text
1/6 + 1/6 + 1/6 + 1/6 + 2/6 = 1
```

The discount values estimated from this toy corpus are:

```python
self.discounts_by_order = {
    2: (0.5, 1.0, 1.5),
    3: (2 / 3, 1.0, 1.5),
}
```

The tuple for each order is always in this order:

```text
(D1, D2, D3_plus)
```

The recursive probability values calculated later for the toy example are:

```text
continuation_probability["sat"] = 1 / 6
probability("sat", ("cat",)) = 1 / 3
probability("sat", ("The", "cat")) = 7 / 18
probability("slept", ("cat", "sat")) = 1 / 18
probability("sat", ("sat", "The")) = 1 / 12
```

For the worked validation row in Step 9, the model evaluates four predicted
tokens and obtains:

```text
validation sentence: "The cat sat"
predicted tokens: 4
sentence probability: 6517 / 46656
perplexity: approximately 1.6357
```

The normalization check should return `1` for every supported history:

```python
normalization_total(()) = 1
normalization_total(("The",)) = 1
normalization_total(("cat",)) = 1
normalization_total(("The", "cat")) = 1
```

## Step 1: Start at the entry point

The program starts at the bottom of `mkn_wikitext2.py`:

```python
if __name__ == "__main__":
    main()
```

This means: when we run the file directly with Python, call `main()`.

### What `main()` does

The complete high-level flow is:

```text
main()
  -> read command-line arguments
  -> load the locally saved WikiText-2 splits
  -> choose the requested training and validation lines
  -> build an MKN model from the training lines
  -> print example probabilities, perplexity, and normalization checks
```

In code, the important calls appear in this order:

```python
dataset_splits = load_saved_huggingface_splits(args.dataset_path)

train_sentences = take_first_lines(
    dataset_splits["train"], args.max_train_lines
)

validation_sentences = take_first_lines(
    dataset_splits["validation"], args.max_valid_lines
)

model = MKNModel.from_sentences(train_sentences, max_order=3)

print_demo(model, validation_sentences)
```

We will study each call later. For now, read the code as a pipeline: **data
enters from the saved dataset, training text builds the model, and validation
text is used only after the model has been built.**

There is one important detail here: `dataset_splits` contains three splits,
namely `train`, `validation`, and `test`, but the current `main()` selects only
the training and validation rows. Therefore, the test split is loaded and
checked by the loader, but it is not yet passed to `print_demo()`.

### Command-line settings

The program accepts these important settings:

```text
--dataset-path       where the saved WikiText-2 dataset is located
--max-train-lines    how many non-empty training rows to use
--max-valid-lines    how many non-empty validation rows to use
```

If we run the program without extra arguments, the defaults are:

```text
dataset path:       data/wikitext-2
training rows:      5000
validation rows:    500
maximum n-gram:     3
```

The value `3` means that this experiment is a trigram model. It can use:

```text
unigram information: P(word)
bigram information:  P(word | previous word)
trigram information: P(word | previous two words)
```

The line limit exists so that we can first run a small, understandable
experiment. Setting a limit to `0` means “use all rows in that split.”

### Why training, validation, and test are separate

The training rows are used to construct the model's statistics. The validation
rows are kept aside and used to test how well the already-built model predicts
new text while we are still developing the experiment. The test rows are
reserved for the final evaluation after we have stopped making choices.

For example:

```text
training rows:
    used to count n-grams and calculate MKN parameters

validation rows:
    not used to build those counts
    used while developing, comparing, or checking the experiment

test rows:
    not used to build counts
    not used to choose settings
    used once for the final reported perplexity
```

If we used validation rows while building the counts, the evaluation would no
longer be a fair test of prediction on unseen text.

Similarly, if we repeatedly changed the implementation or chose discount
settings based on test perplexity, the test set would gradually influence our
decisions. It would no longer be a genuinely unseen final test.

### Why does the current script measure validation perplexity only?

The current script is intentionally a first learning version. Its discounts
are estimated directly from the training counts, and it does not yet perform a
hyperparameter search or compare several model choices. For that narrow
purpose, validation perplexity gives us a convenient number to inspect while
we learn the implementation.

However, this means the current script is not yet following the complete
train-validation-test evaluation workflow. **A more complete version should**:

```text
1. build the MKN model from train
2. use validation to compare choices, if choices need to be made
3. freeze the implementation and settings
4. calculate final perplexity on test
```

If there are no choices to tune, we can still report both validation and test
perplexity. The test value should be treated as the final headline result.

### What does `MKNModel.from_sentences(...)` mean?

This line:

```python
model = MKNModel.from_sentences(train_sentences, max_order=3)
```

creates the model from the training sentences. The class method eventually
calls the constructor, which performs the initial model-building work:

```text
training sentences
  -> tokenize each sentence and add <s> and </s>
  -> count unigrams, bigrams, and trigrams
  -> find distinct predecessor words
  -> calculate or receive D1, D2, and D3+ discounts
  -> store everything needed for probability calculation
```

So `model` is not merely an empty object. After this line, it contains the
learned MKN statistics for the selected training rows.

### A tiny picture of the data flow

Imagine the saved dataset contains many rows:

```text
saved WikiText-2 dataset
  -> train split:       [row 1, row 2, row 3, ..., row 5000, ...]
  -> first 5000 rows:   [row 1, row 2, row 3, ..., row 5000]
  -> MKNModel.from_sentences(...)
  -> counts and MKN parameters
```

The validation rows follow a separate path:

```text
saved WikiText-2 dataset
  -> validation split:      [row 1, row 2, ..., row 500]
  -> kept aside
  -> passed to print_demo(...)
  -> used for perplexity after training
```

The test rows are currently loaded but stop before evaluation:

```text
saved WikiText-2 dataset
  -> test split
  -> loaded and checked by load_saved_huggingface_splits()
  -> not yet passed to validation_perplexity()
```

### What we know after Step 1

At this point, we know the overall control flow:

- `main()` is the entry point.
- The saved dataset is loaded from disk; the main program does not download it.
- Training rows build the MKN model.
- Validation rows currently evaluate the finished model during this learning
  stage.
- The test split is present, but the current script does not yet calculate test
  perplexity.
- `max_order=3` means the model uses unigrams, bigrams, and trigrams.
- `print_demo()` reports inference results and validation perplexity.

We have not yet calculated any n-gram counts. That is the next step: follow
the preprocessing path from a raw WikiText-2 row to the tokens that the model
actually counts.

## Step 2: Preprocessing the text

Preprocessing converts the saved dataset into the exact token sequences that
the model will use. The preprocessing path is:

```text
saved DatasetDict
  -> read the text field from each row
  -> remove surrounding whitespace
  -> discard empty rows
  -> select the requested prefix of rows
  -> split each training row into words
  -> add <s> at the beginning and </s> at the end
```

The important idea is that the model does not count raw Python strings. It
counts token lists such as:

```text
raw row:
    The cat sat

after tokenization and boundary markers:
    ["<s>", "The", "cat", "sat", "</s>"]
```

The two boundary tokens are part of the model's input representation. `<s>`
means “start of a sequence” and `</s>` means “end of a sequence.” They allow
the model to learn facts such as:

```text
P(The | <s>)
P(</s> | sat)
```

Without `<s>`, the model would not explicitly model what can begin a row.
Without `</s>`, it would not model when a row ends.

### 2.1 Load the saved dataset

`main()` first calls:

```python
dataset_splits = load_saved_huggingface_splits(args.dataset_path)
```

The function `load_saved_huggingface_splits()` performs four simple jobs:

```text
1. Open the DatasetDict saved by download_wikitext2.py.
2. Confirm that train, validation, and test splits exist.
3. Read the text field from every row.
4. Strip whitespace and keep only non-empty text rows.
```

For example, suppose a split contains these rows:

```text
row 1: "  The cat sat  "
row 2: ""
row 3: "   "
row 4: "The dog ran"
```

After loading and cleaning, the returned list is:

```python
["The cat sat", "The dog ran"]
```

The empty rows are ignored. They do not represent a sequence that should
contribute n-grams to this experiment.

The loader returns a dictionary shaped like this:

```python
{
    "train": ["first training row", "second training row", ...],
    "validation": ["first validation row", "second validation row", ...],
    "test": ["first test row", "second test row", ...],
}
```

At this stage, the values are still ordinary strings. Boundary tokens have not
yet been added.

### 2.2 Select a manageable number of rows

Next, `main()` calls `take_first_lines()`:

```python
train_sentences = take_first_lines(
    dataset_splits["train"], args.max_train_lines
)
```

If `max_train_lines` is `5000`, the function returns the first 5000 available
training rows. If it is `0`, it returns all training rows.

The same operation is performed for validation rows. This limit controls the
size of the experiment; it does not change the meaning of an individual row.

For a tiny example:

```python
all_training_rows = [
    "The cat sat",
    "The dog ran",
    "Birds fly",
]

take_first_lines(all_training_rows, 2)
```

returns:

```python
["The cat sat", "The dog ran"]
```

The third row is not used in this particular small run.

### 2.3 Tokenize one training row

When the model is created, its constructor applies `tokenize_sentence()` to
each selected training row:

```python
self.tokenized_sentences = [
    tokenize_sentence(sentence) for sentence in sentences
]
```

For one row, the calculation is:

```text
input string:
    "The cat sat"

sentence.split():
    ["The", "cat", "sat"]

add boundaries:
    ["<s>", "The", "cat", "sat", "</s>"]
```

For two rows, the result is kept as two separate sequences:

```python
[
    ["<s>", "The", "cat", "sat", "</s>"],
    ["<s>", "The", "dog", "ran", "</s>"],
]
```

This separation matters. The model must not create a bigram such as
`("</s>", "<s>")` by joining the end of one row to the beginning of the next
row. Each row gets its own start and end markers.

### 2.4 Why `<s>` is not in the predicted vocabulary

The model removes `<s>` from its prediction vocabulary:

```python
vocabulary.discard(START_TOKEN)
```

This is because `<s>` is a context marker, not a word that the model should
predict as the next token. The model may use `<s>` in a history, for example
`P(The | <s>)`, but it should not ask for `P(<s> | history)`.

The end marker `</s>` is different. It is predicted at the end of every row,
so it remains in the vocabulary and contributes a probability such as
`P(</s> | sat)`.

### What we know after Step 2

After preprocessing:

- rows are loaded from the locally saved dataset;
- empty rows are removed;
- optional row limits make the first experiment manageable;
- each training row becomes a separate token sequence;
- `<s>` is added at the beginning of every sequence;
- `</s>` is added at the end of every sequence;
- the model is ready to count unigrams, bigrams, and trigrams.

No MKN probability has been calculated yet. The next step is to slide windows
of sizes 1, 2, and 3 over these token sequences and build
`counts_by_order`.

## Step 3: Build `counts_by_order`

The model now turns token sequences into frequency tables. The relevant code
is:

```python
counts_by_order = {
    order: {} for order in range(1, self.max_order + 1)
}

for tokens in self.tokenized_sentences:
    for order in counts_by_order:
        for start in range(len(tokens) - order + 1):
            ngram = tuple(tokens[start : start + order])
            counts = counts_by_order[order]
            counts[ngram] = counts.get(ngram, 0) + 1
```

Because `max_order=3`, this creates three dictionaries:

```python
counts_by_order = {
    1: {},  # unigram counts
    2: {},  # bigram counts
    3: {},  # trigram counts
}
```

The key `1`, `2`, or `3` tells us the n-gram order. Inside each dictionary,
the key is the n-gram itself and the value is how many times that n-gram
occurred.

### 3.1 A small example

Use these two training sequences:

```text
The cat sat
The cat slept
```

After preprocessing, they are:

```python
sentence_1 = ["<s>", "The", "cat", "sat", "</s>"]
sentence_2 = ["<s>", "The", "cat", "slept", "</s>"]
```

We will now calculate all three orders by hand.

### 3.2 Order 1: unigrams

A unigram is a window containing one token.

For the first sequence:

```text
<s>     The     cat     sat     </s>
```

we see:

```text
(<s>)      once
(The)      once
(cat)      once
(sat)      once
(</s>)     once
```

For the second sequence:

```text
<s>     The     cat     slept     </s>
```

we see:

```text
(<s>)      once more
(The)      once more
(cat)      once more
(slept)    once
(</s>)     once more
```

Combining both sequences gives:

```python
counts_by_order[1] = {
    ("<s>",): 2,
    ("The",): 2,
    ("cat",): 2,
    ("sat",): 1,
    ("slept",): 1,
    ("</s>",): 2,
}
```

The count of `("cat",)` is `2` because `cat` appears once in each sequence.
The count of `("sat",)` is `1` because it appears only in the first sequence.

### 3.3 Order 2: bigrams

A bigram is a window containing two consecutive tokens. For the first
sequence, the windows are:

```text
(<s>, The)
(The, cat)
(cat, sat)
(sat, </s>)
```

For the second sequence, the windows are:

```text
(<s>, The)
(The, cat)
(cat, slept)
(slept, </s>)
```

Now combine identical bigrams:

```python
counts_by_order[2] = {
    ("<s>", "The"): 2,
    ("The", "cat"): 2,
    ("cat", "sat"): 1,
    ("sat", "</s>"): 1,
    ("cat", "slept"): 1,
    ("slept", "</s>"): 1,
}
```

Notice the difference between a bigram's **count** and the number of distinct
bigrams:

```text
count of ("The", "cat") = 2
number of distinct bigram types = 6
```

The dictionary contains six different bigram keys, but some keys have counts
larger than one.

### 3.4 Order 3: trigrams

A trigram is a window containing three consecutive tokens. For the first
sequence, the windows are:

```text
(<s>, The, cat)
(The, cat, sat)
(cat, sat, </s>)
```

For the second sequence, they are:

```text
(<s>, The, cat)
(The, cat, slept)
(cat, slept, </s>)
```

Therefore:

```python
counts_by_order[3] = {
    ("<s>", "The", "cat"): 2,
    ("The", "cat", "sat"): 1,
    ("cat", "sat", "</s>"): 1,
    ("The", "cat", "slept"): 1,
    ("cat", "slept", "</s>"): 1,
}
```

The trigram `("<s>", "The", "cat")` has count `2` because both sequences
begin with those three tokens.

### 3.5 Why does the loop use `len(tokens) - order + 1`?

Suppose a token sequence has length `5`:

```text
[<s>, The, cat, sat, </s>]
```

For bigrams (`order=2`), the number of valid windows is:

```text
5 - 2 + 1 = 4
```

Those four windows start at positions `0`, `1`, `2`, and `3`. A window
starting at position `4` would need one more token and would run past the end.

For trigrams (`order=3`), the number is:

```text
5 - 3 + 1 = 3
```

Those three windows start at positions `0`, `1`, and `2`.

The `+1` is needed because both endpoints are included. In general:

```text
number of windows = sequence length - window length + 1
```

### 3.6 What does `counts.get(ngram, 0) + 1` do?

This line handles both cases without a separate membership check:

```python
counts[ngram] = counts.get(ngram, 0) + 1
```

If the n-gram has not appeared before:

```text
counts.get(ngram, 0) = 0
new count = 0 + 1 = 1
```

If it has already appeared twice:

```text
counts.get(ngram, 0) = 2
new count = 2 + 1 = 3
```

So the dictionary gradually records occurrence counts as the sequences are
processed.

### 3.7 Why are the keys tuples?

The tokens `"The"` and `"cat"` together form one bigram. A tuple keeps them
together as one dictionary key:

```python
("The", "cat")
```

This is different from storing the words separately:

```python
"The"
"cat"
```

The tuple preserves both the identity and the order of the n-gram. Thus:

```text
("The", "cat") != ("cat", "The")
```

That distinction is essential because language-model probabilities depend on
word order.

### What we know after Step 3

For every order from `1` to `3`, the model now has a dictionary mapping each
distinct n-gram to its occurrence count:

```text
counts_by_order[1] -> unigram counts
counts_by_order[2] -> bigram counts
counts_by_order[3] -> trigram counts
```

These are ordinary occurrence counts. MKN has not discounted anything yet.
The next model-building step is to derive the predecessor sets and
continuation counts needed for the Kneser-Ney lower-order distribution.

## Step 4: Build predecessor sets and continuation counts

Kneser-Ney does not use ordinary unigram frequency as its lowest-order
distribution. Instead, it asks:

> How many different words have appeared immediately before this word?

This quantity is called the word's **continuation count**.

For a word `w`, write its continuation count as:

```text
N1+(· w) = number of distinct words that precede w
```

The dot `·` means “we do not care which predecessor it is; count the distinct
predecessors.”

### 4.1 The code that builds predecessor sets

The model uses the distinct bigram keys from `counts_by_order[2]`:

```python
def _build_predecessor_sets(self):
    predecessors = {}
    for bigram in self.ngram_counts[2]:
        previous_word, word = bigram
        if word not in predecessors:
            predecessors[word] = set()
        predecessors[word].add(previous_word)
    return predecessors
```

Important detail: the loop goes through the **dictionary keys**, not through
the bigram counts as repeated events. Therefore, it sees each distinct bigram
type once.

For example, even though `("The", "cat")` occurred twice, the loop processes
the bigram type `("The", "cat")` once. We want a set of distinct
predecessors, not a list containing repeated copies.

### 4.2 Calculate the predecessor sets by hand

Recall the distinct bigram types from Step 3:

```text
(<s>, The)
(The, cat)
(cat, sat)
(sat, </s>)
(cat, slept)
(slept, </s>)
```

Read each bigram as an arrow from predecessor to current word:

```text
<s>   -> The
The   -> cat
cat   -> sat
sat   -> </s>
cat   -> slept
slept -> </s>
```

Now group arrows by the word on the right:

```python
predecessors_by_word = {
    "The": {"<s>"},
    "cat": {"The"},
    "sat": {"cat"},
    "slept": {"cat"},
    "</s>": {"sat", "slept"},
}
```

The curly braces represent sets. A set stores each item only once.

For example, `"</s>"` has two distinct predecessors:

```text
sat   -> </s>
slept -> </s>
```

Therefore:

```text
predecessors_by_word["</s>"] = {"sat", "slept"}
N1+(· </s>) = 2
```

### 4.3 Continuation counts

The function `continuation_count(word)` returns the size of the predecessor
set:

```python
def continuation_count(self, word):
    self._check_word(word)
    return len(self.predecessors_by_word[word])
```

For our example:

```text
word       distinct predecessors       continuation count
----------------------------------------------------------
The        {<s>}                       1
cat        {The}                       1
sat        {cat}                       1
slept      {cat}                       1
</s>        {sat, slept}                2
```

The important contrast is between ordinary occurrence counts and continuation
counts:

```text
ordinary unigram count of cat:       2
distinct predecessors of cat:       1

ordinary unigram count of </s>:     2
distinct predecessors of </s>:     2
```

`cat` appeared twice, but both appearances followed the same word, `The`.
So Kneser-Ney says that `cat` has only one type of continuation context, not
two different contexts.

### 4.4 The continuation denominator

The model stores:

```python
self.continuation_denominator = len(self.ngram_counts[2])
```

In our example, there are six distinct bigram types, so:

```text
continuation_denominator = 6
```

The Kneser-Ney continuation probability is:

```text
P_KN(word) = N1+(· word) / N1+(· ·)
```

Here, `N1+(· ·)` is the total number of distinct bigram types. Thus:

```text
P_KN(The)  = 1 / 6
P_KN(cat)  = 1 / 6
P_KN(sat)  = 1 / 6
P_KN(slept) = 1 / 6
P_KN(</s>)  = 2 / 6
```

The sum is:

```text
1/6 + 1/6 + 1/6 + 1/6 + 2/6 = 6/6 = 1
```

So this is a valid probability distribution over the model vocabulary.

### 4.5 Why this solves the “Francisco” problem

Suppose the training data contains:

```text
San Francisco
San Francisco
San Francisco
New York
New York
```

The ordinary unigram count of `Francisco` is `3`, while the ordinary unigram
count of `York` is `2`. A raw-frequency lower-order model might therefore give
`Francisco` a relatively large probability.

But their distinct predecessor sets are:

```text
Francisco -> {San}
York      -> {New}
```

Both words have one distinct predecessor. Kneser-Ney therefore recognizes that
neither word is broadly distributed across different contexts. Repeated
occurrences inside the same phrase do not make a word a good general
continuation.

Now compare a word that appears after several different words:

```text
people like music
children like music
birds hear music
```

The word `music` has predecessors `{like, hear}` in this small example, so its
continuation count is `2`. Its lower-order probability reflects the variety of
contexts in which it occurs, not merely how many times it occurred inside one
repeated phrase.

### 4.6 Why the denominator counts bigram types, not bigram tokens

Our two sequences contain eight bigram occurrences in total:

```text
4 bigram positions in sentence 1
4 bigram positions in sentence 2
---------------------------------
8 bigram tokens
```

But only six of them are distinct types because `("<s>", "The")` and
`("The", "cat")` each repeat. Kneser-Ney uses:

```text
number of distinct bigram types = 6
```

not:

```text
total number of bigram occurrences = 8
```

This is why the code uses:

```python
len(self.ngram_counts[2])
```

The dictionary length counts distinct keys.

### What we know after Step 4

The model now has the lower-order structural information needed by Kneser-Ney:

```text
predecessors_by_word[word]
    -> set of distinct words that precede word

continuation_count(word)
    -> size of that set

continuation_denominator
    -> total number of distinct bigram types

continuation_probability(word)
    -> continuation count divided by the denominator
```

This continuation probability will be the base distribution used at the
bottom of the recursive MKN probability calculation. The next step is to
understand how the model estimates the discount values `D1`, `D2`, and `D3+`.

## Step 5: Estimate `D1`, `D2`, and `D3+`

Before MKN calculates probabilities, it needs to decide how much probability
mass to remove from observed n-grams. This is the **discounting** step.

For an observed n-gram with count `c`, absolute discounting changes the count
approximately like this:

```text
original count       discounted count
      c          ->        c - D
```

The removed amount is later redistributed through a lower-order model. We will
study that redistribution later. For now, this step answers:

```text
Which D should be used for an observed count?
```

### 5.1 Why are there three discounts?

The code uses three discount buckets:

```text
count = 1       -> use D1
count = 2       -> use D2
count >= 3      -> use D3+
```

The function that selects the bucket is:

```python
def discount_for_count(count, discounts):
    if count <= 0:
        return 0.0
    if count == 1:
        return discounts[0]
    if count == 2:
        return discounts[1]
    return discounts[2]
```

For example, if `discounts = (0.5, 1.0, 1.5)`, then:

```text
discount_for_count(1, discounts) = 0.5
discount_for_count(2, discounts) = 1.0
discount_for_count(3, discounts) = 1.5
discount_for_count(8, discounts) = 1.5
```

The last two calls both use `D3+` because every count of three or more belongs
to the same bucket.

### 5.2 What is frequency-of-frequencies?

To estimate the discounts, the function first asks how many distinct n-gram
types have each count. This is called a **frequency-of-frequencies** table:

```text
frequency_of_frequencies[r]
    = number of distinct n-gram types whose count is exactly r
```

This is different from the ordinary n-gram count table. For example, if the
distinct n-gram counts are:

```text
[2, 2, 1, 1, 1, 1]
```

then:

```text
n1 = 4    # four n-gram types occur once
n2 = 2    # two n-gram types occur twice
n3 = 0
n4 = 0
```

The code builds the table like this:

```python
frequency_of_frequencies = {}
for count in ngram_counts:
    frequency_of_frequencies[count] = (
        frequency_of_frequencies.get(count, 0) + 1
    )
```

Then it reads the four relevant values:

```python
n1 = frequency_of_frequencies.get(1, 0)
n2 = frequency_of_frequencies.get(2, 0)
n3 = frequency_of_frequencies.get(3, 0)
n4 = frequency_of_frequencies.get(4, 0)
```

The `.get(..., 0)` means “return the stored value, or return zero if that
count class does not exist.”

### 5.3 Bigram discounts on the toy corpus

From Step 3, the bigram count values are:

```text
[2, 2, 1, 1, 1, 1]
```

Therefore:

```text
n1 = 4
n2 = 2
n3 = 0
n4 = 0
```

The code first calculates:

```text
y = n1 / (n1 + 2 * n2)
```

Substitute the values:

```text
y = 4 / (4 + 2 * 2)
  = 4 / 8
  = 0.5
```

Now calculate `D1`:

```text
D1 = 1 - 2 * y * n2 / n1
   = 1 - 2 * 0.5 * 2 / 4
   = 1 - 2 / 4
   = 0.5
```

The standard formula for `D2` needs `n3`:

```text
D2 = 2 - 3 * y * n3 / n2
```

Our toy corpus has `n3 = 0`, so this tiny-corpus implementation uses its
documented fallback:

```text
D2 = 1.0
```

The standard `D3+` formula needs both `n3` and `n4`:

```text
D3+ = 3 - 4 * y * n4 / n3
```

Since `n3 = 0`, the formula cannot be used. The implementation uses:

```text
D3+ = 1.5
```

Thus, the bigram discount tuple is:

```python
bigram_discounts = (D1, D2, D3_plus)
                   = (0.5, 1.0, 1.5)
```

### 5.4 Trigram discounts on the toy corpus

The trigram count values from Step 3 are:

```text
[2, 1, 1, 1, 1]
```

Therefore:

```text
n1 = 4    # four trigram types occur once
n2 = 1    # one trigram type occurs twice
n3 = 0
n4 = 0
```

The calculation is performed separately for trigrams because their count
distribution may differ from the bigram distribution:

```text
y = n1 / (n1 + 2 * n2)
  = 4 / (4 + 2 * 1)
  = 4 / 6
  = 2/3
```

Now calculate `D1`:

```text
D1 = 1 - 2 * y * n2 / n1
   = 1 - 2 * (2/3) * 1 / 4
   = 1 - 1/3
   = 2/3
```

Again, `n3 = 0`, so the implementation uses:

```text
D2 = 1.0
D3+ = 1.5
```

The trigram discount tuple is therefore:

```python
trigram_discounts = (D1, D2, D3_plus)
                     = (2/3, 1.0, 1.5)
```

### 5.5 The final discount table

The model stores one discount triple for each higher n-gram order:

```python
self.discounts_by_order = {
    2: (0.5, 1.0, 1.5),  # bigrams: D1, D2, D3+
    3: (2/3, 1.0, 1.5),   # trigrams: D1, D2, D3+
}
```

There is no key `1` because the unigram level is the Kneser-Ney continuation
distribution; this implementation does not discount unigrams with `D1`, `D2`,
and `D3+`. There is no key `4` because `max_order=3`, so this model does not
build 4-gram counts.

### 5.6 What does discounting do to a count?

Using the bigram discount tuple `(0.5, 1.0, 1.5)`:

```text
observed count 1 -> discounted count = 1 - D1   = 1 - 0.5 = 0.5
observed count 2 -> discounted count = 2 - D2   = 2 - 1.0 = 1.0
observed count 3 -> discounted count = 3 - D3+  = 3 - 1.5 = 1.5
```

For example, the observed bigram `("The", "cat")` has count `2`, so its
direct discounted count is:

```text
c_discounted(The, cat) = 2 - D2
                        = 2 - 1.0
                        = 1.0
```

The model will later divide this discounted count by the history count of
`("The",)` to obtain the direct higher-order contribution. The removed amount
will be assigned to the lower-order continuation model.

### What we know after Step 5

The model now knows:

```text
which discount bucket belongs to each observed count;
how many n-gram types occur once, twice, three times, and four times;
how to estimate D1, D2, and D3+ when the required count classes exist;
which fallback values to use on a tiny corpus lacking some count classes;
that bigrams and trigrams receive separate discount triples.
```

For this toy corpus:

```python
self.discounts_by_order = {
    2: (0.5, 1.0, 1.5),
    3: (2/3, 1.0, 1.5),
}
```

The lambda values calculated later for the toy histories are:

```python
lambda_by_history = {
    ("The",): 0.5,
    ("cat",): 0.5,
    ("sat",): 0.5,
    ("The", "cat"): 2/3,
    ("cat", "sat"): 2/3,
}
```

The next step is to use these discounts to calculate the probability mass
removed after a history: `lambda_for_history(history)`.

## Step 6: Calculate `lambda_for_history(history)`

The function `lambda_for_history(history)` calculates how much probability
mass must be given to the lower-order model after a history has been
discounted.

The central idea is:

```text
lambda(history)
    = total amount removed from observed continuations
      --------------------------------------------------
                 number of times history occurred
```

Why divide by the history count? Because the n-gram counts are eventually
converted into conditional probabilities. If a history occurred many times,
the removed count must be expressed as a fraction of those occurrences.

### 6.1 The code structure

The function begins by identifying the order:

```python
history = tuple(history)
order = len(history) + 1
```

This is why:

```text
history length 1 -> use bigram counts, order 2
history length 2 -> use trigram counts, order 3
```

It then finds the history count:

```python
history_count = self.history_count(history)
```

For a one-word history such as `("The",)`, this asks for the unigram count
of `The`. For a two-word history such as `("The", "cat")`, this asks for the
bigram count of `("The", "cat")`.

The function then visits every observed n-gram of the required order and keeps
only those whose prefix equals the history:

```python
for ngram, count in self.ngram_counts[order].items():
    if ngram[:-1] == history:
        removed_count += self.discount_for_count(count, order)
```

The expression `ngram[:-1]` means “all tokens except the final token.” For
example:

```text
("The", "cat", "sat")[:-1] = ("The", "cat")
("cat", "sat")[:-1]       = ("cat",)
```

### 6.2 Lambda for history `("The",)`

We begin with the one-word history:

```text
history = ("The",)
```

Because the history has length `1`, we use bigrams. The observed bigrams whose
prefix is `("The",)` are:

```text
("The", "cat")
```

Its count is `2`:

```text
c(The, cat) = 2
```

The bigram discount for count `2` is `D2 = 1.0`. Therefore, the removed count
is:

```text
removed_count = D2
              = 1.0
```

The history count is the unigram count of `The`:

```text
history_count = c(The) = 2
```

Therefore:

```text
lambda((The,)) = removed_count / history_count
                = 1.0 / 2
                = 0.5
```

Interpretation:

```text
After observing The, 50% of the probability mass is reserved for the
lower-order continuation model.
```

### 6.3 Lambda for history `("cat",)`

Now consider:

```text
history = ("cat",)
```

The observed bigrams beginning with `cat` are:

```text
("cat", "sat")
("cat", "slept")
```

Both have count `1`, so both use `D1 = 0.5` for bigrams:

```text
removed_count = 0.5 + 0.5
              = 1.0
```

The history count is:

```text
history_count = c(cat) = 2
```

Therefore:

```text
lambda((cat,)) = 1.0 / 2
                = 0.5
```

Here the two observed continuations each lose `0.5` count mass. Together,
they lose `1.0`, which is half of the two occurrences of `cat`.

### 6.4 Lambda for history `("The", "cat")`

Now move one order higher:

```text
history = ("The", "cat")
```

Because the history has length `2`, we use trigram counts. The observed
trigrams with this history are:

```text
("The", "cat", "sat")
("The", "cat", "slept")
```

Both have count `1`. For trigrams, the value of `D1` is `2/3`, so:

```text
removed_count = 2/3 + 2/3
              = 4/3
```

The history count is the bigram count:

```text
history_count = c(The, cat) = 2
```

Therefore:

```text
lambda((The, cat)) = (4/3) / 2
                    = 4/6
                    = 2/3
```

Interpretation:

```text
After the history The cat, two-thirds of the probability mass is reserved
for the lower-order model.
```

The value is larger than the lambda for `("The",)` because the trigram
discount used here is `2/3` for each singleton trigram.

### 6.5 Lambda for history `("cat", "sat")`

Consider a history that appears only once:

```text
history = ("cat", "sat")
```

The only observed trigram continuing this history is:

```text
("cat", "sat", "</s>")
```

Its count is `1`, so the trigram discount is `D1 = 2/3`:

```text
removed_count = 2/3
```

The history count is:

```text
history_count = c(cat, sat) = 1
```

Therefore:

```text
lambda((cat, sat)) = (2/3) / 1
                    = 2/3
```

### 6.6 What if the history was never observed?

The function contains this branch:

```python
if history_count == 0:
    return 1.0
```

For example, suppose we ask about:

```text
history = ("sat", "The")
```

The bigram `("sat", "The")` does not appear in the toy corpus, so:

```text
history_count = 0
lambda((sat, The)) = 1.0
```

This means: there is no reliable higher-order evidence for this history, so
the calculation gives all probability mass to the lower-order model. In the
later recursive calculation, the history itself will be backed off by removing
its oldest token.

### 6.7 Why are unseen continuations not included in lambda?

Suppose the history is `("cat",)`. The observed continuations are only:

```text
sat, slept
```

The function sums discounts only for those observed bigrams:

```text
discount(cat, sat) + discount(cat, slept)
```

It does not loop over every possible vocabulary word and assign a discount to
unseen bigrams. An unseen bigram has count zero, so there is no observed count
from which to subtract a discount. The total removed mass is exactly the mass
that must be redistributed to the lower-order model.

### 6.8 The mass-conservation intuition

For history `("cat",)`, the original observed count mass is:

```text
c(cat, sat) + c(cat, slept)
    = 1 + 1
    = 2
```

After discounting:

```text
(1 - 0.5) + (1 - 0.5)
    = 0.5 + 0.5
    = 1.0
```

The removed count is:

```text
2.0 - 1.0 = 1.0
```

Dividing by `c(cat) = 2` gives:

```text
1.0 / 2 = 0.5
```

So lambda is precisely the probability mass that was removed from this
history's observed continuations.

### What we know after Step 6

For this toy corpus:

```python
lambda_by_history = {
    ("The",): 0.5,
    ("cat",): 0.5,
    ("sat",): 0.5,
    ("The", "cat"): 2/3,
    ("cat", "sat"): 2/3,
}
```

The meaning is:

```text
lambda = fraction of probability mass reserved for the lower-order model
```

The next step is to calculate a complete probability for one word and one
history, first at the bottom continuation level and then recursively upward.

## Step 7: Calculate a probability recursively

We will calculate:

```text
P(sat | The, cat)
```

The history has two words, so this begins as a trigram calculation. If the
trigram evidence is insufficient, the model recursively uses a shorter
history:

```text
P(sat | The, cat)
    -> P(sat | cat)
        -> P(sat)
```

The code implements this through `_probability(word, history, trace)`.

### The complete recursive formula

The complete recursive formula is:

```text
P(sat | The, cat)
=
max(c(The, cat, sat) - D(c(The, cat, sat)), 0)
------------------------------------------------
                    c(The, cat)
+
lambda(The, cat) * P(sat | cat)
```

where:

```text
P(sat | cat)
=
max(c(cat, sat) - D(c(cat, sat)), 0)
-------------------------------------
                c(cat)
+
lambda(cat) * P(sat)
```

and the base case is:

```text
P(sat)
=
N1+(· sat)
----------
N1+(· ·)
```

### 7.1 The general interpolated formula

For a seen history, the code calculates:

```text
P(word | history)
    = direct_contribution
      + lambda(history) * lower_order_probability
```

The direct contribution is:

```text
direct_contribution
    = max(count(history, word) - selected_discount, 0)
      -------------------------------------------------------
                    count(history)
```

The two parts have different meanings:

```text
direct contribution
    -> evidence from the current n-gram order

lambda * lower-order probability
    -> probability mass reserved for shorter-context evidence
```

### Important distinction: `selected_discount` versus `lambda`

`selected_discount` depends on the specific n-gram `(history, word)` because it
is selected from that n-gram's observed count:

```text
count(history, word) = 1  -> use D1
count(history, word) = 2  -> use D2
count(history, word) >= 3 -> use D3+
```

`lambda` depends only on the history. It sums the discounts of **all observed
continuations** after that history and divides by the history count:

```text
selected_discount -> depends on (history, word)
lambda(history)   -> depends only on history
```

### 7.2 First call: begin with `P(sat | The, cat)`

The actual code does **not** begin by calculating `P(sat)`. It begins with
the probability requested by the caller:

```python
model.probability("sat", ("The", "cat"))
```

Inside `_probability()`, the current values are:

```text
word = "sat"
history = ("The", "cat")
order = len(history) + 1 = 3
```

The current trigram is observed:

```text
c(The, cat, sat) = 1
```

The function can immediately calculate the current-order pieces:

```text
selected_discount = D1_trigram = 2/3
history_count = c(The, cat) = 2
direct_contribution = (1 - 2/3) / 2 = 1/6
lambda((The, cat)) = 2/3
```

But the final probability is not ready yet. The formula still needs:

```text
lower_order_probability = P(sat | cat)
```

Therefore, the code recursively calls itself with the oldest history token
removed:

```python
self._probability("sat", ("cat",), trace)
```

At this point, the trigram call is paused. It has remembered its direct
contribution and lambda, but it cannot combine them until the recursive call
returns.

### 7.3 Second call: calculate `P(sat | cat)`

Use the one-word history:

```text
history = ("cat",)
word = "sat"
```

The corresponding bigram is `("cat", "sat")`, whose count is:

```text
c(cat, sat) = 1
```

Because this is a bigram and the count is `1`, select the bigram `D1`:

```text
selected_discount = D1_bigram = 0.5
```

The history count is the unigram count of `cat`:

```text
c(cat) = 2
```

Therefore, the direct bigram contribution is:

```text
direct_contribution = (1 - 0.5) / 2
                    = 0.5 / 2
                    = 0.25
```

From Step 6:

```text
lambda((cat,)) = 0.5
```

The current call still needs:

```text
lower_order_probability = P(sat)
```

So it recursively calls:

```python
self._probability("sat", (), trace)
```

The bigram call is now paused, waiting for the base case.

### 7.4 Third call: reach the base case and calculate `P(sat)`

Now the recursive history is empty:

```text
word = "sat"
history = ()
```

The code reaches:

```python
if not history:
    probability = self.continuation_probability(word)
    return probability
```

For `sat`, the continuation count is `1` and the total number of distinct
bigrams is `6`:

```text
P(sat) = N1+(· sat) / N1+(· ·)
       = 1 / 6
```

This is the bottom of the recursion. The base-case call returns `1/6` to the
paused bigram call.

### 7.5 Return from the base case: finish `P(sat | cat)`

The bigram call now receives:

```text
lower_order_probability = P(sat) = 1/6
```

Combine the two pieces:

```text
P(sat | cat)
    = 0.25 + 0.5 * (1/6)
    = 0.25 + 1/12
    = 3/12 + 1/12
    = 4/12
    = 1/3
```

So:

```text
P(sat | cat) = 1/3
```

The mental picture is:

```text
P(sat | cat)
    = current bigram evidence
      + leftover mass passed to the continuation model
```

The returned value is:

```text
P(sat | cat) = 1/3
```

This value is returned to the paused trigram call.

### 7.6 Return again: finish `P(sat | The, cat)`

The trigram call now receives:

```text
lower_order_probability = P(sat | cat) = 1/3
```

Now combine the trigram contribution with the lower-order contribution:

```text
P(sat | The, cat)
    = 1/6 + (2/3) * (1/3)
    = 1/6 + 2/9
    = 3/18 + 4/18
    = 7/18
    ≈ 0.3888889
```

The final result is:

```text
P(sat | The, cat) = 7/18
```

### 7.7 The complete recursion as a diagram

The code follows this sequence:

```text
P(sat | The, cat)
    trigram direct contribution = 1/6
    lambda(The cat) = 2/3
    lower probability = P(sat | cat)
        bigram direct contribution = 1/4
        lambda(cat) = 1/2
        lower probability = P(sat) = 1/6
            continuation probability = 1/6

    P(sat | cat) = 1/4 + (1/2)(1/6) = 1/3
    P(sat | The, cat) = 1/6 + (2/3)(1/3) = 7/18
```

At each level, the model performs the same two actions:

```text
1. use the current-order n-gram if it was observed;
2. pass the remaining mass to a shorter history.
```

### 7.8 What happens to the trace?

If we call:

```python
model.probability("sat", ("The", "cat"), return_trace=True)
```

the function returns both the probability and a list describing each level.
Conceptually, the trace contains:

```text
order 3: history (The, cat), seen trigram, direct contribution 1/6
order 2: history (cat), seen bigram, direct contribution 1/4
order 1: empty history, continuation probability 1/6
```

The trace exposes intermediate values that would otherwise be hidden inside
the recursive function.

### What we know after Step 7

For the example `P(sat | The, cat)`:

```text
base continuation probability:       P(sat) = 1/6
bigram probability:                  P(sat | cat) = 1/3
trigram probability:                 P(sat | The, cat) = 7/18
```

The next step is to calculate what happens when the trigram itself is unseen,
including the difference between an unseen n-gram with a seen history and an
unseen history.

## Interlude: Verify that probabilities sum to one

A language model must produce a valid probability distribution. For every
fixed history, the probabilities of all possible next words must sum to one:

```text
sum over every vocabulary word w of P(w | history) = 1
```

The code checks this with:

```python
def normalization_total(self, history=()):
    return sum(self.probability(word, history) for word in self.vocabulary)
```

The function loops over the model vocabulary and adds one probability at a
time. In our toy corpus:

```python
self.vocabulary = ("</s>", "The", "cat", "sat", "slept")
```

### 9.1 Normalization at the continuation level

For the empty history, the model uses continuation probabilities:

```text
P(The)   = 1/6
P(cat)   = 1/6
P(sat)   = 1/6
P(slept) = 1/6
P(</s>)  = 2/6
```

Add them:

```text
1/6 + 1/6 + 1/6 + 1/6 + 2/6
    = 6/6
    = 1
```

Therefore:

```text
normalization_total(()) = 1
```

### 9.2 Normalization for history `("The",)`

The observed continuation after `The` is `cat`. Its bigram count is `2`, so
its direct contribution is:

```text
(2 - D2_bigram) / c(The)
    = (2 - 1) / 2
    = 1/2
```

The lambda is `1/2`, so every lower-order probability is multiplied by `1/2`.
The complete distribution is:

```text
word       direct part     lower-order part       total
----------------------------------------------------------
cat        1/2             (1/2)(1/6)             7/12
The        0               (1/2)(1/6)             1/12
sat        0               (1/2)(1/6)             1/12
slept      0               (1/2)(1/6)             1/12
</s>       0               (1/2)(2/6)             1/6
```

Now sum the totals:

```text
7/12 + 1/12 + 1/12 + 1/12 + 1/6
    = 7/12 + 1/12 + 1/12 + 1/12 + 2/12
    = 12/12
    = 1
```

Therefore:

```text
normalization_total((The,)) = 1
```

### 9.3 Normalization for history `("cat",)`

The observed continuations after `cat` are `sat` and `slept`. Each has count
`1`, so each direct contribution is:

```text
(1 - D1_bigram) / c(cat)
    = (1 - 0.5) / 2
    = 1/4
```

The lambda is `1/2`. The probabilities are:

```text
P(sat | cat)   = 1/4 + (1/2)(1/6) = 1/3
P(slept | cat) = 1/4 + (1/2)(1/6) = 1/3
P(The | cat)   = 0   + (1/2)(1/6) = 1/12
P(cat | cat)   = 0   + (1/2)(1/6) = 1/12
P(</s> | cat)  = 0   + (1/2)(2/6) = 1/6
```

Sum them:

```text
1/3 + 1/3 + 1/12 + 1/12 + 1/6
    = 4/12 + 4/12 + 1/12 + 1/12 + 2/12
    = 12/12
    = 1
```

Therefore:

```text
normalization_total((cat,)) = 1
```

### 9.4 Normalization for history `("The", "cat")`

The observed trigram continuations are `sat` and `slept`. For each one:

```text
direct contribution
    = (1 - D1_trigram) / c(The, cat)
    = (1 - 2/3) / 2
    = 1/6
```

The lambda is `2/3`. The lower-order distribution is `P(word | cat)`:

```text
P(sat | cat)   = 1/3
P(slept | cat) = 1/3
P(The | cat)   = 1/12
P(cat | cat)   = 1/12
P(</s> | cat)  = 1/6
```

The final probabilities are:

```text
P(sat | The, cat)
    = 1/6 + (2/3)(1/3)
    = 7/18

P(slept | The, cat)
    = 1/6 + (2/3)(1/3)
    = 7/18

P(The | The, cat)
    = (2/3)(1/12)
    = 1/18

P(cat | The, cat)
    = (2/3)(1/12)
    = 1/18

P(</s> | The, cat)
    = (2/3)(1/6)
    = 1/9
```

Now sum them using denominator `18`:

```text
7/18 + 7/18 + 1/18 + 1/18 + 1/9
    = 7/18 + 7/18 + 1/18 + 1/18 + 2/18
    = 18/18
    = 1
```

Therefore:

```text
normalization_total((The, cat)) = 1
```

### 9.5 Why does the sum equal one mathematically?

For an observed history `h`, the direct contributions over all vocabulary
words sum to:

```text
sum of direct contributions
    = [history_count - removed_count] / history_count
    = 1 - lambda(h)
```

The lower-order probabilities already form a distribution, so:

```text
sum over words of lower_order_probability = 1
```

Multiplying that distribution by lambda gives:

```text
sum of lambda * lower_order_probability = lambda(h)
```

Now add both parts:

```text
sum of final probabilities
    = [1 - lambda(h)] + lambda(h)
    = 1
```

For an unseen history, the code directly returns a lower-order distribution.
That distribution already sums to one, so normalization is preserved during
backoff as well.

### 9.6 What would indicate a bug?

The normalization check is a powerful debugging tool:

```text
total < 1
    -> probability mass was lost

total > 1
    -> probability mass was counted more than once

total = 1
    -> the interpolation and continuation probabilities are consistent
```

In real floating-point computation, the result may be very close to `1`, such
as `0.9999999999999999`. The code therefore uses `math.isclose()` rather than
requiring exact binary equality.

### What we know after the normalization check

The model passes the central probability sanity check:

```python
normalization_total(history) = 1
```

This works because discounting removes exactly `lambda(history)` mass, and the
lower-order model receives exactly that same amount. The next step is to
connect the recursive probability function to the `trace` output printed by
`print_demo()`.

## Step 8: Calculate probabilities for unseen cases

Smoothing is most useful when the exact n-gram was not observed. The code has
two different branches for two different situations:

```text
case 1: history was observed, but the requested n-gram was not observed
        -> keep the history and use lower-order probability

case 2: history itself was never observed
        -> remove the oldest history token and recurse immediately
```

These cases must not be confused. An unseen continuation does not mean that
the history is unseen.

### 8.1 Case 1: unseen trigram, seen history

Calculate:

```text
P(slept | cat, sat)
```

The history is `("cat", "sat")`, and the corresponding trigram is:

```text
("cat", "sat", "slept")
```

This trigram does not occur:

```text
c(cat, sat, slept) = 0
```

But the history does occur once because of:

```text
("cat", "sat", "</s>")
```

Therefore:

```text
c(cat, sat) = 1
```

The code enters the seen-history branch. Because the requested trigram count
is zero:

```python
selected_discount = self.discount_for_count(0, 3)
```

The helper returns zero for a non-positive count:

```text
selected_discount = 0.0
```

The direct contribution is therefore:

```text
direct_contribution = max(0 - 0, 0) / 1
                    = 0
```

The lambda for this history is calculated from the observed continuation
`</s>`:

```text
lambda((cat, sat))
    = D1_trigram / c(cat, sat)
    = (2/3) / 1
    = 2/3
```

The code then recursively asks for the shorter history:

```text
P(slept | sat)
```

### 8.2 Continue the recursion: `P(slept | sat)`

The bigram `("sat", "slept")` is also unseen:

```text
c(sat, slept) = 0
```

However, the history `("sat",)` is observed once because of:

```text
("sat", "</s>")
```

Thus:

```text
c(sat) = 1
```

Again, the direct contribution is zero:

```text
direct_contribution = max(0 - 0, 0) / 1
                    = 0
```

The only observed continuation after `sat` is `</s>`, whose bigram count is
`1`. Therefore:

```text
lambda((sat,))
    = D1_bigram / c(sat)
    = 0.5 / 1
    = 0.5
```

The recursion continues to the continuation probability:

```text
P(slept) = 1 / 6
```

Now finish the bigram call:

```text
P(slept | sat)
    = 0 + 0.5 * (1/6)
    = 1/12
```

Return to the paused trigram call:

```text
P(slept | cat, sat)
    = 0 + (2/3) * (1/12)
    = 2/36
    = 1/18
```

The key lesson is:

```text
unseen trigram with seen history
    -> direct trigram contribution is zero
    -> lambda still passes mass to the lower-order model
```

### 8.3 Case 2: unseen history

Now calculate:

```text
P(sat | sat, The)
```

The history itself is unseen:

```text
history = ("sat", "The")
c(sat, The) = 0
```

The code takes the unseen-history branch before calculating a current-order
contribution:

```python
if history_count == 0:
    lower_history = history[1:]
    lower_probability = self._probability(word, lower_history, trace)
    return lower_probability
```

For this history:

```text
history[1:] = ("The",)
```

So the code immediately changes the problem to:

```text
P(sat | The)
```

Notice the difference:

```text
seen history + unseen n-gram
    -> calculate lambda(history), then recurse

unseen history
    -> skip current-order calculation and recurse with shorter history
```

### 8.4 Finish `P(sat | The)`

The history `("The",)` is observed:

```text
c(The) = 2
```

But the bigram `("The", "sat")` is unseen:

```text
c(The, sat) = 0
```

Thus:

```text
direct_contribution = max(0 - 0, 0) / 2
                    = 0
```

From Step 6:

```text
lambda((The,)) = 0.5
```

The lower-order probability is:

```text
P(sat) = 1/6
```

Therefore:

```text
P(sat | The)
    = 0 + 0.5 * (1/6)
    = 1/12
```

Because the unseen history `("sat", "The")` backed off directly to
`("The",)`, the final answer is also:

```text
P(sat | sat, The) = 1/12
```

### 8.5 Why does an unseen history return lower probability directly?

The model has no observed count for the history itself. It cannot calculate a
meaningful fraction such as:

```text
discounted_count / history_count
```

because the denominator would be zero. Therefore, it removes the oldest
context word and asks the lower-order model to handle the prediction.

This is the recursive backoff path:

```text
P(word | w1, w2)
    if (w1, w2) unseen
    -> P(word | w2)
```

### What we know after Step 8

The model now handles both important unseen situations:

```text
seen history + unseen n-gram
    -> zero direct contribution
    -> interpolate using lambda(history) and lower-order probability

unseen history
    -> skip current-order calculation
    -> remove the oldest history token
    -> recurse immediately
```

For the toy examples:

```text
P(slept | cat, sat) = 1/18
P(sat | sat, The)   = 1/12
```

The normalization check appears above as an interlude. It verifies that the
probability mass remains valid across both seen and unseen cases.

## Interlude: Measure validation perplexity

After the model has been trained, `print_demo()` calls:

```python
perplexity, predicted_tokens = validation_perplexity(
    model, validation_sentences
)
```

This does not change the model. It only asks:

```text
How much probability does the finished model assign to new validation text?
```

### 9.1 Convert one validation row into prediction events

Take this validation row:

```text
The cat sat
```

The function first adds boundaries:

```text
[<s>, The, cat, sat, </s>]
```

The model predicts every token after the first token:

```text
prediction 1: predict "The" using history (<s>,)
prediction 2: predict "cat" using history (<s>, The)
prediction 3: predict "sat" using history (The, cat)
prediction 4: predict "</s>" using history (cat, sat)
```

The first real word is predicted from `<s>`, and the end token is also
predicted. Therefore, this row contributes four predicted tokens.

The code creates each history using the maximum supported context length:

```python
history_start = max(0, position - (model.max_order - 1))
history = tuple(tokens[history_start:position])
```

Since `max_order=3`, the model uses at most two previous tokens:

```text
position 1 -> one-token history: (<s>,)
position 2 -> two-token history: (<s>, The)
position 3 -> two-token history: (The, cat)
position 4 -> two-token history: (cat, sat)
```

### 9.2 Handle validation words unseen during training

The model's vocabulary is built only from training text. A validation row may
contain a word absent from that vocabulary.

The function `evaluation_words()` handles this:

```text
if <unk> is in the training vocabulary:
    replace each unknown validation word with <unk>
else:
    raise an error if validation contains an unknown word
```

This prevents validation data from secretly expanding the training vocabulary.
If we added new validation words to the vocabulary after training, the
evaluation would no longer represent the model trained on the original data.

In the non-raw WikiText-2 configuration, unknown words are represented by
`<unk>`, so the intended path is:

```text
unknown validation word -> <unk> -> model probability for <unk>
```

### 9.3 Calculate the four probabilities for the toy row

For this tutorial's toy corpus, the four prediction probabilities are:

```text
P(The | <s>)         = 7/12
P(cat | <s>, The)    = 19/24
P(sat | The, cat)    = 7/18
P(</s> | cat, sat)   = 7/9
```

The first probability is obtained from the observed bigram `(<s>, The)`:

```text
P(The | <s>)
    = (2 - 1) / 2 + (1/2)(1/6)
    = 1/2 + 1/12
    = 7/12
```

For the second probability, the trigram `(<s>, The, cat)` has count `2`:

```text
P(cat | <s>, The)
    = (2 - 1) / 2 + (1/2) P(cat | The)
```

The lower-order probability is:

```text
P(cat | The) = 7/12
```

Therefore:

```text
P(cat | <s>, The)
    = 1/2 + (1/2)(7/12)
    = 1/2 + 7/24
    = 19/24
```

The last two probabilities were calculated in earlier steps:

```text
P(sat | The, cat) = 7/18
P(</s> | cat, sat) = 7/9
```

### 9.4 Multiply probabilities to obtain row probability

The probability assigned to the complete row is the product of the
conditional probabilities:

```text
P(The, cat, sat, </s>)
    = P(The | <s>)
      * P(cat | <s>, The)
      * P(sat | The, cat)
      * P(</s> | cat, sat)
```

Substitute the values:

```text
    = (7/12) * (19/24) * (7/18) * (7/9)
    = 6517 / 46656
    ≈ 0.1396819
```

The implementation does not multiply these values directly. It adds their
logarithms:

```python
total_log_probability += math.log(probability)
```

This is mathematically equivalent because:

```text
log(a * b * c) = log(a) + log(b) + log(c)
```

Logarithms are safer for a large corpus because multiplying thousands of
small probabilities directly can underflow to zero in floating-point
arithmetic.

### 9.5 Calculate perplexity

For `N` predicted tokens, the word-level perplexity is:

```text
perplexity = exp(- total_log_probability / N)
```

For the toy row:

```text
N = 4
```

Therefore:

```text
perplexity
    = (1 / sentence_probability)^(1/4)
    = (46656 / 6517)^(1/4)
    ≈ 1.6357
```

The fourth root appears because there are four predicted tokens. Perplexity is
the reciprocal geometric mean probability assigned to the predicted tokens.

### 9.6 Intuition for perplexity

Perplexity can be understood as the model's average uncertainty expressed as
an equivalent number of choices:

```text
lower perplexity -> the model assigned higher probability to the real text
high perplexity  -> the model spread probability away from the real text
```

Perplexity is meaningful only when the evaluation protocol is fixed. We must
compare models on the same validation or test data, with the same token-counting
rules and the same treatment of `<unk>` and `</s>`.

### 9.7 Validation versus test perplexity

This implementation currently prints validation perplexity because it is still
being used as a learning-time diagnostic. The test split is loaded but is not
yet evaluated by `print_demo()`.

The eventual final-evaluation workflow should be:

```text
train split
    -> build counts and MKN parameters

validation split
    -> inspect or compare choices during development

test split
    -> calculate final perplexity after choices are frozen
```

### What we know after the perplexity calculation

The complete evaluation path is now clear:

```text
validation row
  -> add <s> and </s>
  -> predict each token after <s>
  -> call model.probability(...) for each prediction
  -> add log probabilities
  -> divide by number of predicted tokens
  -> exponentiate the negative average
  -> report perplexity
```

The next step is to connect this tutorial flow to the exact output printed by
`print_demo()`, including the selected counts, discount table, recursive
traces, perplexity, and normalization checks.
