# Step-by-Step Implementation of Modified Kneser-Ney

This file follows the code in `mkn_simulator.py` one small step at a time.

We will use the same toy corpus throughout. Every count and probability will be calculated by hand before we connect it to the corresponding code.

## Broad Step 1: Enter the program and load the raw sentences

This broad step combines four small code actions:

1. Start at the program entry point.
2. Choose the toy-corpus file.
3. Read the file into Python.
4. Pass the raw sentences to the model.

### 1.1 Start at the entry point

The program starts here:

```python
if __name__ == "__main__":
    main()
```

This means:

1. Python checks whether this file is being run directly.
2. If it is, Python calls the `main()` function.
3. We therefore begin our learning journey inside `main()`.

### 1.2 Choose the toy corpus file

The first meaningful line inside `main()` is:

```python
corpus_path = Path(__file__).with_name("toy_corpus.txt")
```

Read this as:

> Find the file named `toy_corpus.txt` in the same folder as `mkn_simulator.py`.

Here, `corpus_path` is not the corpus itself. It is the location of the corpus file.

The toy corpus contains 12 sentences:

```text
the cat sat on the mat
the cat sat on the rug
the cat sat near the mat
the dog sat on the mat
the dog sat on the rug
the dog sat near the mat
a cat sat on the mat
a cat sat on the rug
a dog sat on the mat
a dog sat on the rug
birds fly over the mat
birds fly over the rug
```

For now, each line is treated as one complete sentence. The words are separated by spaces.

### 1.3 Read the corpus into Python

Next, `main()` calls:

```python
sentences = load_corpus(corpus_path)
```

The function `load_corpus` does three simple things:

```python
path = Path(path)
sentences = [
    line.strip()
    for line in path.read_text().splitlines()
    if line.strip()
]
```

In plain language:

1. Open the file.
2. Split the file into separate lines.
3. Remove extra whitespace from each line.
4. Ignore empty lines.
5. Store the remaining lines in a Python list.

After this function finishes, `sentences` is conceptually:

```python
[
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the cat sat near the mat",
    "the dog sat on the mat",
    "the dog sat on the rug",
    "the dog sat near the mat",
    "a cat sat on the mat",
    "a cat sat on the rug",
    "a dog sat on the mat",
    "a dog sat on the rug",
    "birds fly over the mat",
    "birds fly over the rug",
]
```

Important clarification: these are still ordinary strings. We have not yet added `<s>` or `</s>`, and we have not yet created unigrams, bigrams, or trigrams.

### 1.4 Ask the model to learn from these sentences

The next line is:

```python
model = MKNModel.from_sentences(sentences, max_order=3)
```

This asks for a model that can use up to 3-word sequences:

```text
order 1: unigram   -> one word
order 2: bigram    -> two words
order 3: trigram   -> three words
```

The name `from_sentences` is a class method. It is a convenient entry point that internally creates the object by calling the constructor:

```python
MKNModel(sentences, max_order=3)
```

The constructor then begins building the information needed by MKN. In the next step, we will examine the very first transformation it performs:

```python
self.tokenized_sentences = [
    tokenize_sentence(sentence)
    for sentence in sentences
]
```

That transformation adds explicit sentence-boundary tokens. Only after we understand those boundaries will we count n-grams.

### What we know after Broad Step 1

At the end of this first step:

```text
Input file       -> toy_corpus.txt
Number of lines  -> 12 sentences
Maximum order    -> 3
MKN probabilities calculated -> no
N-gram counts built           -> not yet
Discounts D1, D2, D3+ known   -> not yet
Lambda values known           -> not yet
```

The only goal so far was to get the 12 raw sentences into the model. Next we will add `<s>` and `</s>` to one sentence at a time.

## Step 2: Add boundary tokens to the first sentence

The constructor now runs:

```python
self.tokenized_sentences = [
    tokenize_sentence(sentence)
    for sentence in sentences
]
```

Let us temporarily look at only the first sentence:

```text
the cat sat on the mat
```

The function receives this ordinary string:

```python
sentence = "the cat sat on the mat"
```

Inside `tokenize_sentence`, the first operation is:

```python
words = sentence.split()
```

`split()` separates the string wherever there is whitespace. Therefore:

```python
"the cat sat on the mat".split()
```

produces:

```python
["the", "cat", "sat", "on", "the", "mat"]
```

At this moment, `words` contains six ordinary word tokens.

The function then returns:

```python
return [START_TOKEN, *words, END_TOKEN]
```

The constants are defined near the top of the file:

```python
START_TOKEN = "<s>"
END_TOKEN = "</s>"
```

The `*words` notation means:

> Put every item from the `words` list into this new list.

So the returned result is:

```python
[
    "<s>",
    "the",
    "cat",
    "sat",
    "on",
    "the",
    "mat",
    "</s>",
]
```

We can write the same result in one line:

```text
<s> the cat sat on the mat </s>
```

### Why do we add `<s>` and `</s>`?

The model should learn that `the` can begin a sentence and that `</s>` can follow `mat` at the end of this sentence.

Without the boundary tokens, the model would only see:

```text
the cat sat on the mat
```

It would count internal transitions such as:

```text
the -> cat
cat -> sat
sat -> on
on -> the
the -> mat
```

But it would not count:

```text
<s> -> the
mat -> </s>
```

Those two transitions matter because a language model predicts words in sequence, including the first word and the end of the sentence.

For example, after adding boundaries, the bigrams from this one sentence are:

```text
(<s>, the)
(the, cat)
(cat, sat)
(sat, on)
(on, the)
(the, mat)
(mat, </s>)
```

There are 8 tokens and therefore 7 adjacent bigrams. The general rule is:

```text
number of bigrams = number of tokens - 1
```

For this sentence:

```text
7 = 8 - 1
```

We are only identifying the possible bigrams here. We have not yet combined this sentence with the other 11 sentences, and we have not yet calculated their counts.

### What the list comprehension does for all 12 sentences

The list comprehension applies the same transformation to every raw sentence:

```python
self.tokenized_sentences = [
    tokenize_sentence(sentence)
    for sentence in sentences
]
```

Conceptually, it performs this sequence:

```python
tokenized_sentence_1 = tokenize_sentence(sentences[0])
tokenized_sentence_2 = tokenize_sentence(sentences[1])
tokenized_sentence_3 = tokenize_sentence(sentences[2])
# ... and so on until sentence 12
```

For example, the first two results are:

```python
[
    ["<s>", "the", "cat", "sat", "on", "the", "mat", "</s>"],
    ["<s>", "the", "cat", "sat", "on", "the", "rug", "</s>"],
]
```

### What we know after Step 2

```text
Raw first sentence       -> "the cat sat on the mat"
Tokenized first sentence -> ["<s>", "the", "cat", "sat", "on", "the", "mat", "</s>"]
Number of tokens         -> 8
Possible bigrams         -> 7
N-gram counts built      -> not yet
MKN probabilities        -> not yet
```

The next step will apply the same boundary-token transformation to all 12 sentences and verify the resulting token positions.

## Step 3: Tokenize all 12 sentences

The list comprehension applies `tokenize_sentence()` once to each of the 12 raw strings. The first 10 sentences contain 6 words, while the last 2 contain 5 words.

Therefore, the first 10 tokenized sentences contain:

```text
1 start token + 6 words + 1 end token = 8 tokens
```

The complete tokenized corpus is therefore:

```python
[
    ["<s>", "the", "cat", "sat", "on", "the", "mat", "</s>"],
    ["<s>", "the", "cat", "sat", "on", "the", "rug", "</s>"],
    ["<s>", "the", "cat", "sat", "near", "the", "mat", "</s>"],
    ["<s>", "the", "dog", "sat", "on", "the", "mat", "</s>"],
    ["<s>", "the", "dog", "sat", "on", "the", "rug", "</s>"],
    ["<s>", "the", "dog", "sat", "near", "the", "mat", "</s>"],
    ["<s>", "a", "cat", "sat", "on", "the", "mat", "</s>"],
    ["<s>", "a", "cat", "sat", "on", "the", "rug", "</s>"],
    ["<s>", "a", "dog", "sat", "on", "the", "mat", "</s>"],
    ["<s>", "a", "dog", "sat", "on", "the", "rug", "</s>"],
    ["<s>", "birds", "fly", "over", "the", "mat", "</s>"],
    ["<s>", "birds", "fly", "over", "the", "rug", "</s>"],
]
```

The first 10 sentences have 6 words. The last 2 sentences have 5 words:

```text
birds fly over the mat
```

Therefore, the token counts are:

```text
First 10 sentences: 10 × 8 = 80 tokens
Last 2 sentences:    2 × 7 = 14 tokens
Total:              80 + 14 = 94 token positions
```

This is a count of token positions, including repeated tokens. It is not yet the size of the vocabulary. For example, the word `the` appears in many positions, but it is still only one distinct vocabulary item.

The number of adjacent bigram positions can also be calculated now:

```text
First 10 sentences: 10 × (8 - 1) = 70 bigram positions
Last 2 sentences:    2 × (7 - 1) = 12 bigram positions
Total:              70 + 12 = 82 bigram positions
```

Likewise, the number of adjacent trigram positions is:

```text
First 10 sentences: 10 × (8 - 2) = 60 trigram positions
Last 2 sentences:    2 × (7 - 2) = 10 trigram positions
Total:              60 + 10 = 70 trigram positions
```

These are positions before merging identical n-grams. For example, the bigram `("the", "cat")` appears in multiple positions, but later it will be stored once with a count showing how many times it appeared.

### What we know after Step 3

```text
Tokenized sentences       -> 12
Total token positions     -> 94
Total bigram positions    -> 82
Total trigram positions   -> 70
Distinct vocabulary size  -> not calculated yet
N-gram frequency counts   -> not calculated yet
MKN probabilities         -> not calculated yet
```

The next step will take one tokenized sentence and manually enumerate its unigrams, bigrams, and trigrams before the code processes all sentences.

## Step 4: Generate n-grams from one sentence

We will use only the first tokenized sentence:

```python
tokens = ["<s>", "the", "cat", "sat", "on", "the", "mat", "</s>"]
```

Give each token its position:

```text
position:  0      1      2      3      4     5      6       7
token:    <s>    the    cat    sat    on    the    mat     </s>
```

An n-gram is a consecutive window of `n` tokens. The window slides from left to right one position at a time.

### 4.1 Unigrams: order 1

For order 1, the window contains one token:

```text
(<s>)
(the)
(cat)
(sat)
(on)
(the)
(mat)
(</s>)
```

There are 8 unigram positions because the sentence has 8 tokens:

```text
8 - 1 + 1 = 8
```

The word `the` occurs twice. Therefore, if we count this sentence so far, the counts are:

```python
{
    ("<s>",): 1,
    ("the",): 2,
    ("cat",): 1,
    ("sat",): 1,
    ("on",): 1,
    ("mat",): 1,
    ("</s>",): 1,
}
```

There are 8 unigram positions but only 7 distinct unigram types, because the two occurrences of `the` are merged into one dictionary key with count 2.

### 4.2 Bigrams: order 2

For order 2, the window contains two neighboring tokens:

```text
positions 0-1: (<s>, the)
positions 1-2: (the, cat)
positions 2-3: (cat, sat)
positions 3-4: (sat, on)
positions 4-5: (on, the)
positions 5-6: (the, mat)
positions 6-7: (mat, </s>)
```

There are 7 bigram positions:

```text
8 - 2 + 1 = 7
```

In this particular sentence, all 7 bigrams are different, so the bigram dictionary for this sentence has 7 keys, each with count 1.

### 4.3 Trigrams: order 3

For order 3, the window contains three neighboring tokens:

```text
positions 0-2: (<s>, the, cat)
positions 1-3: (the, cat, sat)
positions 2-4: (cat, sat, on)
positions 3-5: (sat, on, the)
positions 4-6: (on, the, mat)
positions 5-7: (the, mat, </s>)
```

There are 6 trigram positions:

```text
8 - 3 + 1 = 6
```

Again, all 6 trigrams are different in this one sentence, so each has count 1 at this stage.

### 4.4 The general counting formula

If a tokenized sentence has `L` tokens, the number of n-gram positions of order `n` is:

```text
L - n + 1
```

For our first sentence, `L = 8`:

```text
order 1: 8 - 1 + 1 = 8 positions
order 2: 8 - 2 + 1 = 7 positions
order 3: 8 - 3 + 1 = 6 positions
```

This formula counts positions, not distinct types. Distinct types are obtained only after identical tuples are placed under the same dictionary key.

### What we know after Step 4

For the first sentence only:

```text
unigram positions   -> 8
bigram positions    -> 7
trigram positions   -> 6
distinct unigrams   -> 7
distinct bigrams    -> 7
distinct trigrams   -> 6
```

The next step will connect these hand-generated windows to the nested loops inside `_build_ngram_counts()`.

## Step 5: See how `_build_ngram_counts()` creates the windows

The relevant code begins like this:

```python
def _build_ngram_counts(self):
    counts_by_order = {
        order: {} for order in range(1, self.max_order + 1)
    }
```

Because this model was created with `max_order=3`, the expression
`range(1, self.max_order + 1)` becomes:

```python
range(1, 4)
```

This produces the orders:

```text
1, 2, 3
```

Therefore, the empty starting structure is:

```python
counts_by_order = {
    1: {},  # unigram counts will go here
    2: {},  # bigram counts will go here
    3: {},  # trigram counts will go here
}
```

The number before each dictionary is the n-gram order. The dictionary itself will map an n-gram tuple to the number of times it has appeared.

### 5.1 The outer loops

The function then loops through sentences and orders:

```python
for tokens in self.tokenized_sentences:
    for order in counts_by_order:
        for start in range(len(tokens) - order + 1):
            ngram = tuple(tokens[start : start + order])
            counts = counts_by_order[order]
            counts[ngram] = counts.get(ngram, 0) + 1
```

For now, pretend `self.tokenized_sentences` contains only our first sentence:

```python
tokens = ["<s>", "the", "cat", "sat", "on", "the", "mat", "</s>"]
```

The outer loop selects this sentence. The next loop selects `order=1`, then `order=2`, and finally `order=3`.

### 5.2 How the `start` positions are calculated

For our sentence, `len(tokens)` is 8.

When `order=1`:

```python
range(len(tokens) - order + 1)
range(8 - 1 + 1)
range(8)
```

So `start` takes the values:

```text
0, 1, 2, 3, 4, 5, 6, 7
```

That gives 8 unigram windows.

When `order=2`:

```python
range(8 - 2 + 1)
range(7)
```

So `start` takes the values `0` through `6`, giving 7 bigram windows.

When `order=3`:

```python
range(8 - 3 + 1)
range(6)
```

So `start` takes the values `0` through `5`, giving 6 trigram windows.

This is the same arithmetic we performed by hand in Step 4.

### 5.3 How one window becomes an n-gram

Consider the first bigram window. At this moment:

```python
order = 2
start = 0
```

The slice is:

```python
tokens[start : start + order]
tokens[0 : 0 + 2]
tokens[0 : 2]
```

That selects the tokens at positions 0 and 1:

```python
["<s>", "the"]
```

Then `tuple(...)` converts the list into a tuple:

```python
ngram = ("<s>", "the")
```

The tuple is used as a dictionary key because it identifies the exact n-gram.

### 5.4 How one count is added

At the beginning, the bigram dictionary is empty:

```python
counts_by_order[2] = {}
```

The code sets:

```python
counts = counts_by_order[order]
```

Since `order=2`, this means:

```python
counts = counts_by_order[2]
```

Now the update is:

```python
counts[ngram] = counts.get(ngram, 0) + 1
```

Substitute the actual n-gram:

```python
counts[("<s>", "the")] = counts.get(("<s>", "the"), 0) + 1
```

Because the dictionary does not contain this key yet:

```python
counts.get(("<s>", "the"), 0) = 0
```

Therefore:

```python
counts[("<s>", "the")] = 0 + 1
counts[("<s>", "the")] = 1
```

The dictionary is now:

```python
counts_by_order[2] = {
    ("<s>", "the"): 1,
}
```

The next bigram window has `start=1`:

```python
tokens[1 : 1 + 2]
tokens[1 : 3]
```

This produces:

```python
("the", "cat")
```

Since this key is also new, it receives count 1. Repeating this process for `start=0` through `start=6` creates all 7 bigram entries for this sentence.

### 5.5 Why `.get(ngram, 0) + 1` handles repetition

Later, when the window reaches the second occurrence of a repeated n-gram, the key already exists. For example, the unigram `("the",)` occurs twice in the first sentence.

First occurrence:

```python
counts[("the",)] = counts.get(("the",), 0) + 1
counts[("the",)] = 0 + 1
counts[("the",)] = 1
```

Second occurrence:

```python
counts[("the",)] = counts.get(("the",), 0) + 1
counts[("the",)] = 1 + 1
counts[("the",)] = 2
```

The dictionary does not store two separate `("the",)` keys. It stores one key with value 2.

### What we know after Step 5

For one tokenized sentence, the loops create:

```text
order 1 -> 8 unigram positions
order 2 -> 7 bigram positions
order 3 -> 6 trigram positions
```

They place the results into separate dictionaries:

```python
counts_by_order[1]  # unigram counts
counts_by_order[2]  # bigram counts
counts_by_order[3]  # trigram counts
```

The next step will run this same process over the remaining 11 sentences and combine repeated n-grams by increasing their dictionary counts.

## Step 6: Process the second sentence and merge counts

The second raw sentence is:

```text
the cat sat on the rug
```

After tokenization:

```python
tokens = ["<s>", "the", "cat", "sat", "on", "the", "rug", "</s>"]
```

The loops process this sentence in exactly the same way as the first one. The important difference is that some n-grams have already appeared, while others are new.

### 6.1 Merging unigram counts

After the first sentence, the relevant unigram counts were:

```python
{
    ("<s>",): 1,
    ("the",): 2,
    ("cat",): 1,
    ("sat",): 1,
    ("on",): 1,
    ("mat",): 1,
    ("</s>",): 1,
}
```

The second sentence contributes these unigram occurrences:

```text
<s>, the, cat, sat, on, the, rug, </s>
```

Apply the update one item at a time:

```text
<s>: 1 -> 2
the: 2 -> 3
cat: 1 -> 2
sat: 1 -> 2
on: 1 -> 2
the: 3 -> 4
rug: new -> 1
</s>: 1 -> 2
```

The second occurrence of `the` is important. It appears twice inside the second sentence, so its count increases twice.

The unigram dictionary after two sentences is:

```python
{
    ("<s>",): 2,
    ("the",): 4,
    ("cat",): 2,
    ("sat",): 2,
    ("on",): 2,
    ("mat",): 1,
    ("rug",): 1,
    ("</s>",): 2,
}
```

### 6.2 Merging bigram counts

The first sentence created these bigrams:

```text
(<s>, the)
(the, cat)
(cat, sat)
(sat, on)
(on, the)
(the, mat)
(mat, </s>)
```

The second sentence creates:

```text
(<s>, the)
(the, cat)
(cat, sat)
(sat, on)
(on, the)
(the, rug)
(rug, </s>)
```

The first five bigrams are repeated, so their counts increase from 1 to 2:

```python
("<s>", "the"): 1 -> 2
("the", "cat"): 1 -> 2
("cat", "sat"): 1 -> 2
("sat", "on"): 1 -> 2
("on", "the"): 1 -> 2
```

The final two bigrams are new:

```python
("the", "rug"): 0 -> 1
("rug", "</s>"): 0 -> 1
```

The bigrams ending with `mat` remain unchanged because the second sentence ends with `rug`, not `mat`:

```python
("the", "mat"): 1
("mat", "</s>"): 1
```

### 6.3 Merging trigram counts

The first sentence produced:

```text
(<s>, the, cat)
(the, cat, sat)
(cat, sat, on)
(sat, on, the)
(on, the, mat)
(the, mat, </s>)
```

The second sentence produces:

```text
(<s>, the, cat)
(the, cat, sat)
(cat, sat, on)
(sat, on, the)
(on, the, rug)
(the, rug, </s>)
```

The first four trigrams repeat, so each count becomes 2:

```text
(<s>, the, cat): 1 -> 2
(the, cat, sat): 1 -> 2
(cat, sat, on): 1 -> 2
(sat, on, the): 1 -> 2
```

The last two trigrams are new:

```text
(on, the, rug): 0 -> 1
(the, rug, </s>): 0 -> 1
```

The trigrams involving `mat` remain at count 1.

### 6.4 What the code is doing during a repeated update

Take the repeated bigram `("the", "cat")` as an example. When the second sentence reaches it, the code effectively performs:

```python
ngram = ("the", "cat")
counts = counts_by_order[2]
counts[ngram] = counts.get(ngram, 0) + 1
```

At this point:

```python
counts.get(("the", "cat"), 0) = 1
```

Therefore:

```python
counts[("the", "cat")] = 1 + 1
counts[("the", "cat")] = 2
```

For the new bigram `("the", "rug")`, the same code behaves differently:

```python
counts.get(("the", "rug"), 0) = 0
counts[("the", "rug")] = 0 + 1
counts[("the", "rug")] = 1
```

The code therefore needs no separate “is this new?” branch. `.get(..., 0)` handles both cases.

### Final value of `counts_by_order` after all 12 sentences

After the loop has processed all 12 sentences, the final `counts_by_order` dictionary is:

```python
self.ngram_counts = counts_by_order = {
    1: {
        ("<s>",): 12,
        ("the",): 18,
        ("cat",): 5,
        ("sat",): 10,
        ("on",): 8,
        ("mat",): 7,
        ("</s>",): 12,
        ("rug",): 5,
        ("near",): 2,
        ("dog",): 5,
        ("a",): 4,
        ("birds",): 2,
        ("fly",): 2,
        ("over",): 2,
    },
    2: {
        ("<s>", "the"): 6,
        ("the", "cat"): 3,
        ("cat", "sat"): 5,
        ("sat", "on"): 8,
        ("on", "the"): 8,
        ("the", "mat"): 7,
        ("mat", "</s>"): 7,
        ("the", "rug"): 5,
        ("rug", "</s>"): 5,
        ("sat", "near"): 2,
        ("near", "the"): 2,
        ("the", "dog"): 3,
        ("dog", "sat"): 5,
        ("<s>", "a"): 4,
        ("a", "cat"): 2,
        ("a", "dog"): 2,
        ("<s>", "birds"): 2,
        ("birds", "fly"): 2,
        ("fly", "over"): 2,
        ("over", "the"): 2,
    },
    3: {
        ("<s>", "the", "cat"): 3,
        ("the", "cat", "sat"): 3,
        ("cat", "sat", "on"): 4,
        ("sat", "on", "the"): 8,
        ("on", "the", "mat"): 4,
        ("the", "mat", "</s>"): 7,
        ("on", "the", "rug"): 4,
        ("the", "rug", "</s>"): 5,
        ("cat", "sat", "near"): 1,
        ("sat", "near", "the"): 2,
        ("near", "the", "mat"): 2,
        ("<s>", "the", "dog"): 3,
        ("the", "dog", "sat"): 3,
        ("dog", "sat", "on"): 4,
        ("dog", "sat", "near"): 1,
        ("<s>", "a", "cat"): 2,
        ("a", "cat", "sat"): 2,
        ("<s>", "a", "dog"): 2,
        ("a", "dog", "sat"): 2,
        ("<s>", "birds", "fly"): 2,
        ("birds", "fly", "over"): 2,
        ("fly", "over", "the"): 2,
        ("over", "the", "mat"): 1,
        ("over", "the", "rug"): 1,
    },
}
```

These are the actual counts used by the model for orders 1, 2, and 3. The dictionary contains one key for each distinct n-gram type, and the value tells us how many times that n-gram occurred in the full corpus.

### What we know after Step 6

After processing all 12 sentences:

```text
Total token positions     -> 94
Total bigram positions    -> 82
Total trigram positions   -> 70
Distinct unigram types    -> 14
Distinct bigram types     -> 20
Distinct trigram types    -> 24
```

The position totals come from adding the n-gram windows across all sentences. The distinct-type totals are the number of keys in `counts_by_order[1]`, `counts_by_order[2]`, and `counts_by_order[3]`.

We will now use this complete dictionary for the next part of the model-building process. The consolidated `# Final MKN model values` section will be created only after all later values have been finalized.

## Step 7: Build the vocabulary

After tokenization, the constructor calls:

```python
self.vocabulary = self._build_vocabulary()
```

The vocabulary is the collection of distinct tokens that the model is allowed to predict.

The function starts with an empty set:

```python
vocabulary = set()
```

A set stores each item only once. This is important because the word `the` appears many times in the corpus, but it should appear only once in the vocabulary.

The function then visits every tokenized sentence:

```python
for tokens in self.tokenized_sentences:
    vocabulary.update(tokens)
```

For example, after processing the first sentence, the set contains:

```python
{
    "<s>", "the", "cat", "sat", "on", "mat", "</s>"
}
```

When the remaining sentences are processed, new tokens such as `rug`, `near`, `dog`, `a`, `birds`, `fly`, and `over` are added. Repeated tokens do not create duplicate entries.

From the complete corpus, the set before removing `<s>` is:

```python
{
    "<s>", "the", "cat", "sat", "on", "mat", "</s>",
    "rug", "near", "dog", "a", "birds", "fly", "over"
}
```

There are 14 distinct tokens in this set. This agrees with the 14 distinct unigram keys in `counts_by_order[1]`.

### 7.1 Why remove `<s>`?

The function executes:

```python
vocabulary.discard(START_TOKEN)
```

Since `START_TOKEN` is `"<s>"`, this removes `<s>` from the set.

The reason is that `<s>` is a context marker, not a word that the model predicts. At the beginning of a sentence, the model uses `<s>` as history and predicts a real word such as `the`, `a`, or `birds`.

So the model uses this transition:

```text
<s> -> the
```

but it does not need to calculate a probability for predicting `<s>`.

### 7.2 Why keep `</s>`?

The end marker is not removed. Therefore, `</s>` remains in the vocabulary.

This is intentional because the model should be able to predict the end of a sentence:

```text
mat -> </s>
```

That probability tells the model that a sentence may finish after `mat`.

### 7.3 Sort the vocabulary and convert it to a tuple

The final line is:

```python
return tuple(sorted(vocabulary))
```

This performs two operations:

1. `sorted(vocabulary)` puts the tokens in a predictable order.
2. `tuple(...)` stores the sorted result as an immutable sequence.

The final vocabulary is:

```python
self.vocabulary = (
    "</s>",
    "a",
    "birds",
    "cat",
    "dog",
    "fly",
    "mat",
    "near",
    "on",
    "over",
    "rug",
    "sat",
    "the",
)
```

There are now 13 prediction candidates. The original set had 14 distinct tokens, but one of them—`<s>`—was removed.

### What we know after Step 7

```text
Distinct tokens before removing <s> -> 14
Removed context marker              -> <s>
Prediction vocabulary size          -> 13
End marker </s> included             -> yes
Vocabulary type                     -> sorted tuple
```

The next step will build the predecessor sets used by Kneser-Ney continuation probabilities.

## Step 8: Build predecessor sets

The constructor next calls:

```python
self.predecessors_by_word = self._build_predecessor_sets()
```

**For each word, this dictionary stores the set of distinct tokens that appeared immediately before it in the training bigrams.**

For example, from these bigrams:

```text
(<s>, the)
(on, the)
(near, the)
(over, the)
```

we collect:

```python
predecessors_by_word["the"] = {"<s>", "on", "near", "over"}
```

The word `the` occurred many more than four times in the corpus, but its predecessor set contains only four different predecessors.

### 8.1 The code that builds the dictionary

The function is:

```python
predecessors = {}

for bigram in self.ngram_counts[2]:
    previous_word, word = bigram

    if word not in predecessors:
        predecessors[word] = set()

    predecessors[word].add(previous_word)

return predecessors
```

The loop visits the keys of `self.ngram_counts[2]`. Therefore, it visits each distinct bigram type once, rather than visiting every occurrence separately.

For one bigram:

```python
bigram = ("on", "the")
previous_word, word = bigram
```

Python assigns:

```python
previous_word = "on"
word = "the"
```

If `"the"` has not appeared as a dictionary key yet, the code creates an empty set:

```python
predecessors["the"] = set()
```

Then it adds `"on"`:

```python
predecessors["the"].add("on")
```

The result at this moment is:

```python
{
    "the": {"on"}
}
```

When another distinct bigram such as `("near", "the")` is processed, `"the"` already exists, so the code does not create a new set. It simply adds `"near"` to the existing set:

```python
{
    "the": {"on", "near"}
}
```

If the same bigram occurred many times, it would still contribute only one predecessor because a set does not store duplicates.

### 8.2 Complete predecessor dictionary for the toy corpus

Using all 20 distinct bigram types, the final dictionary is:

```python
self.predecessors_by_word = predecessors_by_word = {
    "the": {"<s>", "on", "near", "over"},
    "cat": {"the", "a"},
    "sat": {"cat", "dog"},
    "on": {"sat"},
    "mat": {"the"},
    "</s>": {"mat", "rug"},
    "rug": {"the"},
    "near": {"sat"},
    "dog": {"the", "a"},
    "a": {"<s>"},
    "birds": {"<s>"},
    "fly": {"birds"},
    "over": {"fly"},
}
```

The order in which set elements are displayed may differ in Python. The important information is which predecessors belong to each set, not their display order.

### 8.3 Examples of continuation counts

The function `continuation_count(word)` returns the size of one predecessor set:

```python
continuation_count("the")
```

The set is:

```python
{"<s>", "on", "near", "over"}
```

Therefore:

```text
continuation_count("the") = 4
```

For `sat`:

```python
predecessors_by_word["sat"] = {"cat", "dog"}
```

Therefore:

```text
continuation_count("sat") = 2
```

For `fly`:

```python
predecessors_by_word["fly"] = {"birds"}
```

Therefore:

```text
continuation_count("fly") = 1
```

This is the central Kneser-Ney idea: a word receives continuation evidence from the number of different contexts in which it appears, not simply from its raw unigram frequency.

### What we know after Step 8

```text
Distinct bigram types                  -> 20
Predecessor sets built                  -> yes
Different predecessors of "the"        -> 4
Different predecessors of "sat"        -> 2
Different predecessors of "fly"        -> 1
```

The next step will calculate the continuation denominator and then use these predecessor sets to calculate continuation probabilities.

## Step 9: Calculate continuation probabilities

The constructor stores the continuation denominator with:

```python
self.continuation_denominator = len(self.ngram_counts[2])
```

The expression `self.ngram_counts[2]` is the dictionary of distinct bigram types. We found that it contains 20 keys, so:

```text
continuation_denominator = 20
```

This denominator is not the total number of bigram positions, which was 82. Kneser-Ney uses the number of distinct bigram types:

```text
20 distinct bigram types
```

### 9.1 The continuation-probability formula

The method is:

```python
def continuation_probability(self, word):
    return self.continuation_count(word) / self.continuation_denominator
```

In words:

```text
continuation probability of a word
= number of different predecessors of that word
  / number of distinct bigram types in the corpus
```

Using mathematical notation, this is often written as:

```text
P_KN(word) = N1+(· word) / N1+(· ·)
```

Here:

```text
N1+(· word) = number of distinct bigrams ending in word
N1+(· ·)    = number of distinct bigrams overall
```

The dot means “any token.” For example, `N1+(· the)` means “how many different tokens can appear before `the`?”

### 9.2 Calculate one continuation probability completely

For the word `the`:

```python
predecessors_by_word["the"] = {"<s>", "on", "near", "over"}
```

There are four distinct predecessors, so:

```text
continuation_count("the") = 4
continuation_denominator   = 20
```

Therefore:

```text
P_KN(the) = 4 / 20
          = 0.2
```

Notice the contrast:

```text
raw unigram count of the        = 18
distinct predecessors of the   = 4
```

The continuation probability uses 4, not 18. Repeated occurrences of `the` inside the same kinds of phrases do not make it look broadly distributed across contexts.

### 9.3 Calculate a word with only one predecessor

For `fly`:

```python
predecessors_by_word["fly"] = {"birds"}
```

Therefore:

```text
continuation_count("fly") = 1
P_KN(fly) = 1 / 20
          = 0.05
```

The raw unigram count of `fly` is 2, but both occurrences follow the same word, `birds`. Kneser-Ney therefore gives `fly` a relatively small continuation probability.

### 9.4 Calculate all continuation probabilities

The complete calculation is:

```text
word       predecessors                         count   probability
---------------------------------------------------------------------
</s>       {mat, rug}                              2       2/20
a          {<s>}                                   1       1/20
birds      {<s>}                                   1       1/20
cat        {the, a}                                2       2/20
dog        {the, a}                                2       2/20
fly        {birds}                                 1       1/20
mat        {the}                                   1       1/20
near       {sat}                                   1       1/20
on         {sat}                                   1       1/20
over       {fly}                                   1       1/20
rug        {the}                                   1       1/20
sat        {cat, dog}                              2       2/20
the        {<s>, on, near, over}                   4       4/20
```

The numerator column is exactly the size of each predecessor set. The denominator is 20 for every word.

### 9.5 Check that the continuation probabilities sum to one

Add the numerators:

```text
2 + 1 + 1 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 2 + 4
= 20
```

Therefore:

```text
sum of continuation probabilities
= 20 / 20
= 1
```

**This is an important correctness check. The continuation probabilities form a valid probability distribution over the 13-word vocabulary.**

### What we know after Step 9

```text
Distinct bigram types                  -> 20
Continuation denominator               -> 20
P_KN(the)                              -> 4/20 = 0.2
P_KN(fly)                              -> 1/20 = 0.05
Sum of all continuation probabilities  -> 1
```

The next step will explain how the model estimates the three discount values `D1`, `D2`, and `D3+` from the n-gram counts.


## Step 10: Estimate `D1`, `D2`, and `D3+`

The constructor calls:

```python
self.discounts_by_order = self._build_discounts(discounts_by_order)
```

Because we did not supply custom discounts, `_build_discounts()` calls:

```python
estimate_discounts(self.ngram_counts[order].values())
```

It does this separately for bigrams (`order=2`) and trigrams (`order=3`).

### 10.1 What does `ngram_counts` contain here?

For discount estimation, the function receives **only the counts** of the distinct n-gram types—not the n-gram tuples themselves.

For example, the complete bigram dictionary contains entries such as:

```python
("<s>", "the"): 6
("the", "cat"): 3
("sat", "on"): 8
```

The function receives the values:

```text
6, 3, 8, ...
```

It then asks:

> How many different n-gram types have count 1? Count 2? Count 3? Count 4?

These are called frequency-of-frequencies:

```text
n1 = number of n-gram types whose count is exactly 1
n2 = number of n-gram types whose count is exactly 2
n3 = number of n-gram types whose count is exactly 3
n4 = number of n-gram types whose count is exactly 4
```

The `n` in `n1`, `n2`, and so on does not mean n-gram order. It refers to the count of an individual n-gram type.

### 10.2 Frequency-of-frequencies for bigrams

The bigram counts have this distribution:

```text
bigram count of a type     number of bigram types
--------------------------------------------------
1                          0
2                          8
3                          2
4                          1
5                          4
6                          1
7                          2
8                          2
```

Therefore:

```text
n1 = 0
n2 = 8
n3 = 2
n4 = 1
```

Check the number of distinct bigram types:

```text
0 + 8 + 2 + 1 + 4 + 1 + 2 + 2 = 20
```

This agrees with the 20 keys in `counts_by_order[2]`.

### 10.3 Why the bigram `D1` uses a fallback

The standard formula for `D1` needs both `n1` and `n2`. Here:

```text
n1 = 0
n2 = 8
```

There are no bigram types that occurred exactly once. Dividing by `n1` would therefore be impossible.

The code handles this small-corpus situation with:

```python
y = 0.5
d1 = 0.5
```

So:

```text
D1_bigram = 0.5
```

This is a teaching fallback used because our corpus is small. It is not saying that the mathematically estimated value is naturally 0.5; it says that the code needs a safe positive value when the required count category is absent.

### 10.4 Calculate the bigram `D2`

The code uses:

```text
D2 = 2 - 3 * y * n3 / n2
```

Substitute the bigram values:

```text
D2_bigram = 2 - 3 * 0.5 * 2 / 8
           = 2 - 3 / 8
           = 2 - 0.375
           = 1.625
```

Therefore:

```text
D2_bigram = 1.625
```

### 10.5 Calculate the bigram `D3+`

The code uses:

```text
D3+ = 3 - 4 * y * n4 / n3
```

Substitute the bigram values:

```text
D3+_bigram = 3 - 4 * 0.5 * 1 / 2
            = 3 - 1
            = 2.0
```

Therefore, the complete bigram discount tuple is:

```python
discounts_by_order[2] = (0.5, 1.625, 2.0)
```

### 10.6 Frequency-of-frequencies for trigrams

The trigram counts have this distribution:

```text
trigram count of a type    number of trigram types
---------------------------------------------------
1                          4
2                          9
3                          4
4                          4
5                          1
6                          0
7                          1
8                          1
```

Therefore:

```text
n1 = 4
n2 = 9
n3 = 4
n4 = 4
```

Check the number of distinct trigram types:

```text
4 + 9 + 4 + 4 + 1 + 0 + 1 + 1 = 24
```

This agrees with the 24 keys in `counts_by_order[3]`.

### 10.7 Calculate the trigram `y`

Because both `n1` and `n2` are available, the code calculates:

```text
y = n1 / (n1 + 2 * n2)
```

Substitute the trigram values:

```text
y = 4 / (4 + 2 * 9)
  = 4 / 22
  = 2 / 11
  = 0.1818181818...
```

### 10.8 Calculate the trigram discounts

For `D1`:

```text
D1_trigram = 1 - 2 * y * n2 / n1
            = 1 - 2 * (2/11) * 9 / 4
            = 1 - 9/11
            = 2/11
            = 0.1818181818...
```

For `D2`:

```text
D2_trigram = 2 - 3 * y * n3 / n2
            = 2 - 3 * (2/11) * 4 / 9
            = 2 - 8/33
            = 58/33
            = 1.7575757576...
```

For `D3+`:

```text
D3+_trigram = 3 - 4 * y * n4 / n3
             = 3 - 4 * (2/11) * 4 / 4
             = 3 - 8/11
             = 25/11
             = 2.2727272727...
```

Therefore:

```python
discounts_by_order[3] = (
    0.1818181818...,  # D1
    1.7575757576...,  # D2
    2.2727272727...,  # D3+
)
```

### 10.9 How `discount_for_count()` selects a discount

Once the tuples exist, the helper function chooses a discount based on the observed count:

```python
if count == 1:
    return discounts[0]  # D1
if count == 2:
    return discounts[1]  # D2
return discounts[2]      # D3+
```

For example, **for trigrams**:

```text
count 1 -> D1_trigram = 0.1818181818...
count 2 -> D2_trigram = 1.7575757576...
count 3 -> D3+_trigram = 2.2727272727...
count 8 -> D3+_trigram = 2.2727272727...
```

The `D3+` label means “use this discount for count 3 or greater.” It does not mean that every n-gram has count exactly 3.

### What we know after Step 10

```python
discounts_by_order = {
    2: (0.5, 1.625, 2.0),
    3: (0.1818181818..., 1.7575757576..., 2.2727272727...),
}
```

The model now knows how much to subtract from an observed bigram or trigram count. It has not yet calculated any lambda value or final conditional probability. The next step will calculate `history_count()` and `lambda_for_history()` for one concrete history.

## Step 11: Calculate lambda for one history

The model has now prepared n-gram counts, predecessor sets, continuation probabilities, and discount values. It can now calculate how much probability mass should be transferred from a higher-order model to a lower-order model.

### 11.1 What is a history?

A history is the context before the word we want to predict.

```text
history:        the cat
predicted word: sat
full trigram:   the cat sat
```

In code:

```python
history = ("the", "cat")
```

Because the history has two words, adding one predicted word creates a trigram:

```text
history length = 2
n-gram order   = 2 + 1 = 3
```

### 11.2 Calculate history_count(("the", "cat"))

The function is:

```python
def history_count(self, history):
    history = tuple(history)
    if not history:
        return 0

    return self.count(len(history), history)
```

For history = ("the", "cat"):

```python
len(history) = 2
self.count(2, ("the", "cat"))
```

The final bigram dictionary contains:

```python
("the", "cat"): 3
```

Therefore:

```text
history_count(("the", "cat")) = 3
```

The phrase "the cat" occurred in these three sentences:

```text
the cat sat on the mat
the cat sat on the rug
the cat sat near the mat
```

### 11.3 Find the observed continuation

To calculate lambda, find every trigram whose first two tokens equal the history ("the", "cat").

The final trigram dictionary contains:

```python
("the", "cat", "sat"): 3
```

So the only observed continuation is:

```text
word after "the cat" -> sat
count                 -> 3
```

The history count agrees with the continuation count:

```text
history count       = 3
continuation counts = 3
```

### 11.4 Select the discount

The matching trigram has count 3. The helper function uses:

```python
if count == 1:
    use D1
if count == 2:
    use D2
otherwise:
    use D3+
```

Therefore:

```text
selected discount = D3+_trigram
                  = 25/11
                  = 2.2727272727...
```

### 11.5 Understand what discounting removes

The original trigram count is 3. After discounting:

```text
discounted count
= 3 - 25/11
= 33/11 - 25/11
= 8/11
= 0.7272727273...
```

The model keeps an effective count of 8/11 for the direct trigram contribution and removes 25/11 count units for lower-order information.

This is the meaning of discounting: reduce the observed n-gram's count and reserve the removed amount for the lower-order model.

### 11.6 Calculate lambda_for_history(("the", "cat"))

The important part of the function is:

```python
history_count = self.history_count(history)
removed_count = 0.0

for ngram, count in self.ngram_counts[order].items():
    if ngram[:-1] == history:
        removed_count += self.discount_for_count(count, order)

return removed_count / history_count
```

For our history:

```text
history                  = ("the", "cat")
order                    = 3
history_count            = 3
matching trigram         = ("the", "cat", "sat")
selected discount        = 25/11
removed_count            = 25/11
```

Therefore:

```text
lambda(("the", "cat"))
= removed_count / history_count
= (25/11) / 3
= 25/33
= 0.7575757576...
```

So:

```text
lambda_for_history(("the", "cat")) = 25/33
```

Lambda is the fraction of probability mass handed to the lower-order model.

The direct higher-order mass is:

```text
1 - lambda
= 1 - 25/33
= 8/33
= 0.2424242424...
```

This agrees with the discounted count divided by the history count:

```text
(8/11) / 3 = 8/33
```

### 11.7 Why divide by the history count?

The discount is measured in count units. Lambda must be a fraction of probability mass. Dividing by the total history count converts:

```text
removed count / total history count
= (25/11) / 3
= 25/33
```

For this history, approximately 75.76% of the mass goes to lower-order evidence, and approximately 24.24% remains as direct trigram evidence.

### 11.8 A bigram example: history ("cat",)

Now use a one-word history:

```python
history = ("cat",)
```

This creates a bigram when a predicted word is added:

```text
order = len(("cat",)) + 1
      = 1 + 1
      = 2
```

The unigram count is:

```python
("cat",): 5
```

Therefore:

```text
history_count(("cat",)) = 5
```

The observed bigram is:

```python
("cat", "sat"): 5
```

Its count is 5, so it uses D3+ for bigrams:

```text
D3+_bigram = 2.0
removed_count = 2.0
```

Thus:

```text
lambda(("cat",)) = 2.0 / 5 = 0.4
direct mass       = 1 - 0.4 = 0.6
```

### 11.9 An unseen history

Consider:

```python
history = ("the", "birds")
```

This bigram does not occur:

```text
history_count(("the", "birds")) = 0
```

The function returns 1.0 immediately:

```python
if history_count == 0:
    return 1.0
```

Therefore:

```text
lambda_for_history(("the", "birds")) = 1.0
```

All probability mass comes from the lower-order model because there is no higher-order evidence for this history.

### What we know after Step 11

```text
history_count(("the", "cat"))  -> 3
lambda(("the", "cat"))         -> 25/33 = 0.7575757576...
direct mass                    -> 8/33 = 0.2424242424...

history_count(("cat",))        -> 5
lambda(("cat",))               -> 0.4
direct mass                    -> 0.6

lambda(("the", "birds"))       -> 1.0 because the history is unseen
```

Lambda is not an unrelated extra probability. It is the fraction of mass removed by discounting and handed to the lower-order model.

## Step 12: Calculate one complete recursive MKN probability

We will calculate:

```python
model.probability("sat", ("the", "cat"))
```

This means:

```text
P(sat | the cat)
```

Because the history has two words, the model starts at the trigram level.

### 12.1 The recursive structure

The function follows this chain:

```text
P(sat | the cat)
    -> trigram level
    -> needs P(sat | cat)
        -> bigram level
        -> needs P(sat)
            -> continuation level
```

In code, the history becomes shorter through:

```python
history[1:]
```

For our history:

```python
("the", "cat")[1:] = ("cat",)
("cat",)[1:]      = ()
```

The empty history means that the model has reached the continuation probability.

### 12.2 Start at the trigram level

The history is:

```python
history = ("the", "cat")
word = "sat"
```

The full trigram is:

```python
ngram = ("the", "cat", "sat")
```

From the final counts:

```text
count("the", "cat", "sat") = 3
history_count(("the", "cat")) = 3
```

Since the trigram count is 3, use:

```text
D3+_trigram = 25/11
```

The direct trigram contribution is:

```text
direct contribution
= (count - selected discount) / history count
= (3 - 25/11) / 3
= (8/11) / 3
= 8/33
= 0.2424242424...
```

The lambda for this history was calculated in Step 11:

```text
lambda(("the", "cat")) = 25/33
```

So the trigram calculation is:

```text
P(sat | the cat)
= 8/33 + (25/33) * P(sat | cat)
```

The function now recursively asks the lower-order model to calculate P(sat | cat).

### 12.3 Move to the bigram level

Now the history is:

```python
history = ("cat",)
word = "sat"
```

The full bigram is:

```python
ngram = ("cat", "sat")
```

From the final counts:

```text
count("cat", "sat") = 5
history_count(("cat",)) = 5
```

Since the bigram count is 5, use:

```text
D3+_bigram = 2.0
```

The direct bigram contribution is:

```text
direct contribution
= (count - selected discount) / history count
= (5 - 2) / 5
= 3/5
= 0.6
```

The lambda for this history is:

```text
lambda(("cat",)) = 0.4 = 2/5
```

Therefore:

```text
P(sat | cat)
= 3/5 + (2/5) * P(sat)
```

The model now recursively asks for the continuation probability P(sat).

### 12.4 Reach the continuation level

At the empty-history level, the code executes:

```python
if not history:
    probability = self.continuation_probability(word)
```

For sat, the predecessor set is:

```python
predecessors_by_word["sat"] = {"cat", "dog"}
```

Therefore:

```text
continuation_count("sat") = 2
continuation_denominator   = 20
```

The continuation probability is:

```text
P(sat)
= 2 / 20
= 1/10
= 0.1
```

This is the base probability from which the recursion begins.

### 12.5 Finish the bigram calculation

Substitute P(sat) = 1/10:

```text
P(sat | cat)
= 3/5 + (2/5) * (1/10)
= 3/5 + 2/50
= 3/5 + 1/25
= 15/25 + 1/25
= 16/25
= 0.64
```

So:

```text
P(sat | cat) = 0.64
```

### 12.6 Finish the trigram calculation

Substitute P(sat | cat) = 16/25:

```text
P(sat | the cat)
= 8/33 + (25/33) * (16/25)
```

The 25 in the lambda cancels with the 25 in the lower-order probability:

```text
P(sat | the cat)
= 8/33 + 16/33
= 24/33
= 8/11
= 0.7272727273...
```

Therefore:

```text
P(sat | the cat) = 8/11 = 0.7272727273...
```

### 12.7 Match the calculation to the code trace

The code records one trace entry at each level. Conceptually:

```python
[
    {
        "order": 3,
        "history": ("the", "cat"),
        "count": 3,
        "selected_discount": 25/11,
        "history_count": 3,
        "direct_contribution": 8/33,
        "lambda": 25/33,
        "lower_order_probability": 16/25,
        "probability": 8/11,
    },
    {
        "order": 2,
        "history": ("cat",),
        "count": 5,
        "selected_discount": 2.0,
        "history_count": 5,
        "direct_contribution": 3/5,
        "lambda": 2/5,
        "lower_order_probability": 1/10,
        "probability": 16/25,
    },
    {
        "order": 1,
        "history": (),
        "lower_order_probability": 1/10,
        "probability": 1/10,
    },
]
```

The actual trace stores decimal floating-point values, but these fractions show the exact hand calculation more clearly.

### What we know after Step 12

```text
P(sat)               -> 1/10 = 0.1
P(sat | cat)         -> 16/25 = 0.64
P(sat | the cat)     -> 8/11 = 0.7272727273...
```

The next step will verify that recursive probabilities form a valid distribution by summing probabilities over the complete vocabulary.

## Step 13: Verify that probabilities sum to one

A language model must produce a valid probability distribution. For any fixed history, the probabilities of all possible next tokens must add up to one.

The code checks this with:

```python
def normalization_total(self, history=()):
    return sum(self.probability(word, history) for word in self.vocabulary)
```

The vocabulary contains 13 possible predicted tokens. The start marker is excluded, but the end marker is included.

### 13.1 Normalization at the continuation level

At the empty-history level:

```text
P(word) = continuation_count(word) / 20
```

The continuation-count numerators are:

```text
2 + 1 + 1 + 2 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 2 + 4 = 20
```

Therefore:

```text
sum of P(word)
= 20 / 20
= 1
```

The base continuation distribution is normalized.

### 13.2 Normalization for the history ("cat",)

For the history ("cat",), the observed bigram is:

```text
("cat", "sat") has count 5
```

The probability of sat is:

```text
P(sat | cat) = 16/25
```

For every other word, the bigram count is zero. Those words receive only the lower-order contribution:

```text
P(word | cat)
= (2/5) * P(word)
```

The continuation numerators for all words add to 20. The numerator for sat is 2, so all other words together have numerator:

```text
20 - 2 = 18
```

Their combined probability is:

```text
sum of P(other word | cat)
= (2/5) * (18/20)
= (2/5) * (9/10)
= 18/50
= 9/25
```

Now add the observed word sat:

```text
sum of P(word | cat)
= P(sat | cat) + sum of P(other word | cat)
= 16/25 + 9/25
= 25/25
= 1
```

The lower-order probability mass is exactly what fills the probability space not occupied by the discounted observed bigram.

### 13.3 Normalization for the history ("the", "cat")

For the history ("the", "cat"), the only observed trigram is:

```text
("the", "cat", "sat") has count 3
```

We already calculated:

```text
P(sat | the cat) = 8/11
lambda(("the", "cat")) = 25/33
```

For every word other than sat, the trigram count is zero. Therefore, those words receive:

```text
P(word | the cat)
= (25/33) * P(word | cat)
```

At the bigram level, all words other than sat together have probability 9/25. Therefore, their combined trigram-level probability is:

```text
sum of P(other word | the cat)
= (25/33) * (9/25)
= 9/33
= 3/11
```

Now add sat:

```text
sum of P(word | the cat)
= 8/11 + 3/11
= 11/11
= 1
```

This demonstrates the recursive normalization idea:

```text
higher-order direct mass
+ lower-order mass passed through lambda
= 1
```

### 13.4 Connect the calculation to the code

The code evaluates:

```python
model.normalization_total(("the", "cat"))
```

It loops through every word in the vocabulary:

```python
sum(
    self.probability(word, ("the", "cat"))
    for word in self.vocabulary
)
```

For this toy model, the expected result is:

```text
normalization_total(("the", "cat")) = 1.0000000000
```

The small decimal difference that can sometimes appear in computer output is floating-point rounding. The mathematical result is exactly 1.

### 13.5 Unseen histories also remain normalized

If a history has never occurred, the code returns the lower-order probability directly:

```python
if history_count == 0:
    return lower_probability
```

For example:

```text
P(word | the birds) = P(word | birds)
```

So an unseen history does not create a new unnormalized distribution. It simply reuses a lower-order distribution that is already normalized.

### What we know after Step 13

```text
sum of P(word)                 -> 1
sum of P(word | cat)           -> 1
sum of P(word | the cat)       -> 1
```

The next step will calculate a probability for an observed history with an unseen continuation, so we can see how backoff/interpolation behaves when the requested n-gram itself has count zero.

## Step 14: Calculate an unseen continuation

An observed history and an observed n-gram are different things.

For example:

```text
history ("cat",)             -> observed
bigram ("cat", "fly")         -> unseen
```

The history can have evidence even when a particular next word has never appeared after it.

### 14.1 Calculate P(fly | cat)

We ask the model for:

```python
model.probability("fly", ("cat",))
```

The history is:

```python
history = ("cat",)
```

Its count is:

```text
history_count(("cat",)) = 5
```

The requested bigram is:

```python
ngram = ("cat", "fly")
```

From the final bigram counts:

```text
count("cat", "fly") = 0
```

### 14.2 What discount is used for count zero?

The helper function begins with:

```python
if count <= 0:
    return 0.0
```

Therefore:

```text
selected_discount = 0
```

This is important: an unseen n-gram does not receive D1. D1 is used for an observed n-gram whose count is exactly 1. A count-zero n-gram has no direct higher-order evidence, so its direct contribution is zero.

### 14.3 Calculate the direct contribution

The code calculates direct contribution only when count is greater than zero:

```python
direct_contribution = 0.0

if count > 0:
    direct_contribution = max(count - selected_discount, 0.0) / history_count
```

Here count = 0, so:

```text
direct contribution = 0
```

The model therefore does not assign any probability directly from the nonexistent bigram ("cat", "fly").

### 14.4 Use the lambda mass

The history ("cat",) is observed, so its lambda is:

```text
lambda(("cat",)) = 2/5
```

The model recursively asks for the lower-order probability:

```text
P(fly)
= continuation_count("fly") / continuation_denominator
= 1 / 20
```

Therefore:

```text
P(fly | cat)
= direct contribution + lambda * P(fly)
= 0 + (2/5) * (1/20)
= 2/100
= 1/50
= 0.02
```

So the unseen bigram still receives a positive probability:

```text
P(fly | cat) = 0.02
```

The probability comes entirely from the lower-order continuation distribution.

### 14.5 Why this is interpolation, not only backoff

For an observed history, the code always calculates:

```text
direct contribution + lambda * lower-order probability
```

For ("cat",):

```text
P(word | cat)
= direct bigram contribution
+ lambda(("cat",)) * P(word)
```

For the particular unseen word fly, the direct contribution happens to be zero:

```text
P(fly | cat)
= 0 + lambda(("cat",)) * P(fly)
```

The lower-order model is therefore still interpolated into the result, even though the requested bigram was unseen.

### 14.6 Calculate P(on | the cat)

Now use a two-word observed history:

```python
model.probability("on", ("the", "cat"))
```

The history count is:

```text
history_count(("the", "cat")) = 3
```

The requested trigram is:

```python
("the", "cat", "on")
```

Its count is zero:

```text
count("the", "cat", "on") = 0
```

Therefore:

```text
direct trigram contribution = 0
lambda(("the", "cat"))      = 25/33
```

We must calculate the lower-order probability P(on | cat). The bigram ("cat", "on") is also unseen:

```text
count("cat", "on") = 0
direct bigram contribution = 0
lambda(("cat",)) = 2/5
```

The continuation probability of on is:

```text
P(on) = 1 / 20
```

Thus:

```text
P(on | cat)
= 0 + (2/5) * (1/20)
= 1/50
```

Now return to the trigram level:

```text
P(on | the cat)
= 0 + (25/33) * (1/50)
= 25/1650
= 1/66
= 0.0151515152...
```

So:

```text
P(on | the cat) = 1/66 = 0.0151515152...
```

This calculation demonstrates two levels of unseen continuation:

```text
unseen trigram ("the", "cat", "on")
    -> back to P(on | cat)

unseen bigram ("cat", "on")
    -> back to P(on)
```

### 14.7 How the trace identifies this case

For a seen history and unseen n-gram, the code sets:

```python
event = "seen_history_unseen_ngram"
```

For P(fly | cat), the trace conceptually contains:

```python
{
    "order": 2,
    "history": ("cat",),
    "event": "seen_history_unseen_ngram",
    "ngram": ("cat", "fly"),
    "count": 0,
    "selected_discount": 0.0,
    "history_count": 5,
    "direct_contribution": 0.0,
    "lambda": 0.4,
    "lower_order_probability": 0.05,
    "probability": 0.02,
}
```

For an unseen history, the event would instead be:

```python
"unseen_history_backoff"
```

The distinction is:

```text
seen history + unseen n-gram
    -> interpolate with zero direct contribution

unseen history
    -> directly use the lower-order probability
```

### What we know after Step 14

```text
P(fly | cat)       -> 1/50 = 0.02
P(on | cat)        -> 1/50 = 0.02
P(on | the cat)    -> 1/66 = 0.0151515152...
```

The next step will calculate a probability for an unseen history and compare it with the seen-history/unseen-n-gram case.

## Step 15: Validate a probability query

Before calculating a probability, the public method probability() checks that the request is valid.

The method begins:

```python
def probability(self, word, history=(), return_trace=False):
    self._check_word(word)
    history = tuple(history)

    if len(history) >= self.max_order:
        raise ValueError("history is longer than the supported n-gram order")

    trace = []
    probability = self._probability(word, history, trace)

    if return_trace:
        return probability, trace

    return probability
```

This public method is the entry point for probability queries. It prepares the input and then delegates the actual recursive calculation to _probability().

### 15.1 Check that the word belongs to the vocabulary

The first line is:

```python
self._check_word(word)
```

The helper function is:

```python
def _check_word(self, word):
    if word not in self.vocabulary:
        raise ValueError(f"word is not in the vocabulary: {word!r}")
```

For a valid query:

```python
model.probability("sat", ("the", "cat"))
```

sat is in the vocabulary, so the method continues.

For an invalid query:

```python
model.probability("elephant", ("the", "cat"))
```

elephant does not occur in the toy corpus. The method stops and raises:

```text
ValueError: word is not in the vocabulary: 'elephant'
```

This prevents the model from assigning probability to a word it was never allowed to predict.

### 15.2 Convert the history to a tuple

The next line is:

```python
history = tuple(history)
```

This makes the history have a consistent form.

For example:

```python
history = ["the", "cat"]
history = tuple(history)
```

The result is:

```python
("the", "cat")
```

The model uses tuples because n-gram keys in the dictionaries are tuples. This allows the history to be compared directly with ngram[:-1].

If the history is already a tuple, conversion changes nothing:

```python
tuple(("the", "cat")) == ("the", "cat")
```

### 15.3 Check the maximum supported history length

This model has:

```python
max_order = 3
```

A trigram has two history words plus one predicted word. Therefore, the longest supported history has length 2.

The method checks:

```python
if len(history) >= self.max_order:
    raise ValueError(...)
```

A valid longest history is:

```python
("the", "cat")
```

Its length is 2:

```text
len(("the", "cat")) = 2
2 < 3
```

A history of length 3 would require a 4-gram:

```python
model.probability("sat", ("the", "cat", "on"))
```

That query is rejected because the model only supports up to trigrams:

```text
ValueError: history is longer than the supported n-gram order
```

### 15.4 Create the trace container

The method creates:

```python
trace = []
```

This empty list is passed into the recursive helper:

```python
probability = self._probability(word, history, trace)
```

Each recursive level adds a dictionary describing its calculation. For the query P(sat | the cat), the trace eventually contains entries for:

```text
order 3: history ("the", "cat")
order 2: history ("cat",)
order 1: history ()
```

The trace does not calculate a second probability. It records the calculations that already occurred.

### 15.5 Choose whether to return the trace

If return_trace is false, the method returns only the number:

```python
probability = model.probability("sat", ("the", "cat"))
```

Conceptually:

```python
0.7272727273...
```

If return_trace is true, it returns a pair:

```python
probability, trace = model.probability(
    "sat",
    ("the", "cat"),
    return_trace=True,
)
```

Conceptually:

```python
(
    0.7272727273...,
    [
        # trigram calculation
        # bigram calculation
        # continuation calculation
    ],
)
```

This is useful for learning because we can inspect every recursive step instead of seeing only the final number.

### What we know after Step 15

```text
valid word                  -> calculation continues
unknown word                -> ValueError
history length 0, 1, or 2   -> supported for max_order=3
history length 3 or more    -> ValueError
return_trace=False          -> return probability only
return_trace=True           -> return probability and trace
```

The next step will inspect the internal _probability() branches and connect each branch to the cases we calculated by hand.

## Step 16: Understand the recursive _probability() branches

The public probability() method validates the query and then calls:

```python
self._probability(word, history, trace)
```

The internal function has three main cases:

```text
Case 1: empty history
Case 2: unseen history
Case 3: seen history
```

### 16.1 Case 1: empty history

The first branch is:

```python
if not history:
    probability = self.continuation_probability(word)
    trace.append(...)
    return probability
```

An empty tuple means that there is no higher-order context left:

```python
history = ()
```

The function therefore returns the continuation probability directly.

For sat:

```text
P(sat) = continuation_count("sat") / continuation_denominator
       = 2 / 20
       = 1/10
       = 0.1
```

The recursion stops here. It does not try to create a zero-order n-gram.

The trace records this as:

```python
{
    "order": 1,
    "history": (),
    "event": "continuation_probability",
    "probability": 0.1,
}
```

The actual trace contains additional fields, but these are the important ones.

### 16.2 Case 2: unseen history

After checking for an empty history, the function calculates:

```python
order = len(history) + 1
history_count = self.history_count(history)
```

If the history count is zero, this branch runs:

```python
if history_count == 0:
    lower_history = history[1:]

    step = {
        "order": order,
        "history": history,
        "event": "unseen_history_backoff",
        "direct_contribution": 0.0,
        "lambda": 1.0,
    }

    trace.append(step)
    lower_probability = self._probability(word, lower_history, trace)
    step["lower_order_probability"] = lower_probability
    step["probability"] = lower_probability
    return lower_probability
```

There is no direct higher-order contribution because the history was never observed.

The function removes the first token from the history:

```python
lower_history = history[1:]
```

For example:

```python
("the", "birds")[1:] = ("birds",)
```

This means:

```text
P(word | the birds) = P(word | birds)
```

The function is not assigning lambda by calculating discounts here. It simply sets lambda to 1.0 because 100% of the probability must come from the lower-order model.

### 16.3 Calculate an unseen-history example

Consider:

```python
model.probability("sat", ("the", "birds"))
```

The history ("the", "birds") is absent from the bigram dictionary:

```text
history_count(("the", "birds")) = 0
```

So the function backs off to:

```text
lower_history = ("birds",)
```

Now calculate P(sat | birds).

The history count is:

```text
history_count(("birds",)) = count(("birds",)) = 2
```

The observed bigram after birds is:

```text
("birds", "fly") has count 2
```

The bigram count 2 selects:

```text
D2_bigram = 1.625
```

The removed mass is:

```text
removed_count = 1.625
lambda(("birds",)) = 1.625 / 2
                  = 0.8125
                  = 13/16
```

The requested bigram ("birds", "sat") is unseen, so its direct contribution is zero. The lower-order probability is:

```text
P(sat) = 1/10
```

Therefore:

```text
P(sat | birds)
= 0 + (13/16) * (1/10)
= 13/160
= 0.08125
```

Because ("the", "birds") is unseen:

```text
P(sat | the birds) = P(sat | birds) = 13/160 = 0.08125
```

### 16.4 Case 3: seen history

If the history count is not zero, the function continues:

```python
ngram = history + (word,)
count = self.count(order, ngram)
selected_discount = self.discount_for_count(count, order)
direct_contribution = 0.0

if count > 0:
    direct_contribution = max(
        count - selected_discount,
        0.0,
    ) / history_count

interpolation_weight = self.lambda_for_history(history)
```

This branch handles both:

```text
seen history + seen n-gram
seen history + unseen n-gram
```

### 16.5 Seen history and seen n-gram

For:

```python
model.probability("sat", ("the", "cat"))
```

The history is observed:

```text
history_count(("the", "cat")) = 3
```

The full trigram is also observed:

```text
count(("the", "cat", "sat")) = 3
```

So count > 0 and the direct contribution is calculated:

```text
direct contribution = (3 - 25/11) / 3
                    = 8/33
```

The event label is:

```python
"seen_history_seen_ngram"
```

### 16.6 Seen history and unseen n-gram

For:

```python
model.probability("fly", ("cat",))
```

The history is observed:

```text
history_count(("cat",)) = 5
```

But the full bigram is unseen:

```text
count(("cat", "fly")) = 0
```

Therefore:

```text
selected_discount  = 0
direct_contribution = 0
```

The event label is:

```python
"seen_history_unseen_ngram"
```

The function still calculates lambda and recursively obtains the lower-order probability:

```text
P(fly | cat)
= 0 + lambda(("cat",)) * P(fly)
= 0 + (2/5) * (1/20)
= 1/50
```

### 16.7 Assemble the final probability

After calculating the direct contribution and lower-order probability, the function uses:

```python
probability = (
    direct_contribution
    + interpolation_weight * lower_probability
)
```

This is the central interpolated MKN equation in the code:

```text
higher-order direct contribution
+ lambda(history) * lower-order probability
```

The function then records:

```python
step["lower_order_probability"] = lower_probability
step["probability"] = probability
return probability
```

The lower-order result is calculated first by recursion, then inserted into the higher-order equation while the recursion unwinds.

### 16.8 The recursion unwinds from the bottom upward

For P(sat | the cat), the order of events is:

```text
1. Start at P(sat | the cat)
2. Ask for P(sat | cat)
3. Ask for P(sat)
4. Calculate P(sat) = 1/10
5. Calculate P(sat | cat) = 16/25
6. Calculate P(sat | the cat) = 8/11
7. Return 8/11
```

This is why the trace is built while descending into lower orders but receives final probability values while the recursion returns upward.

### What we know after Step 16

```text
empty history
    -> use continuation probability

unseen history
    -> lambda = 1 and use a shorter history

seen history + seen n-gram
    -> direct contribution + lambda * lower-order probability

seen history + unseen n-gram
    -> zero direct contribution + lambda * lower-order probability
```

The main recursive logic of the MKN model is now fully connected to the code. The next step will inspect the final normalization checks and the learner-facing output produced by print_demo().

## Step 17: Understand the final demo and verification output

After the model has been built, main() calls:

```python
print_demo(model)
```

The print_demo() function does not train the model or change any values. It simply displays selected information so we can inspect the result.

### 17.1 Basic model summary

The first output is:

```text
sentences: 12
vocabulary size (excluding <s>): 13
distinct bigram types: 20
```

These values mean:

```text
12 -> number of training sentences
13 -> number of tokens the model can predict
20 -> number of distinct bigram types
```

The start marker is excluded from the vocabulary, but the end marker remains a prediction candidate.

### 17.2 Printed discount values

The next output is:

```text
order 2: D1=0.500000, D2=1.625000, D3+=2.000000
order 3: D1=0.181818, D2=1.757576, D3+=2.272727
```

These are the same values calculated in Step 10. The output uses six decimal places, so these are rounded versions of the stored values.

### 17.3 Printed selected counts

The demo checks:

```python
selected_counts = [
    (2, ("the", "cat")),
    (3, ("the", "cat", "sat")),
    (2, ("cat", "fly")),
]
```

The results are:

```text
c(("the", "cat"))        = 3
c(("the", "cat", "sat")) = 3
c(("cat", "fly"))        = 0
```

These deliberately test:

```text
observed bigram
observed trigram
unseen bigram
```

The third count confirms that an n-gram can be absent even when its history, "cat", is present.

### 17.4 Printed trace for P(sat | the cat)

The demo asks:

```python
model.probability("sat", ("the", "cat"), return_trace=True)
```

The final output is:

```text
P(sat | ("the", "cat")) = 0.7272727273
```

This is:

```text
P(sat | the cat) = 8/11 = 0.7272727273...
```

The trace shows:

```text
order 3 -> history ("the", "cat")
order 2 -> history ("cat",)
order 1 -> empty history
```

Important values in the trace are:

```text
trigram direct contribution = 0.2424242424...
trigram lambda             = 0.7575757575...
bigram probability         = 0.64
continuation probability   = 0.1
final probability          = 0.7272727273...
```

### 17.5 Printed trace for P(fly | cat)

The second query is:

```python
model.probability("fly", ("cat",), return_trace=True)
```

The output is:

```text
P(fly | ("cat",)) = 0.0200000000
```

The trace identifies:

```text
event               -> seen_history_unseen_ngram
count               -> 0
direct contribution -> 0.0
lambda              -> 0.4
P(fly)              -> 0.05
final probability   -> 0.02
```

The calculation is:

```text
P(fly | cat)
= 0 + (2/5) * (1/20)
= 1/50
= 0.02
```

This confirms that an unseen continuation receives a nonzero probability through the lower-order model.

### 17.6 Normalization checks printed by the demo

The demo checks:

```python
histories = [
    (),
    ("the",),
    ("cat",),
    ("the", "cat"),
]
```

For each history, it calculates the sum over all 13 vocabulary items. The output is:

```text
()             -> 1.0000000000 [PASS]
("the",)       -> 1.0000000000 [PASS]
("cat",)       -> 1.0000000000 [PASS]
("the", "cat") -> 1.0000000000 [PASS]
```

Each PASS means that the probability distribution for that history sums to one, up to tiny floating-point rounding.

The code uses:

```python
math.isclose(total, 1.0, abs_tol=1e-9)
```

The tolerance allows tiny computer-representation errors while still detecting a meaningful normalization failure.

### 17.7 What the complete program has demonstrated

```text
1. loaded the corpus
2. added sentence boundaries
3. counted unigrams, bigrams, and trigrams
4. built the prediction vocabulary
5. built predecessor sets
6. calculated continuation probabilities
7. estimated discounts
8. calculated lambda values
9. recursively calculated MKN probabilities
10. handled unseen histories and unseen n-grams
11. verified normalization
```

### What we know after Step 17

```text
P(sat | the cat) -> 0.7272727273
P(fly | cat)     -> 0.0200000000
all demo checks   -> PASS
```

The core implementation walkthrough is now complete. The next step can create the consolidated final reference section containing all finalized MKN model values.

# Final MKN model values

This section is the consolidated reference for the completed toy-corpus model. The values below are the final values used by the code.

## 1. Corpus and model settings

```text
corpus sentences       = 12
maximum n-gram order   = 3
start token            = <s>
end token              = </s>
total token positions  = 94
```

## 2. Vocabulary

The model predicts 13 tokens. The start token is excluded; the end token is included.

```python
vocabulary = (
    "</s>", "a", "birds", "cat", "dog", "fly",
    "mat", "near", "on", "over", "rug", "sat", "the",
)
```

## 3. Final n-gram counts

The dictionary key is the n-gram tuple. The value is its corpus count.

### Unigram counts

```python
{
    ("<s>",): 12,
    ("the",): 18,
    ("cat",): 5,
    ("sat",): 10,
    ("on",): 8,
    ("mat",): 7,
    ("</s>",): 12,
    ("rug",): 5,
    ("near",): 2,
    ("dog",): 5,
    ("a",): 4,
    ("birds",): 2,
    ("fly",): 2,
    ("over",): 2,
}
```

### Bigram counts

```python
{
    ("<s>", "the"): 6,
    ("the", "cat"): 3,
    ("cat", "sat"): 5,
    ("sat", "on"): 8,
    ("on", "the"): 8,
    ("the", "mat"): 7,
    ("mat", "</s>"): 7,
    ("the", "rug"): 5,
    ("rug", "</s>"): 5,
    ("sat", "near"): 2,
    ("near", "the"): 2,
    ("the", "dog"): 3,
    ("dog", "sat"): 5,
    ("<s>", "a"): 4,
    ("a", "cat"): 2,
    ("a", "dog"): 2,
    ("<s>", "birds"): 2,
    ("birds", "fly"): 2,
    ("fly", "over"): 2,
    ("over", "the"): 2,
}
```

### Trigram counts

```python
{
    ("<s>", "the", "cat"): 3,
    ("the", "cat", "sat"): 3,
    ("cat", "sat", "on"): 4,
    ("sat", "on", "the"): 8,
    ("on", "the", "mat"): 4,
    ("the", "mat", "</s>"): 7,
    ("on", "the", "rug"): 4,
    ("the", "rug", "</s>"): 5,
    ("cat", "sat", "near"): 1,
    ("sat", "near", "the"): 2,
    ("near", "the", "mat"): 2,
    ("<s>", "the", "dog"): 3,
    ("the", "dog", "sat"): 3,
    ("dog", "sat", "on"): 4,
    ("dog", "sat", "near"): 1,
    ("<s>", "a", "cat"): 2,
    ("a", "cat", "sat"): 2,
    ("<s>", "a", "dog"): 2,
    ("a", "dog", "sat"): 2,
    ("<s>", "birds", "fly"): 2,
    ("birds", "fly", "over"): 2,
    ("fly", "over", "the"): 2,
    ("over", "the", "mat"): 1,
    ("over", "the", "rug"): 1,
}
```

Summary:

```text
distinct unigrams = 14
distinct bigrams  = 20
distinct trigrams = 24
```

## 4. Predecessor sets

```python
{
    "the": {"<s>", "on", "near", "over"},
    "cat": {"the", "a"},
    "sat": {"cat", "dog"},
    "on": {"sat"},
    "mat": {"the"},
    "</s>": {"mat", "rug"},
    "rug": {"the"},
    "near": {"sat"},
    "dog": {"the", "a"},
    "a": {"<s>"},
    "birds": {"<s>"},
    "fly": {"birds"},
    "over": {"fly"},
}
```

The continuation denominator is:

```text
continuation_denominator = 20
```

## 5. Continuation probabilities

The formula is:

```text
P_KN(word) = number of distinct predecessors of word / 20
```

```text
word       predecessor count   probability
-------------------------------------------
</s>       2                    2/20 = 0.10
a          1                    1/20 = 0.05
birds      1                    1/20 = 0.05
cat        2                    2/20 = 0.10
dog        2                    2/20 = 0.10
fly        1                    1/20 = 0.05
mat        1                    1/20 = 0.05
near       1                    1/20 = 0.05
on         1                    1/20 = 0.05
over       1                    1/20 = 0.05
rug        1                    1/20 = 0.05
sat        2                    2/20 = 0.10
the        4                    4/20 = 0.20
-------------------------------------------
sum        20                   1.00
```

## 6. Discount values

```python
discounts_by_order = {
    2: (0.5, 1.625, 2.0),
    3: (
        0.1818181818...,  # D1
        1.7575757576...,  # D2
        2.2727272727...,  # D3+
    ),
}
```

The discount selected for an observed count is:

```text
count 1       -> D1
count 2       -> D2
count 3 or up -> D3+
count 0       -> 0
```

Frequency-of-frequencies used:

```text
bigrams:
n1 = 0, n2 = 8, n3 = 2, n4 = 1

trigrams:
n1 = 4, n2 = 9, n3 = 4, n4 = 4
```

## 7. Important lambda values

Lambda is calculated on demand for a history. It is the removed discounted mass divided by the history count.

```text
history                 lambda
--------------------------------
("<s>",)                0.468750
("a",)                  0.812500
("birds",)              0.812500
("cat",)                0.400000
("dog",)                0.400000
("fly",)                0.812500
("mat",)                0.285714
("near",)               0.812500
("on",)                 0.250000
("over",)               0.812500
("rug",)                0.400000
("sat",)                0.362500
("the",)                0.444444
("<s>", "a")            0.878788
("<s>", "birds")        0.878788
("<s>", "the")          0.757576
("a", "cat")            0.878788
("a", "dog")            0.878788
("birds", "fly")        0.878788
("cat", "sat")           0.490909
("dog", "sat")           0.490909
("fly", "over")          0.878788
("mat", "</s>")          0.000000
("near", "the")          0.878788
("on", "the")            0.568182
("over", "the")          0.181818
("rug", "</s>")          0.000000
("sat", "near")          0.878788
("sat", "on")            0.284091
("the", "cat")           0.757576
("the", "dog")           0.757576
("the", "mat")           0.324675
("the", "rug")           0.454545
```

For an unseen history, lambda is 1.0 because the model uses only the lower-order distribution.

## 8. Example final probabilities

```text
P(sat)                 = 1/10 = 0.1000000000
P(sat | cat)           = 16/25 = 0.6400000000
P(sat | the cat)       = 8/11 = 0.7272727273
P(fly | cat)           = 1/50 = 0.0200000000
P(on | the cat)        = 1/66 = 0.0151515152
P(sat | the birds)     = 13/160 = 0.0812500000
```

## 9. Verification results

```text
normalization_total(())                 = 1.0000000000  PASS
normalization_total(("the",))           = 1.0000000000  PASS
normalization_total(("cat",))           = 1.0000000000  PASS
normalization_total(("the", "cat"))     = 1.0000000000  PASS
```

This completes the end-to-end toy-corpus MKN reference. Future hand calculations can use this section without searching through the earlier steps.
