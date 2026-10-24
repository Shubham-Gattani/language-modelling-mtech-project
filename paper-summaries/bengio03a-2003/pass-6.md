# Pass 6: Deep-dive into Section 2 and implement the core model

Pass 6 focuses on the mathematical heart of the paper: **Section 2, “A Neural Model.”** We will read the formulas carefully and implement a tiny version from scratch.

The goal is not to reproduce the paper’s full large-scale experiment. The goal is to make the important quantities concrete:

- learned distributed word representations;
- shared parameters across context positions;
- a neural network for combining context words;
- a probability distribution over the next word; and
- training with negative log-likelihood and gradient descent.

## Overall workflow

```text
Read Section 2 for the complete flow
        ↓
Read each equation and identify its shapes
        ↓
Reconstruct one forward pass by hand
        ↓
Implement a tiny model in NumPy
        ↓
Implement the loss and parameter updates
        ↓
Train on a tiny corpus until it overfits
        ↓
Compare with a toy n-gram model
        ↓
Reconstruct the paper without looking
```

## Step 1: Read Section 2 for structure

Read the section once without trying to memorize every formula. Identify this flow:

```text
context word IDs → word vectors → combined context representation
→ hidden layer → scores for vocabulary words → probabilities
→ probability of the observed word → loss
```

Answer only these questions on the first reading:

1. What does the model receive as input?
2. What does it predict?
3. Which quantities are learned?
4. Where does probability enter the model?

## Step 2: Read each equation by function

For every equation, write:

```text
Plain English: What is it calculating?
Symbols: What does every symbol mean?
Shapes: What is the size of every input and output?
Purpose: Why is this calculation needed?
```

Organize the equations into this sequence.

### 2.1 Word representation

Understand the lookup operation:

```text
word ID → learned vector C(word ID)
```

For the toy implementation, use vocabulary `{the, cat, dog, sat, <UNK>}` and 3 features per word vector. The vector table `C` is learned jointly with the language model.

### 2.2 Context combination

Use two context words. Retrieve their vectors and concatenate them:

```text
C(w₁), C(w₂) → x = [C(w₁); C(w₂)]
```

With 3-dimensional vectors, `x` has 6 values. The same word-vector table is reused at every context position, while concatenation preserves positional order.

### 2.3 Hidden layer

Calculate the hidden representation:

```text
h = tanh(d + Hx)
```

Use a hidden layer of 4 units and track the shapes:

```text
x: 6 values
H: 4 × 6 matrix
d: 4 values
h: 4 values
```

### 2.4 Output scores and softmax

Calculate one score for every vocabulary word:

```text
y = b + Wh
```

Then convert the scores into probabilities:

```text
P(word | context) = softmax(y)
```

The output must contain 5 probabilities, one per vocabulary item, and they must sum to 1.

### 2.5 Loss

For an example whose actual next word is `w*`, select its probability:

```text
loss = -log P(w* | context)
```

Distinguish the complete output distribution from the single probability used by the loss.

## Step 3: Reconstruct one forward pass by hand

Use the example:

```text
context: the cat
target: sat
```

Follow this sequence:

1. Look up `C(the)` and `C(cat)`.
2. Concatenate them into `x`.
3. Calculate `h`.
4. Calculate one score for each vocabulary word.
5. Apply softmax.
6. Select `P(sat | the cat)`.
7. Calculate `-log P(sat | the cat)`.

The purpose is to make the code a direct translation of the mathematics.

## Step 4: Implement the forward pass first

Create a small NumPy implementation with functions for lookup, concatenation, the hidden layer, output scores, softmax, and loss. Do not train yet.

Check that every array has the expected shape, softmax sums to approximately 1, and the loss is finite and positive.

## Step 5: Implement learning

The trainable parameters are:

```text
C: word-vector table
H: hidden-layer weights
d: hidden-layer bias
W: output weights
b: output bias
```

For each example:

1. Run the forward pass.
2. Calculate negative log-likelihood.
3. Backpropagate through softmax and the output layer.
4. Backpropagate through `tanh` and the hidden layer.
5. Accumulate gradients for the context word vectors.
6. Update all parameters with gradient descent.

The key point is that word vectors are updated by the same language-model loss as the rest of the network. They are not fixed features supplied from outside.

## Step 6: Train on a tiny corpus

Use:

```text
the cat sat
the dog sat
```

Create these examples:

```text
context       target
--------------------
the cat       sat
the dog       sat
```

Train until the model overfits. Then verify that both `P(sat | the cat)` and `P(sat | the dog)` are high. Overfitting this tiny corpus is a debugging test: if the model cannot memorize two simple examples, some implementation detail is wrong.

## Step 7: Test the representation-based idea

Inspect `C(cat)` and `C(dog)`. Their exact values are not important in such a tiny experiment. The important idea is that the same vector table and the same network are reused for both contexts.

The model does not contain a separate manually written rule for each sequence. It learns reusable parameters that can affect both `the cat → sat` and `the dog → sat`.

## Step 8: Add unknown-word handling

Map words outside the fixed vocabulary to `<UNK>`:

```text
the bird → the <UNK>
```

The model then looks up `C(the)` and `C(<UNK>)`. This demonstrates the limitation that an unknown word can be processed, but its individual identity is lost.

## Step 9: Compare with a toy n-gram model

Only after the neural implementation works, compare it with a tiny count-based model. Ask:

1. What does each model store?
2. How does each model handle an unseen context?
3. Where can information be shared?
4. What computation is expensive in the neural model?

The central contrast is:

```text
n-gram model: exact counts → smoothing/backoff → probability
neural model: word vectors → shared network → probability
```

## Step 10: Reconstruct the paper without looking

Close the paper and explain:

1. What problem with n-gram models motivates the paper?
2. What is the model’s input and output?
3. How is a word converted into a vector?
4. How are context vectors combined?
5. How does softmax produce the next-word distribution?
6. Why is the loss the negative log-probability of the actual word?
7. Which parameters are updated during training?
8. Why can the model generalize across related words and contexts?
9. What did the experiments establish?

If you get stuck, return only to the relevant equation or subsection. Do not reread the entire paper.

## What we are deliberately not implementing yet

Do not begin with the full Brown-corpus experiment, the parallel implementation, performance optimizations, a large vocabulary, or exact reproduction of every perplexity number. Those add engineering complexity before the core model is understood.

## Final learning objective

By the end of Pass 6, you should be able to move fluently between:

```text
paper equation ↔ array operation ↔ probability ↔ loss ↔ parameter update
```

### Mental model

> Look up word vectors, combine the context, score every possible next word, normalize the scores into probabilities, penalize the probability of the observed word, and update both the network and the word vectors.
