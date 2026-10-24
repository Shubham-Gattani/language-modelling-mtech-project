

## Context

We are now implementing a tiny version of the model from Section 2 of Bengio et al. (2003).
The purpose is to connect the paper's equations to concrete array operations and probabilities.
This is a learning implementation, not an attempt to reproduce the paper's large-scale Brown-corpus experiment.

We will use this workflow:

```text
word IDs and vectors
→ concatenated context vector
→ hidden representation
→ output scores
→ softmax probabilities
→ target-word loss
→ gradients and parameter updates
```

### Fixed toy setup

- Vocabulary: `{the, cat, dog, sat, <UNK>}`
- Vocabulary size: `5`
- Context length: `2` words
- Word-vector size: `3`
- Hidden-layer size: `4`
- Training examples:

  ```text
  context       target
  the cat       sat
  the dog       sat
  ```

- We will first process the example `the cat → sat`.
- All vectors and matrices will be written as row/column arrays explicitly when needed.
- We will use fixed values during the hand-calculation phase so that our outputs are reproducible.

## Step 1: Create the vocabulary, IDs, and word-vector table `C`

### What to do

Assign these IDs exactly:

```text
the   → 0
cat   → 1
dog   → 2
sat   → 3
<UNK> → 4
```

Create the word-vector table `C`. Each row is the 3-dimensional vector for one word:

```text
C =
the   [0.10, 0.20, 0.30]
cat   [0.20, 0.10, 0.40]
dog   [0.30, 0.20, 0.10]
sat   [0.40, 0.30, 0.20]
<UNK> [0.00, 0.10, 0.00]
```

If you use NumPy, store `C` with shape `(5, 3)`.

### Important interpretation

`C` is the model's word-representation table. Looking up a word means selecting the row associated with its ID. At this stage, the values are initialized values; later, training will update them using the language-model loss.

### Checkpoint to report

Before continuing, verify and record only:

```text
ID(the), ID(cat), ID(dog), ID(sat), ID(<UNK>)
shape(C)
C(the)
C(cat)
C(dog)
```

For the first forward pass, we will use only `C(the)` and `C(cat)`. Do not calculate the hidden layer or probabilities yet.

### Values carried forward

```text
context word IDs: [0, 1]
target word ID: 3
C(the) = [0.10, 0.20, 0.30]
C(cat) = [0.20, 0.10, 0.40]
```

When Step 1 matches, say `done` and provide your checkpoint values. Then we will add Step 2: concatenate the two context vectors into `x`.

## Step 2: Concatenate the context vectors into `x`

### What to do

For the context `the cat`, look up the two rows of `C`:

```text
C(the) = [0.10, 0.20, 0.30]
C(cat) = [0.20, 0.10, 0.40]
```

Concatenate them in their original left-to-right order:

```text
x = [C(the); C(cat)]
```

Therefore:

```text
x = [0.10, 0.20, 0.30, 0.20, 0.10, 0.40]
```

### Why this matters

Concatenation combines the context words without losing their positions. The first three values describe the first context position (`the`), and the last three values describe the second context position (`cat`).

This is different from averaging: averaging would lose the distinction between `the cat` and `cat the`, while concatenation preserves the order.

### Checkpoint to report

Verify only:

```text
shape(x) = (6,)
x = [0.10, 0.20, 0.30, 0.20, 0.10, 0.40]
```

### Values carried forward

```text
context: the cat
x = [0.10, 0.20, 0.30, 0.20, 0.10, 0.40]
```

Do not calculate the hidden representation yet. When this matches, say `done`; then we will initialize `H` and `d` and calculate the hidden representation.

## Step 3: Calculate the hidden representation `h`

### What to do

Use the hidden-layer equation:

```text
z = d + Hx
h = tanh(z)
```

Here, `z` is the hidden pre-activation and `h` is the hidden representation.

Use these fixed parameters:

```text
H =
[0.10, 0.00, 0.00, 0.10, 0.00, 0.00]
[0.00, 0.10, 0.00, 0.00, 0.10, 0.00]
[0.00, 0.00, 0.10, 0.00, 0.00, 0.10]
[0.10, 0.10, 0.10, 0.10, 0.10, 0.10]

d = [0.00, 0.00, 0.00, 0.00]
```

The shapes are:

```text
H: (4, 6)
d: (4,)
x: (6,)
z: (4,)
h: (4,)
```

### Calculate the pre-activation

Using the `x` from Step 2:

```text
z = d + Hx
  = [0.03, 0.03, 0.07, 0.11]
```

For example, the first component is:

```text
z₁ = 0.10(0.10) + 0.10(0.20) = 0.03
```

The fourth component is:

```text
z₄ = 0.10(0.10 + 0.20 + 0.30 + 0.20 + 0.10 + 0.40) = 0.11
```

### Apply `tanh`

Apply `tanh` component by component:

```text
h = tanh([0.03, 0.03, 0.07, 0.11])
  ≈ [0.029991, 0.029991, 0.069886, 0.109558]
```

### Why this matters

`H` transforms the concatenated context vector into a smaller hidden representation. `tanh` adds a nonlinear transformation, allowing the network to learn relationships that a purely linear transformation could not represent.

### Checkpoint to report

Verify only:

```text
shape(H) = (4, 6)
shape(d) = (4,)
z = [0.03, 0.03, 0.07, 0.11]
h ≈ [0.029991, 0.029991, 0.069886, 0.109558]
```

### Values carried forward

```text
x = [0.10, 0.20, 0.30, 0.20, 0.10, 0.40]
H: the fixed (4, 6) matrix above
d = [0.00, 0.00, 0.00, 0.00]
h ≈ [0.029991, 0.029991, 0.069886, 0.109558]
```

Do not calculate output scores or softmax probabilities yet. When this checkpoint matches, say `done`; then we will initialize the output parameters and calculate one score for each possible next word.

## Step 4: Calculate one output score for each possible next word

### What to do

Use the paper's output equation:

```text
y = b + W x + U h
```

The paper writes the hidden part as `U tanh(d + Hx)`. Since Step 3 already calculated `h = tanh(d + Hx)`, this is the same equation.

Each component of `y` is the score for one candidate next word. Use this vocabulary order:

```text
[the, cat, dog, sat, <UNK>]
```

Use these fixed output parameters:

> Clarification: The numerical values below are fixed illustrative values so that we can reproduce the hand calculation. In a real implementation, `U` and `W` are typically initialized with small random values; their shapes are determined by the dimensions of the model.

```text
b = [0.010, 0.000, -0.010, 0.020, 0.000]

W =
[0.10, 0.00, 0.00, 0.00, 0.00, 0.00]
[0.00, 0.10, 0.00, 0.00, 0.00, 0.00]
[0.00, 0.00, 0.10, 0.00, 0.00, 0.00]
[0.00, 0.00, 0.00, 0.10, 0.00, 0.00]
[0.00, 0.00, 0.00, 0.00, 0.10, 0.00]

U =
[0.10, 0.00, 0.00, 0.00]
[0.00, 0.10, 0.00, 0.00]
[0.00, 0.00, 0.10, 0.00]
[0.00, 0.00, 0.00, 0.10]
[-0.10, 0.00, 0.00, 0.00]
```

The shapes are:

```text
b: (5,)
W: (5, 6)
U: (5, 4)
y: (5,)
```

### Why does `W` have shape `(|V|, dim(x))`?

`W` maps the context vector `x` to one output score for every vocabulary word:

```text
W shape = (|V|, dim(x))
```

In this toy model:

```text
|V| = 5
dim(x) = 6
W shape = (5, 6)
```

The multiplication works as:

```text
(5, 6) × (6,) = (5,)
```

The resulting five values are the direct context contributions to the five candidate-word scores.

### Calculate the three parts

Using the `x` and `h` from the earlier checkpoints:

```text
W x = [0.010, 0.020, 0.030, 0.020, 0.010]
```

Using `h ≈ [0.029991, 0.029991, 0.069886, 0.109558]`:

```text
U h ≈ [0.002999, 0.002999, 0.006989, 0.010956, -0.002999]
```

Therefore:

```text
y = b + W x + U h
  ≈ [0.022999, 0.022999, 0.026989, 0.050956, 0.007001]
```

### Why this matters

The model now has one raw score for every possible next word. A larger score means that the current parameters prefer that word, but these scores are not probabilities yet. They do not need to sum to one.

The two learned contributions have different roles:

```text
W x: direct contribution from the context vector
U h: contribution from the nonlinear hidden representation
```

The output bias `b` is added once to the combined contributions:

```text
y = b + W x + U h
```

Thus, `b` is a separate output adjustment, not part of `U h`. The hidden layer has its own separate bias `d`.

### Checkpoint to report

Verify only:

```text
shape(W) = (5, 6)
shape(U) = (5, 4)
shape(b) = (5,)
W x = [0.010, 0.020, 0.030, 0.020, 0.010]
y ≈ [0.022999, 0.022999, 0.026989, 0.050956, 0.007001]
```

### Values carried forward

```text
candidate-word order: [the, cat, dog, sat, <UNK>]
y ≈ [0.022999, 0.022999, 0.026989, 0.050956, 0.007001]
```

Do not calculate the loss yet. When this checkpoint matches, say `done`; then we will apply softmax to turn these five scores into a probability distribution.

## Step 5: Convert scores into probabilities with softmax

### What to do

Apply softmax to the five output scores:

```text
P(word | the cat) = exp(y_word) / sum over all candidate words of exp(y_word)
```

For numerical stability, subtract the largest score before taking exponentials. This does not change the final probabilities:

```text
largest score = 0.050956

adjusted scores = y - 0.050956
                ≈ [-0.027957, -0.027957, -0.023967, 0.000000, -0.043955]
```

Then calculate:

```text
exp(adjusted scores)
≈ [0.972430, 0.972430, 0.976318, 1.000000, 0.956997]
```

The sum is approximately:

```text
normalizer ≈ 4.878175
```

### Resulting probability distribution

In the order `[the, cat, dog, sat, <UNK>]`:

```text
P(. | the cat)
≈ [0.199343, 0.199343, 0.200138, 0.204993, 0.196178]
```

The notation `P(. | the cat)` means the complete distribution over every possible next word after the context `the cat`.

### Why this matters

The output scores `y` were arbitrary real numbers. Softmax converts them into positive values that sum to one, so they can be interpreted as probabilities.

The target in this example is `sat`, so the probability used by the training loss will be:

```text
P(sat | the cat) ≈ 0.204993
```

At initialization, this probability is close to `1/5 = 0.20`, because the model has not learned much yet.

### Checkpoint to report

Verify only:

```text
sum of probabilities ≈ 1.000000
P(the | the cat) ≈ 0.199343
P(cat | the cat) ≈ 0.199343
P(dog | the cat) ≈ 0.200138
P(sat | the cat) ≈ 0.204993
P(<UNK> | the cat) ≈ 0.196178
```

### Values carried forward

```text
probabilities ≈ [0.199343, 0.199343, 0.200138, 0.204993, 0.196178]
target: sat
target probability: P(sat | the cat) ≈ 0.204993
```

Do not calculate gradients yet. When this checkpoint matches, say `done`; then we will calculate the negative log-likelihood for the observed target `sat`.

## Step 6: Calculate the negative log-likelihood loss

### What to do

The target word for the context `the cat` is `sat`. Use only the probability assigned to that observed target:

```text
loss = -log P(sat | the cat)
```

From Step 5:

```text
P(sat | the cat) ≈ 0.204993
```

Therefore:

```text
loss = -log(0.204993)
     ≈ 1.5848
```

Here, `log` means the natural logarithm.

### Why this matters

The model produced probabilities for all five candidate words, but this training example directly penalizes only the probability assigned to the correct target `sat`.

- If the model assigns a high probability to `sat`, the loss is small.
- If the model assigns a low probability to `sat`, the loss is large.

The other four probabilities still matter indirectly because all five probabilities were produced by the same softmax normalization.

### Notation for the next step

`y` is the vector of five output scores, not a loss distribution:

```text
y = [y_the, y_cat, y_dog, y_sat, y_<UNK>]
```

At the component level, we write:

```text
dL/dy_i
```

This means: how does the loss change when the particular score `y_i` changes?

At the vector level, we write:

```text
dL/dy
```

This means the complete gradient vector:

```text
dL/dy = [dL/dy_the, dL/dy_cat, dL/dy_dog, dL/dy_sat, dL/dy_<UNK>]
```

Thus, `dL/dy` is shorthand for collecting all the component derivatives. It does not mean that only the target word has a loss; every score affects the loss through softmax normalization.

### Checkpoint to report

```text
target: sat
target probability ≈ 0.204993
loss ≈ 1.5848
```

Do not update parameters yet. When this matches, say `done`; then we will calculate the output-layer gradient produced by this one training example.

## Consolidated checkpoint values: Steps 1–6

These are the important values from the first forward pass for `the cat → sat`.

### Vocabulary and word vectors

```text
vocabulary order: [the, cat, dog, sat, <UNK>]

C(the) = [0.10, 0.20, 0.30]
C(cat) = [0.20, 0.10, 0.40]
C(dog) = [0.30, 0.20, 0.10]
C(sat) = [0.40, 0.30, 0.20]
C(<UNK>) = [0.00, 0.10, 0.00]
```

### Context representation

```text
x = [0.10, 0.20, 0.30, 0.20, 0.10, 0.40]
```

### Hidden layer

```text
H =
[0.10, 0.00, 0.00, 0.10, 0.00, 0.00]
[0.00, 0.10, 0.00, 0.00, 0.10, 0.00]
[0.00, 0.00, 0.10, 0.00, 0.00, 0.10]
[0.10, 0.10, 0.10, 0.10, 0.10, 0.10]

d = [0.00, 0.00, 0.00, 0.00]
z = d + Hx = [0.03, 0.03, 0.07, 0.11]
h = tanh(z) ≈ [0.029991, 0.029991, 0.069886, 0.109558]
```

### Output layer

```text
b = [0.010, 0.000, -0.010, 0.020, 0.000]

W x = [0.010, 0.020, 0.030, 0.020, 0.010]

U h ≈ [0.002999, 0.002999, 0.006989, 0.010956, -0.002999]

y = b + W x + U h
  ≈ [0.022999, 0.022999, 0.026989, 0.050956, 0.007001]
```

### Probability distribution and target

```text
P(. | the cat)
≈ [0.199343, 0.199343, 0.200138, 0.204993, 0.196178]

target word: sat
P(sat | the cat) ≈ 0.204993

loss = -log P(sat | the cat) ≈ 1.5848
```

The next step will use the complete probability distribution and the target `sat` to calculate the output-layer gradient.

## Step 7: Calculate the output error signal

### What to do

For softmax followed by negative log-likelihood, the gradient with respect to the output scores is especially simple:

```text
delta_y = d(loss) / d(y)
        = probabilities - one_hot(target)
```

The target is `sat`, whose position is fourth in the order:

```text
[the, cat, dog, sat, <UNK>]
```

Therefore:

```text
one_hot(sat) = [0, 0, 0, 1, 0]
```

Using the probabilities from Step 5:

```text
probabilities ≈ [0.199343, 0.199343, 0.200138, 0.204993, 0.196178]
```

Calculate:

```text
delta_y ≈ [0.199343, 0.199343, 0.200138, -0.795007, 0.196178]
```

### Interpretation

The gradient says how each score should change to reduce the loss:

- For incorrect words, the entries are positive, meaning the loss increases as those scores move upward. Since gradient descent moves in the opposite direction of the gradient, it pulls those scores downward to reduce the loss.
- For the correct word `sat`, the entry is negative. Gradient descent will increase its score.

The entries sum to approximately zero because increasing one softmax score affects the whole normalized distribution.

### Checkpoint to report

```text
one_hot(sat) = [0, 0, 0, 1, 0]
delta_y ≈ [0.199343, 0.199343, 0.200138, -0.795007, 0.196178]
sum(delta_y) ≈ 0
```

### Why does `sum(delta_y) ≈ 0`?

For softmax followed by cross-entropy, this sum should always be zero, apart from rounding error:

```text
sum(delta_y)
= sum(probabilities) - sum(one_hot(target))
= 1 - 1
= 0
```

The approximation sign appears because we rounded the probabilities. This also reflects an important softmax property: adding the same constant to every output score does not change the probabilities, so the loss has no gradient in that all-scores-equal direction.

This property is specific to softmax with a normalized target distribution; it does not hold for every loss function or output function.

Do not update any parameters yet. When this matches, say `done`; then we will use `delta_y` to calculate gradients for `b`, `W`, and `U`.

## Step 8: Calculate gradients for the output parameters (b, W, U)

### What to do

For the output equation:

```text
y = b + W x + U h
```

use these gradients:

```text
gradient_b = delta_y
gradient_W = outer_product(delta_y, x)
gradient_U = outer_product(delta_y, h)
```

The outer product means that each output-word error is multiplied by every component of the corresponding input vector.

### Why calculate `dL/dy` first?

The loss depends on the predicted probabilities, and the probabilities are produced from the output scores `y`. Therefore, the first gradient needed is:

```text
dL/dy
```

This is the error signal at the output layer. Once it is known, the chain rule gives the gradients for the parameters connected directly to `y`:

```text
dL/db = dL/dy
dL/dW = (dL/dy) xᵀ
dL/dU = (dL/dy) hᵀ
```

Then the same error signal can be propagated into the hidden layer:

```text
dL/dh = Uᵀ(dL/dy)
```

So `dL/dy` is not the only gradient. It is the starting gradient because `y` is the interface between the loss and the rest of the network.

### FAQ: Can we skip the gradient of an intermediate vector?

#### Question

If a sequence of operations produces a vector `v1`, and an operation transforms `v1` into `v2`, can we calculate `dL/dv1` directly instead of calculating `dL/dv2` first?

#### Answer

Not in general. If:

```text
v1 → operation f → v2 → loss L
```

then the chain rule is:

```text
dL/dv1 = dL/dv2 × dv2/dv1
```

The intermediate gradient `dL/dv2` still exists conceptually. We can calculate `dL/dv1` directly only when the derivatives have already been combined into a simpler expression.

That is what happens for softmax followed by cross-entropy:

```text
dL/dy
  = dL/d(probabilities) × d(probabilities)/dy
  = probabilities - one_hot(target)
```

So we are not ignoring the softmax derivative. We are using the simplified result of applying the chain rule through softmax and the loss together. This gradient can then be propagated to the parameters that produced `y`.

### Important results

```text
gradient_b
≈ [0.199343, 0.199343, 0.200138, -0.795007, 0.196178]
```

Using `x = [0.10, 0.20, 0.30, 0.20, 0.10, 0.40]`:

```text
gradient_W[sat, :] ≈ [-0.079501, -0.159001, -0.238502, -0.159001, -0.079501, -0.318003]
```

The full `gradient_W` is:

```text
gradient_W ≈
[ 0.019934,  0.039869,  0.059803,  0.039869,  0.019934,  0.079737]
[ 0.019934,  0.039869,  0.059803,  0.039869,  0.019934,  0.079737]
[ 0.020014,  0.040028,  0.060041,  0.040028,  0.020014,  0.080055]
[-0.079501, -0.159001, -0.238502, -0.159001, -0.079501, -0.318003]
[ 0.019618,  0.039236,  0.058853,  0.039236,  0.019618,  0.078471]
```

The rows are ordered as `[the, cat, dog, sat, <UNK>]`. The full matrix is used when updating `W`; the `sat` row above is only a quick sanity check.

Using `h ≈ [0.029991, 0.029991, 0.069886, 0.109558]`:

```text
gradient_U[sat, :] ≈ [-0.023843, -0.023843, -0.055560, -0.087099]
```

The full `gradient_U` is:

```text
gradient_U ≈
[ 0.005978,  0.005978,  0.013931,  0.021840]
[ 0.005978,  0.005978,  0.013931,  0.021840]
[ 0.006002,  0.006002,  0.013987,  0.021927]
[-0.023843, -0.023843, -0.055560, -0.087099]
[ 0.005884,  0.005884,  0.013710,  0.021493]
```

The rows are ordered as `[the, cat, dog, sat, <UNK>]`. The full matrix is used when updating `U`; the `sat` row above is only a quick sanity check.

### How do we obtain a particular word's row?

`delta_y` has one entry per candidate output word, and each row of `U` also corresponds to one candidate word. The outer product is:

```text
gradient_U = delta_y[:, None] × h[None, :]
```

Therefore, for any word `w`:

```text
gradient_U[w, :] = delta_y[w] × hᵀ
```

For `sat`:

```text
gradient_U[sat, :]
 = -0.795007 × [0.029991, 0.029991, 0.069886, 0.109558]
 ≈ [-0.023843, -0.023843, -0.055560, -0.087099]
```

The same logic gives:

```text
gradient_W[w, :] = delta_y[w] × xᵀ
```

The word selects a row because each output row represents one candidate vocabulary word; the columns represent hidden features for `U` or context features for `W`.

The complete gradient shapes are:

```text
gradient_b: (5,)
gradient_W: (5, 6)
gradient_U: (5, 4)
```

### Why this matters

The `sat` row has negative gradients. During gradient descent, subtracting these negative values increases the parameters that raise the `sat` score. Incorrect-word rows have positive gradients and are pushed in the opposite direction.

At this step, only `b`, `W`, and `U` receive gradients. The error has not yet been propagated into the hidden layer or the word vectors.

### Checkpoint to report

Verify the formulas, shapes, and the two `sat` rows above. Do not update parameters yet. When this matches, say `done`; then we will backpropagate through the hidden layer.

## Step 9: Backpropagate through `U` and `tanh`

### What to do

The hidden representation `h` affects the output through `U h`. Therefore, first propagate the output error into `h`:

```text
dL/dh = Uᵀ delta_y
```

Then use the derivative of `tanh`:

```text
d(tanh(z))/dz = 1 - tanh(z)^2 = 1 - h^2
```

So the error entering `z` is:

```text
delta_z = dL/dz
        = (dL/dh) ⊙ (1 - h^2)
```

Here, `⊙` means element-by-element multiplication.

### Calculate `dL/dh`

Using the fixed `U` and `delta_y`:

```text
dL/dh ≈ [0.000317, 0.019934, 0.020014, -0.079501]
```

### Calculate `delta_z`

Using `h ≈ [0.029991, 0.029991, 0.069886, 0.109558]`:

```text
1 - h^2 ≈ [0.999101, 0.999101, 0.995116, 0.987997]
```

Therefore:

```text
delta_z = dL/dz
        ≈ [0.000316, 0.019916, 0.019916, -0.078546]
```

### Why this matters

`delta_z` is the error signal entering the affine hidden-layer operation:

```text
z = d + Hx
```

It will be used next to calculate gradients for the hidden-layer weights `H`, the hidden bias `d`, and the context vector `x`.

### Checkpoint to report

```text
dL/dh ≈ [0.000317, 0.019934, 0.020014, -0.079501]
delta_z ≈ [0.000316, 0.019916, 0.019916, -0.078546]
```

Do not update parameters yet. When this matches, say `done`; then we will calculate `dL/dH`, `dL/dd`, and the gradient flowing back to the two context word vectors.
