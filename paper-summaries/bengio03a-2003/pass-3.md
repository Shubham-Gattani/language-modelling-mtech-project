# Pass 3: Read equations by function, not by order

The paper’s equations answer four practical questions:

1. How is a word represented?
2. How are several context words combined?
3. How is the next-word probability calculated?
4. How are the parameters learned?

We will use the small example:

```text
Context: the cat
Target:  sat
```

The model receives the context words and assigns a probability to every possible next word.

## 1. How is a word represented?

### Equation

$$
C(i) = \mathbf{c}_i
$$

### Plain English

`C(i)` looks up the vector associated with word `i`.

The model does not use the word’s spelling directly. Each vocabulary word has a learned continuous feature vector.

### Symbols

- `i`: the index of a word in the vocabulary.
- `C`: the word-feature lookup table.
- `C(i)` or $\mathbf{c}_i$: the vector for word `i`.
- If the vocabulary has `V` words and each vector has `m` features, then `C` is a `V × m` matrix.

### Reason

The vector gives the model a compact, reusable representation of a word. Similar words can receive similar vectors, which helps the model generalize to contexts it has not seen exactly during training.

### Tiny example

Suppose the vocabulary is `{the, cat, sat}` and each word has a 3-dimensional vector:

```text
C(the) = [ 0.2,  0.1, -0.3]
C(cat) = [ 0.7, -0.1,  0.4]
C(sat) = [-0.2,  0.8,  0.1]
```

For the context `the cat`, the model retrieves `C(the)` and `C(cat)`. The vector for `sat` is not used as input; it is the word whose probability is being evaluated as the target.

## 2. How are several context words combined?

### Equation

For a context of `n - 1` words, the paper concatenates their feature vectors:

$$
x = [C(i_1); C(i_2); \ldots; C(i_{n-1})]
$$

### Plain English

The model puts the context-word vectors next to one another to create one larger input vector.

This preserves the position of each word. The vector for the first context position is different from the vector for the second position, even when the same word appears in both positions.

### Symbols

- $i_1, ..., i_{n-1}$: the indices of the context words.
- `;`: concatenation, meaning “place these vectors side by side.”
- `x`: the complete input vector given to the neural network.
- `n`: the n-gram order. An n-gram model uses `n - 1` words to predict the `n`th word.
- `m`: the dimensionality of one word vector.
- The size of `x` is `(n - 1)m`.

### Reason

The network needs one representation containing all context words before it can use their joint information to predict the next word.

### Tiny example

For a trigram-style neural model, `n = 3`, so there are two context words. If each word vector has `m = 3` dimensions:

```text
C(the) = [0.2,  0.1, -0.3]
C(cat) = [0.7, -0.1,  0.4]

x = [0.2, 0.1, -0.3, 0.7, -0.1, 0.4]
```

Here `x` has `(3 - 1) × 3 = 6` values.

## 3. How is the next-word probability calculated?

This happens in two steps: the hidden layer creates a context representation, and the output layer converts scores into probabilities.

### 3.1 Hidden-layer representation

### Equation

$$
h = \tanh(d + Hx)
$$

### Plain English

The hidden layer transforms the concatenated context vector `x` into a new representation `h` that captures useful interactions among the context words.

### Symbols

- `x`: concatenated context-word vectors.
- `H`: hidden-layer weight matrix.
- `d`: hidden-layer bias vector.
- `h`: hidden-layer activation vector.
- `tanh`: a nonlinear activation function.

### Reason

Without the hidden layer, the model would mainly combine inputs linearly. The nonlinearity allows it to learn more complex patterns, such as the fact that `the cat` strongly supports nouns or that particular word combinations are especially predictive.

### Tiny example

For one hidden unit, suppose:

```text
d = 0.1
Hx = 0.8
h = tanh(0.1 + 0.8)
  = tanh(0.9)
  ≈ 0.716
```

With many hidden units, the same calculation happens once per hidden unit.

### 3.2 Output scores

### Equation

$$
y = b + Wh
$$

### Plain English

The output layer gives every vocabulary word a score. A higher score means the network currently considers that word more plausible after the context.

### Symbols

- `h`: hidden representation of the context.
- `W`: hidden-to-output weight matrix.
- `b`: output bias vector.
- `y`: vector of raw scores, also called logits.
- $y_j$: the score assigned to vocabulary word `j`.

### Reason

The model must compare all possible next words. The output layer produces one score for each vocabulary item.

### Tiny example

For the vocabulary `{sat, slept, ran}`, suppose the output scores are:

```text
y = [2.0, 1.0, 0.0]
```

These are not probabilities yet. They can be negative, greater than 1, and do not need to sum to 1.

### 3.3 Softmax probability distribution

### Equation

$$
P(w=j \mid \text{context})
=
\frac{\exp(y_j)}{\sum_{k=1}^{V} \exp(y_k)}
$$

### Plain English

Softmax converts all output scores into a probability distribution over the entire vocabulary.

### Symbols

- `w = j`: candidate next word `j`.
- `y_j`: score for candidate word `j`.
- `V`: vocabulary size.
- `k`: index used to sum over every vocabulary word.
- `exp`: the exponential function.
- The denominator normalizes the scores so that all probabilities sum to 1.

### Reason

The model must produce a valid conditional distribution, not just one preferred word. This lets us calculate the probability of the actual next word and train the model with that probability.

### Tiny example

Using scores `[2.0, 1.0, 0.0]`:

```text
exp(2.0) ≈ 7.39
exp(1.0) ≈ 2.72
exp(0.0) = 1.00

total = 7.39 + 2.72 + 1.00 = 11.11

P(sat)   = 7.39 / 11.11 ≈ 0.665
P(slept) = 2.72 / 11.11 ≈ 0.245
P(ran)   = 1.00 / 11.11 ≈ 0.090
```

The probabilities add to approximately 1. The model predicts a distribution; it does not simply output `sat`.

## 4. How are the parameters learned?

### 4.1 Probability of a whole sentence

### Equation

$$
P(w_1, w_2, \ldots, w_T)
=
\prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})
$$

### Plain English

The probability of a sentence is calculated as a product of next-word probabilities. At each position, the model predicts the current word from the preceding context.

In this paper, the practical context is limited to the previous `n - 1` words, so the model uses an n-gram-like context.

### Symbols

- `w_1, ..., w_T`: the words in a training sequence.
- `T`: sequence length.
- `t`: current position.
- `P(w_t | context)`: probability assigned to the actual word at position `t`.
- `∏`: multiply the probabilities across positions.

### Reason

This connects local next-word predictions to the probability of a complete word sequence.

### Tiny example

For the sentence `the cat sat`, suppose:

```text
P(the | start)      = 0.50
P(cat | the)        = 0.40
P(sat | the cat)    = 0.70
```

Then:

```text
P(the cat sat) = 0.50 × 0.40 × 0.70 = 0.14
```

### 4.2 Training objective: maximize the probability of observed words

### Equation

The paper trains the parameters `θ` by maximizing the log-likelihood:

$$
\max_{\theta}
\sum_{t=1}^{T}
\log P_{\theta}(w_t \mid \text{context}_t)
$$

The equivalent loss minimized in practice is the negative log-likelihood:

$$
L(\theta)
=
-\sum_{t=1}^{T}
\log P_{\theta}(w_t \mid \text{context}_t)
$$

### Plain English

For every training example, look at the probability assigned to the actual next word. Penalize the model when that probability is small. Adjust the parameters so that the probability of the observed word increases.

### Symbols

- `θ`: all trainable parameters, including word vectors, hidden-layer weights, output weights, and biases.
- `w_t`: the actual word observed in the training text.
- `context_t`: the preceding context used to predict `w_t`.
- `log`: logarithm, usually the natural logarithm.
- `L`: total training loss.

### Reason

The model should assign high probability to word sequences that occur in real text. The logarithm turns a product of many probabilities into a sum, which is easier to optimize and numerically more stable.

### Tiny example

For one example, if the model assigns:

```text
P(actual word | context) = 0.70
loss = -log(0.70) ≈ 0.36
```

If it assigns only `0.10`:

```text
loss = -log(0.10) ≈ 2.30
```

Thus, confident correct predictions have low loss, while assigning low probability to the observed word produces a large penalty.

### 4.3 Parameter update

### Equation

The parameters are updated in the direction that reduces the loss:

$$
\theta \leftarrow \theta - \eta \, \nabla_{\theta} L(\theta)
$$

### Plain English

Calculate how each parameter affects the loss, then change each parameter slightly in the direction that makes the loss smaller.

### Symbols

- $θ$: current parameters.
- $L(θ)$: loss produced by those parameters.
- $∇_θ L(θ)$: gradient of the loss with respect to every parameter.
- $η$: learning rate, controlling the step size.
- $←$: replace the old parameter values with the updated values.

### Reason

The word vectors and network weights are not manually designed. Backpropagation computes the gradients, and gradient descent repeatedly improves all parameters together.

### Tiny example

Suppose one parameter has:

```text
old parameter θ = 0.50
gradient        = 0.20
learning rate η = 0.10

new θ = 0.50 - (0.10 × 0.20)
      = 0.48
```

The same process updates every word vector and every network weight. This is why the paper learns the distributed word representations and the prediction function jointly.

## The complete equation flow

For one training example, the equations should be read as one pipeline:

```text
word indices
    ↓
lookup word vectors C(i)
    ↓
concatenate context vectors into x
    ↓
hidden representation h = tanh(d + Hx)
    ↓
word scores y = b + Wh
    ↓
softmax probabilities P(next word | context)
    ↓
select probability of the actual next word
    ↓
negative log-likelihood loss
    ↓
backpropagation and gradient update
```

## What to remember after Pass 3

1. `C(i)` converts a symbolic word index into a learned continuous vector.
2. Context vectors are concatenated and passed through the neural network.
3. The output layer produces one score per vocabulary word.
4. Softmax converts those scores into a probability distribution.
5. The loss is the negative log of the probability assigned to the actual next word.
6. Backpropagation updates both the network parameters and the word vectors.

### Mental model

The model is a probability-producing pipeline: **words → vectors → context representation → scores → probabilities → loss → parameter updates**.
