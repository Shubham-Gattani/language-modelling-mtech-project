# Pass 4: Compare the neural model directly with an n-gram model

Both models answer the same question:

> Given the previous words, what is the probability of the next word?

Their main difference is how they represent words and how they share information between contexts.

## Direct comparison

| Question | Traditional n-gram model | Bengio neural model |
|---|---|---|
| Representation of a word | A discrete symbol or vocabulary ID | A learned continuous vector of features |
| Representation of a context | A discrete word sequence, such as `the cat` | The vectors for `the` and `cat`, combined and processed by a neural network |
| How probability is estimated | From n-gram counts and smoothing | From neural-network scores converted by softmax |
| How similar words are treated | Different words usually have separate counts | Words can have similar learned vectors and therefore support similar predictions |
| Generalization | Mainly through shorter contexts and smoothing | Through shared word features and shared network parameters |
| Handling an unseen full context | Back off or interpolate to lower-order n-grams | Still compute a prediction from the word vectors, even if that exact sequence was unseen |
| Parameter sharing | Counts are tied to particular word sequences | The same word vectors and network function are reused across many contexts |
| Main strength | Simple, interpretable, and relatively cheap to train | Better representation-based generalization |
| Main challenge | Data sparsity and rapidly growing count tables | Many computations, optimization difficulty, and large output layers |

## 1. Traditional n-gram model: memorize counts

Suppose the model wants to estimate:

```text
P(sat | the cat)
```

An n-gram model looks for evidence about the exact sequence `the cat sat` and the history `the cat`. A smoothed model may calculate a discounted higher-order probability and combine it with a lower-order probability such as:

```text
P(sat | the cat)
    = higher-order evidence
    + lower-order/backoff evidence
```

If `the cat sat` was never observed, the model does not understand the meaning of `the`, `cat`, or `sat`. It only knows that this exact count is missing, so it must use shorter-context evidence.

For example, it may use information about:

```text
P(sat | cat)
P(sat)
```

Smoothing prevents the probability from becoming exactly zero, but it does not automatically know that `cat` and `dog` are semantically related.

## 2. Bengio’s neural model: learn reusable representations

The neural model also predicts a word such as `sat` from a context such as `the cat`, but it first converts the context words into vectors:

```text
the → C(the)
cat → C(cat)
```

It then passes those vectors through the same learned function:

```text
C(the), C(cat)
    ↓
neural network
    ↓
scores for every possible next word
    ↓
softmax probabilities
```

The model can learn vectors in which words with similar grammatical or semantic behavior have related representations. Consequently, evidence learned from one context can influence predictions in another context.

For example, if the model learns that `cat` and `dog` behave similarly, then information learned from contexts containing `the cat` may help it make a reasonable prediction for an unseen context such as `the dog`. This is representation-based generalization; it is not simply copying an exact n-gram count.

## 3. The most important contrast: where generalization happens

### n-gram generalization

An n-gram model generalizes mainly by reducing the amount of context:

```text
P(word | w₁, w₂)
    → P(word | w₂)
    → P(word)
```

It asks: “If I do not have enough evidence for this exact history, what can I learn from a shorter history?”

### Neural generalization

The neural model generalizes by sharing learned features:

```text
word identity
    → word vector
    → shared neural function
    → prediction
```

It asks: “What other words or contexts have similar learned features, and how can their evidence help here?”

The Bengio model can still have a fixed context length, such as the previous five words. The key difference is not that it automatically remembers unlimited history. The key difference is that it represents the words and contexts continuously, allowing information to be shared across different word sequences.

## 4. Why the neural model can reduce sparsity

Consider these two contexts:

```text
the cat sat
the dog sat
```

An n-gram model stores them as different symbolic sequences:

```text
(`the`, `cat`, `sat`)
(`the`, `dog`, `sat`)
```

The count for one sequence does not directly become evidence for the other sequence.

The neural model instead uses:

```text
C(the), C(cat)
C(the), C(dog)
```

If `C(cat)` and `C(dog)` become similar during training, then the network can respond similarly to these two contexts. A single learned feature can therefore be useful in many contexts.

This is the central advantage of distributed representations: information is not stored in only one exact word-sequence entry. It is distributed across vector dimensions and shared network parameters.

## 5. The trade-off

The neural model does not make language modeling free or automatically perfect.

- The n-gram model is usually easier to interpret: counts, discounts, and backoff probabilities can be inspected directly.
- The neural model requires expensive forward passes, gradient calculations, and updates to many parameters.
- The n-gram model suffers from sparse or unseen sequences.
- The neural model can generalize better, but it must learn useful representations from data and can be difficult to optimize.

The paper’s contribution is therefore not simply “replace counts with a neural network.” It is:

1. represent each word with a learned feature vector;
2. use a shared neural function to combine the context vectors; and
3. learn the representations and the probability model jointly.

## The key conceptual contrast

> An n-gram model generalizes mainly through overlapping counts, smoothing, and shorter contexts. Bengio’s model generalizes through learned word representations and shared neural parameters.

### Mental model

An n-gram model asks: **“Have I seen this word sequence, or can I back off to a shorter one?”**

Bengio’s neural model asks: **“What features do these words have, and what have I learned about similar features in other contexts?”**
