# Pass 2: Draw the complete model flow

The goal of this pass is not to re-learn neural-network mechanics. The important thing is to follow **one training example from its input context to its loss**, and then understand how the same model generates a probability distribution for every possible next word.

## 1. Complete model flow

Suppose the training text contains:

```text
the cat sat down
```

With a context of two words, the training examples are:

```text
input context       observed next word
the cat              sat
cat sat              down
```

For one example, the flow is:

```text
previous words
    ↓
look up each word's learned feature vector
    ↓
concatenate the vectors into one input representation
    ↓
neural network g(·)
    ↓
scores for every possible vocabulary word
    ↓
softmax converts scores into probabilities
    ↓
probability assigned to the observed next word
    ↓
loss: penalize the model when this probability is low
```

The same word-feature matrix and the same network parameters are reused for every position in the text. This is the parameter sharing across time described in the paper.

## 2. What does “probability distribution over the next word” mean?

Assume the vocabulary is only:

```text
{sat, slept, ran}
```

Given the context `the cat`, the neural model does not output only one word. It outputs one probability for **every possible next word**:

| Candidate next word | Model probability |
|---|---:|
| sat | 0.70 |
| slept | 0.20 |
| ran | 0.10 |
| Total | 1.00 |

This complete list is the **conditional probability distribution**:

$$
P(w_i \mid w_{i-2}, w_{i-1})
$$

It means: “given the previous words, how much probability does the model assign to each possible word at position $i$?”

The probabilities must add up to $1$:

$$
P(\text{sat}\mid\text{the cat}) +
P(\text{slept}\mid\text{the cat}) +
P(\text{ran}\mid\text{the cat}) = 1
$$

The model may predict `sat` as the most likely word, but the distribution still retains probabilities for `slept` and `ran`. This matters because language is uncertain: several continuations can be plausible.

### Distribution versus one probability

These two ideas are related but different:

| Term | Meaning |
|---|---|
| Probability distribution | Probabilities for all possible next words |
| $P(\text{sat}\mid\text{the cat})$ | The probability of one particular next word |
| Predicted word | Often the word with the highest probability |

So, when the paper says that the model predicts the next word, it usually means that the network first produces the **whole distribution**. Choosing the highest-probability word is only one possible use of that distribution.

## 3. How does the network produce the distribution?

The final layer first produces an unnormalized score for every vocabulary word. These scores are often called ***logits***. For example:

```text
score(sat)   = 2.0
score(slept) = 0.5
score(ran)   = -0.2
```

Softmax converts these scores into positive numbers that sum to $1$:

$$
P(w\mid h) = \frac{\exp(\text{score}(w,h))}
{\sum_{v\in V}\exp(\text{score}(v,h))}
$$

Here:

- $h$ is the history or context, such as `the cat`;
- $w$ is one candidate next word;
- $V$ is the vocabulary;
- the denominator sums over every candidate word $v$ in the vocabulary.

Therefore, the network's output is a valid probability distribution over the vocabulary.

## 4. What does “maximize the probability of observed next words” mean?

During training, the correct next word is already known because it comes from the text. For the example:

```text
context: the cat
observed next word: sat
```

Suppose the model outputs:

```text
P(sat | the cat)   = 0.70
P(slept | the cat) = 0.20
P(ran | the cat)   = 0.10
```

The training objective focuses on the probability of the word that actually occurred:

$$
P(\text{sat}\mid\text{the cat}) = 0.70
$$

The model is trained so that this probability becomes larger. It is not directly told that `sat` is the only possible continuation; it is told that `sat` should receive high probability relative to the alternatives.

For the complete training text, the objective is:

$$
\max_\theta \sum_i \log P_\theta(w_i\mid h_i)
$$

In plain English:

> Adjust the parameters $\theta$ so that the words that actually appear in the training text receive high probability in their contexts.

The equivalent loss minimization form is negative log-likelihood:

$$
\mathcal{L}(\theta) = -\sum_i \log P_\theta(w_i\mid h_i)
$$

The neural network is trained to minimize this loss.

## 5. Why use the logarithm?

The probability of a whole sentence is a product of conditional probabilities:

$$
P(w_1,\ldots,w_T) = \prod_{i=1}^{T} P(w_i\mid h_i)
$$

Products of many small numbers become extremely small. Taking a logarithm converts the product into a sum:

$$
\log P(w_1,\ldots,w_T)
= \sum_{i=1}^{T}\log P(w_i\mid h_i)
$$

This is easier and more numerically stable to optimize. Maximizing the log-probability is equivalent to maximizing the original probability because the logarithm is monotonically increasing.

## 6. A concrete loss comparison

For the same observed word `sat`, compare two models:

```text
Model A: P(sat | the cat) = 0.70
Model B: P(sat | the cat) = 0.10
```

Their negative log losses are approximately:

$$
\text{Loss}_A = -\log(0.70) \approx 0.36
$$

$$
\text{Loss}_B = -\log(0.10) \approx 2.30
$$

Model B receives a much larger penalty because it considered the observed word unlikely. Gradient descent changes the word vectors and network weights so that, in this context, probability moves toward `sat` and away from incorrect alternatives.

## 7. The distinction that is most important for this paper

The model is **not** learning a single label such as:

```text
the cat → sat
```

It is learning a conditional distribution:

```text
the cat →
    sat:   0.70
    slept: 0.20
    ran:    0.10
```

This distribution is useful for several things:

- selecting the most likely next word;
- assigning a probability to a complete sentence;
- calculating perplexity;
- sampling different plausible continuations.

## 8. What you should be able to explain after Pass 2

You should be able to reconstruct this sentence:

> The model receives a fixed-length history, converts its words into learned distributed vectors, uses a neural network to compute a score for every vocabulary word, converts those scores into a conditional probability distribution with softmax, and updates all parameters to increase the probability of the observed next word.

### Mental model

**The network does not guess one word directly. It distributes one unit of probability mass across the entire vocabulary, and training rewards it for putting more mass on the word that actually occurred.**

---

> So, is the loss calculated by calculating the negative log of the probability assigned to the actual word?

Yes, exactly.
For one training example:

$$
\text{Loss} = -\log P(\text{actual next word} \mid \text{context})
$$

Example:

$$
P(\text{sat}\mid\text{the cat})=0.7
$$

Then:

$$
\text{Loss}=-\log(0.7)\approx0.36
$$

If the model assigns a low probability, such as $0.1$, the loss is much higher:

$$
-\log(0.1)\approx2.30
$$

So the model learns to assign higher probability to the word that actually occurred.
