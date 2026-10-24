# Pass 1: Find the paper’s argument

Read only:

- Abstract
- Introduction
- Section headings
- Conclusion
- Figure and table captions

Then answer, in your own words:

1. What is the weakness of traditional $n$-gram language models?
2. What is Bengio’s proposed solution?
3. What does the model learn?
4. What evidence shows that it works?

For this paper, the central argument is approximately:

> $n$-gram models treat words as unrelated symbols and suffer from sparsity. A neural model can learn distributed word representations, allowing similar words and contexts to share statistical strength.

Do not study every equation yet.

# Section 4: Experimental Results — First-Pass Notes

Section 4 asks one main question:

> Does the neural language model actually predict words better than strong $n$-gram baselines?

## Experimental setup

- The authors evaluate on two corpora: the **Brown corpus** and **AP News**.
- Each corpus is divided into training, validation, and test data.
- The validation set is used for model selection, weight decay, and early stopping. The final comparison is made on the unseen test set.
- Rare or special words are mapped to special symbols to keep the vocabulary manageable.

## What is compared?

- Several neural networks, called **MLPs**, with different context sizes, hidden-layer sizes, and word-feature dimensions.
- Strong $n$-gram baselines, especially modified **Kneser–Ney back-off models** and class-based $n$-grams.
- Some neural models are also **mixed** with an interpolated trigram, meaning their probability predictions are averaged.

## Evaluation metric: perplexity

The main metric is **test perplexity**. Lower perplexity is better because it means the model assigns higher probability to the words that actually occur.

For first-pass understanding, think of perplexity as:

> How surprised is the model by the test text?

## Main results

- On Brown, the best neural model performs substantially better than the best $n$-gram model: roughly **24% lower test perplexity** than the best $n$-gram, and about **33% lower** than the interpolated trigram.
- On AP News, the improvement is smaller but still meaningful: about **8% lower test perplexity** than the best $n$-gram.
- **The neural model benefits from longer context: increasing the context from two words to four words improves the neural model, whereas it does not noticeably help the $n$-gram models.**
- **Hidden units matter: models with a hidden layer perform better than comparable models without hidden units.**
- **Mixing neural and trigram predictions consistently improves performance. This suggests that the two models make different mistakes, so they provide complementary information.**

## What Section 4 establishes

The experiments support the paper’s central claim:

> Learning distributed word representations with a neural network allows the model to share statistical information across related words and to benefit from longer contexts. This leads to better generalization than the tested $n$-gram baselines, although the neural model requires much more computation.

For the first pass, do not memorize every table entry. Remember the pattern: **the neural model wins, longer context helps the neural model, hidden layers help, and combining neural and $n$-gram predictions helps even more.**

### Important result: MLP10

The best model, **MLP10**, uses a 5-word context, one hidden layer with 100 neurons, and 30-dimensional word vectors. It has no direct input-to-output connections. Its prediction is mixed equally with the interpolated MKN trigram model, producing a test perplexity of 252.

# QnA:

## Question 1

What is the meaning of “joint probability function of sequences of words in a language”? If the vocabulary contains only two words, `X` and `Y`, does this mean finding $P(XX)$, $P(XY)$, $P(YX)$, and $P(YY)$?

### Answer

Yes. For sequences of length two, these are examples of joint probabilities:

$$
P(XX),\quad P(XY),\quad P(YX),\quad P(YY)
$$

For example, $P(XY)$ means “the probability that the first word is $X$ **and** the second word is $Y$.”

The language model usually calculates these using conditional probabilities:

$$
P(XY) = P(X)P(Y\mid X)
$$

So the **joint probability** is for the whole sequence, while $P(Y\mid X)$ is the probability of the next word given the previous word.

If we consider only sequences of exactly two words, their probabilities should add up to $1$:

$$
P(XX)+P(XY)+P(YX)+P(YY)=1
$$

In a real language model, we also usually include a special end-of-sentence token, because sentences can have different lengths.

## Question 2

What does the paragraph about continuous variables, discrete spaces, local smoothness, and Hamming distance mean?

### Answer

The paragraph is explaining why **generalization is easier for continuous inputs than for discrete inputs**.

#### 1. What is generalization?

Generalization means using what the model learned from observed examples to make sensible predictions about new examples.

For example, suppose a model has seen the temperature `20°C` and learned that people usually wear light clothing. It can probably make a sensible prediction for `21°C`, even if `21°C` never appeared in the training data.

That is possible because nearby continuous values usually have similar meanings.

#### 2. Why continuous variables are easier

A continuous variable can take values along a range. Examples include:

- temperature: `20.0`, `20.1`, `20.2`, ...
- height: `170 cm`, `170.1 cm`, `170.2 cm`, ...
- a word represented by a vector such as $(0.2, -0.4, 0.7)$

For many real-world problems, changing a continuous input by a small amount causes the desired output to change only a small amount. This is called **local smoothness**.

In plain language:

> Nearby inputs tend to have nearby outputs.

For example, if a function predicts house prices:

$$
\text{price}(100\text{ m}^2) \approx \text{price}(101\text{ m}^2)
$$

The values will not necessarily be identical, but they are likely to be similar. A model can therefore learn from one point and make useful predictions near that point.

Smooth models such as neural networks and Gaussian mixture models are useful because they can represent this gradual change. If the model learns the behavior at one location, that behavior can influence nearby locations.

#### 3. Why discrete variables are harder

A discrete variable does not vary gradually. It takes one value from a set of separate categories.

For example, a word identity might be one of:

```text
cat, dog, car, language, probability, ...
```

There is no naturally meaningful point between `cat` and `dog` in the original representation. If we encode them as IDs:

```text
cat = 1
dog = 2
car = 3
```

then the numerical distance between the IDs is meaningless. `cat` is not necessarily more similar to `dog` than to `car` just because their IDs are closer.

A small change in a discrete variable can therefore have a completely different meaning. Replacing one word with another may drastically change the probability of a sentence:

```text
The dog chased the cat.
The dog chased the democracy.
```

The two sentences differ in only one discrete variable, but their grammaticality and meaning can be very different.

So, unlike the temperature example, we cannot automatically assume:

$$
P(\text{sentence with word } X) \approx P(\text{sentence with word } Y)
$$

merely because $X$ and $Y$ are represented as nearby category IDs—or because they occupy nearby positions in some list.

#### 4. What is Hamming distance?

Hamming distance counts how many positions differ between two sequences of discrete symbols.

For example:

```text
A B C D
A B X D
```

Only the third position differs, so the Hamming distance is $1$.

For word contexts:

```text
the cat sat on
the dog ran fast
```

All four positions differ, so their Hamming distance is $4$.

The important issue is that Hamming distance only says whether symbols are equal or different. It does not know that `cat` and `dog` are both animals, or that `sat` and `ran` are both verbs.

#### 5. What does “almost maximally far” mean?

Suppose a context contains $n$ words and each word is selected from a very large vocabulary. If two randomly selected contexts are compared, the probability that their first words match is small. The same is true for their second words, third words, and so on.

Therefore, two randomly selected contexts will usually differ in almost every position.

For example, with a context length of $4$:

```text
Context 1: the cat sat down
Context 2: a scientist discovered it
```

The Hamming distance is $4$, which is the maximum possible distance for four positions.

This becomes worse as the vocabulary grows. If the vocabulary has $V$ words, two independently selected words match with probability roughly $1/V$. For a large $V$, matching is rare. Thus most context pairs have a Hamming distance close to the maximum $n$.

#### 6. Why this is a problem for traditional $n$-gram models

Traditional $n$-gram models mostly treat each word as an independent discrete symbol. They learn counts such as:

$$
\text{count}(\text{the cat sat})
$$

But if the model encounters:

```text
the dog sat
```

it cannot automatically conclude that this context should behave somewhat like `the cat sat`. The two contexts are different discrete sequences, even though `cat` and `dog` are semantically related.

Because most contexts are far apart in Hamming distance, a count-based model sees relatively little useful overlap between them. This produces the sparsity problem.

#### 7. What Bengio’s model contributes

Bengio’s key idea is to replace each discrete word identity with a learned continuous vector:

$$
\text{cat} \rightarrow (0.2, -0.1, 0.8, \ldots)
$$

$$
\text{dog} \rightarrow (0.3, -0.2, 0.7, \ldots)
$$

If training causes related words to receive related vectors, then the neural network can behave similarly for contexts such as:

```text
the cat sat
the dog sat
```

The model is creating a smoother space for discrete words. It is not claiming that every word replacement should have a small effect. Rather, it learns a representation in which useful similarities can support generalization.

### Mental model

Continuous inputs come with a natural notion of “nearby,” so nearby examples can share information. Discrete word identities do not come with such a notion, so Bengio’s model learns a continuous geometry for them.

> The problem is not merely that there are many words. The deeper problem is that ordinary discrete representations provide no useful path for sharing information between different words and contexts.

## Question 3

What is the meaning of “distributed” in the phrase “distributed word feature vector”?

### Answer

“Distributed” means that the information describing a word is **spread across many numerical features**, rather than stored in one unique symbol or one position.

For example, a traditional one-hot representation might encode three words like this:

```text
cat = [1, 0, 0]
dog = [0, 1, 0]
car = [0, 0, 1]
```

Each word has one active position. This representation tells us that the words are different, but it does not express that `cat` and `dog` are more related than `cat` and `car`.

A distributed representation might look like this:

```text
cat = [0.8, 0.7, 0.1, 0.6]
dog = [0.7, 0.8, 0.2, 0.6]
car = [0.1, 0.2, 0.9, 0.3]
```

Here, each word is represented by a pattern across several dimensions. The dimensions may capture useful properties such as animacy, size, or grammatical behavior, although individual dimensions do not necessarily have simple human-readable meanings.

The information is distributed in two ways:

1. **One word uses many features.** Its meaning is represented by the whole vector.
2. **One feature is shared by many words.** For example, several animal words may have similar values in some dimensions.

This sharing is what helps the neural language model generalize. If `cat` and `dog` have similar vectors, learning something about one can influence predictions involving the other.

### Mental model

> A one-hot vector identifies a word; a distributed vector describes a word through a shared pattern of features.

## Question 4

What does this sentence mean?

> “where $g(\cdot)$ is a function represented by a neural network with a special left-to-right architecture, with the $i$-th output block $g_i(\cdot)$ computing parameters for expressing the conditional distribution of $Z_i$ given the value of the previous $Z$'s, in some arbitrary order.”

### Answer

The sentence describes how a joint probability over several variables can be built from a sequence of conditional probabilities.

Suppose we have variables:

$$
Z_1, Z_2, Z_3, \ldots, Z_n
$$

The chain rule of probability says:

$$
P(Z_1, Z_2, \ldots, Z_n)
= P(Z_1)P(Z_2\mid Z_1)P(Z_3\mid Z_1,Z_2)\cdots P(Z_n\mid Z_1,\ldots,Z_{n-1})
$$

In plain English:

1. Predict $Z_1$ with no previous variables.
2. Predict $Z_2$ given $Z_1$.
3. Predict $Z_3$ given $Z_1$ and $Z_2$.
4. Continue until $Z_n$.

## What are the $Z_i$ variables?

In a language model, each $Z_i$ can represent the word at position $i$ in a sequence:

```text
Z1 = the
Z2 = cat
Z3 = sat
```

Then the factorization becomes:

$$
P(\text{the},\text{cat},\text{sat})
= P(\text{the})
P(\text{cat}\mid\text{the})
P(\text{sat}\mid\text{the},\text{cat})
$$

This is the left-to-right language-modeling direction.

## What is the left-to-right architecture?

The neural network processes the sequence in an order. At position $i$, it receives information about the previous variables and uses them to predict the current variable:

```text
previous words       neural network       prediction
       ↓                    ↓                  ↓
      Z1          →        g1          →      P(Z1)
      Z1, Z2      →        g2          →      P(Z2 | Z1)
      Z1, Z2, Z3  →        g3          →      P(Z3 | Z1, Z2)
```

The phrase “the $i$-th output block $g_i(\cdot)$” means that the network’s output is divided into parts. The $i$-th part is responsible for the prediction at position $i$.

The special architectural restriction is that $g_i$ may use the earlier variables, but not future ones. In other words, the prediction for $Z_i$ cannot secretly look at $Z_{i+1}$ or later variables. This preserves the left-to-right prediction process.

## What does “computing parameters” mean?

The network may not output one final probability directly. Instead, $g_i(\cdot)$ outputs the numbers needed to define the conditional distribution.

For a discrete word variable, these numbers might be scores or probabilities for every possible next word:

```text
the   → 0.10
cat   → 0.45
dog   → 0.30
sat   → 0.15
```

These values describe:

$$
P(Z_i \mid Z_1,\ldots,Z_{i-1})
$$

For continuous variables, the parameters could instead be a mean and variance of a Gaussian distribution. The paper is stating the idea generally; in this language-model application, the relevant distribution is over possible words.

## What does “in some arbitrary order” mean?

The chain rule allows the variables to be ordered in different ways. For three variables, we could write:

$$
P(Z_1,Z_2,Z_3)
=P(Z_1)P(Z_2\mid Z_1)P(Z_3\mid Z_1,Z_2)
$$

or use another valid order:

$$
P(Z_1,Z_2,Z_3)
=P(Z_2)P(Z_3\mid Z_2)P(Z_1\mid Z_2,Z_3)
$$

Both are mathematically valid. For language, however, left-to-right order is natural because we want to predict the next word from preceding words.

### Mental model

> $g_i$ is the part of the neural network that looks at the previous variables and produces the information needed to define the probability distribution of the next variable.
