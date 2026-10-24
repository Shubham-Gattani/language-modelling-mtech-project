# Pass 5: Understand the experiments as evidence

The experimental section is not merely a table of numbers. It is the authors’ evidence for a chain of claims:

1. A neural language model can outperform strong traditional n-gram models.
2. Distributed word representations help the model generalize.
3. A nonlinear hidden layer is useful.
4. The neural model can use a longer context effectively.
5. Neural and n-gram models can make complementary errors, so mixing them can improve performance further.

## 1. Experimental setup

The authors evaluate on two corpora:

- **Brown corpus:** a broad collection of English text from different genres.
- **AP News:** news text, with a somewhat different style and domain.

Each corpus is divided into:

```text
training set   → learn the parameters
validation set  → choose settings and control overfitting
test set        → final evaluation on unseen text
```

The test set is important because a model can perform well on training data simply by memorizing it. The test result asks whether the learned model generalizes to new text.

Rare words are represented using special symbols so that the vocabulary remains manageable. This also means the experiment evaluates the model’s ability to predict within a fixed vocabulary, rather than requiring it to generate arbitrary unseen spellings.

### 1.2 Parameters versus settings

**Learn the parameters** means learn the numerical values inside one chosen model using the training set. Examples include:

```text
word vectors, neural-network weights, and biases
```

**Choose settings** means choose the model design and training choices before or during training. These are often called **hyperparameters**. Examples include:

```text
context length, number of hidden units, word-vector dimension,
learning rate, regularization strength, and mixing weight
```

We do not choose settings only by training performance because a more flexible model can overfit the training text. It may memorize the training examples and appear better without generalizing better. The validation set is used to choose settings that perform well on text not used to learn the parameters. The test set is kept untouched for the final comparison.

## 2. What is the baseline?

The neural models are compared with strong traditional language-model baselines, especially:

- modified Kneser–Ney n-gram models;
- interpolated trigram models; and
- class-based n-gram models.

This matters because the comparison is not against a weak or unsmoothed count model. If the neural model performs better, the result is evidence that the learned representations and neural architecture provide value beyond ordinary count smoothing.

The experiments also compare several neural models with different:

- context lengths;
- numbers of hidden units;
- word-vector dimensions; and
- connections between the input and output layers.

These comparisons help identify which parts of the neural approach are useful.

## 3. What does lower perplexity mean?

Perplexity measures how surprised the model is by the test text. Lower perplexity is better.

The connection to the training objective is:

$$
\mathrm{Perplexity}
=
\exp\left(
-\frac{1}{T}
\sum_{t=1}^{T}
\log P(w_t \mid \text{context}_t)
\right)
$$

The expression inside the exponential is the average negative log-probability of the observed words. Therefore:

- assigning higher probability to the actual next words lowers the loss;
- lower average loss gives lower perplexity; and
- lower perplexity means the model was less surprised by the test text.

For example, if two models are evaluated on the same test words and Model A has perplexity `250` while Model B has perplexity `300`, Model A assigned better overall probability to the observed text.

Perplexity does not mean that the model literally considered exactly 250 words equally likely at every position. It is an exponential transformation of average log-loss, useful for comparing language models.

## 4. Evidence from the main results

### 4.1 Neural models versus n-gram baselines

**Hypothesis:** learned distributed representations can provide better generalization than discrete n-gram counts and smoothing alone.

**Comparison:** the neural models are compared with modified Kneser–Ney and other strong n-gram baselines.

**Observation:** on the Brown corpus, the best neural system achieves substantially lower test perplexity than the best n-gram model—about 24% lower according to the paper’s comparison. It is also about 33% lower than the interpolated trigram baseline.

On AP News, the improvement is smaller but still meaningful—about 8% lower perplexity than the best n-gram model.

**What this supports:** the neural approach is not only fitting the training data. It is producing better probability estimates on unseen text, supporting the paper’s central claim about representation-based generalization.

**What it does not prove:** this result alone does not isolate the contribution of every design choice. A complete explanation must also examine context length, hidden units, vector dimensions, and model mixing.

### 4.2 Longer context

**Hypothesis:** a neural model can benefit from more preceding words because its shared representation lets it use information from a longer context without storing a separate parameter table for every complete sequence.

**Comparison:** neural models using shorter and longer contexts are compared with n-gram models using corresponding context sizes.

**Observation:** increasing the neural context from two words to four words improves performance, while increasing context does not noticeably help the traditional n-gram models in the reported comparison.

**Interpretation:** the neural model can extract useful patterns from a longer context. The n-gram model pays a heavy sparsity cost when the context becomes longer: many longer sequences are rare or unseen.

**Important limitation:** this does not mean the neural model understands unlimited context. In this paper, the model still uses a fixed-size context window. The improvement shows that the neural model uses its available window more effectively.

### 4.3 Hidden layer and nonlinearity

**Hypothesis:** a nonlinear hidden layer can learn interactions among the context words that a simpler model cannot represent as effectively.

**Comparison:** models with hidden units are compared with comparable models without hidden units.

With a hidden layer, the context vector is first transformed nonlinearly:

$$
h = \tanh(d + Hx)
$$

and then converted into output scores:

$$
y = b + Wh
$$

Here, `x` is the concatenated context-vector input, `h` is the hidden representation, `H` and `W` are weight matrices, and `d` and `b` are bias vectors. The `tanh` function introduces nonlinearity.

Without a hidden layer, the model can use a direct linear transformation:

$$
y = b + Wx
$$

The scores `y` are then passed through softmax in both cases. Therefore, the main architectural difference being tested is whether the intermediate nonlinear representation `h` improves the probability distribution.

**Observation:** models with a hidden layer perform better.

**Interpretation:** the hidden layer is doing more than storing word vectors. It combines the context features and learns nonlinear patterns that are useful for predicting the next word.

This supports the role of the neural architecture, not just the role of having a continuous vector for every word.

### 4.4 Mixing neural and n-gram predictions

**Hypothesis:** the neural and n-gram models capture different useful regularities and therefore make different mistakes.

**Method:** combine their probability distributions, for example by averaging the neural prediction with the interpolated MKN trigram prediction:

$$
P_{\mathrm{mixed}}(w \mid h)
=
\alpha P_{\mathrm{neural}}(w \mid h)
+
(1-\alpha)P_{\mathrm{MKN}}(w \mid h)
$$

For equal mixing, `α = 0.5`.

**Observation:** mixing consistently improves performance. The best reported Brown-corpus system, MLP10 mixed with the interpolated MKN trigram model, reaches a test perplexity of 252.

**Interpretation:** the neural model and MKN model are complementary. The neural model contributes representation-based generalization, while the n-gram model contributes precise local count information.

This is important evidence: the neural model is not merely reproducing the same information as the n-gram model. If both models made exactly the same predictions and errors, averaging them would not usually improve the result.

## 5. What does MLP10 demonstrate?

The best model reported in the comparison, **MLP10**, uses:

- a 5-word context;
- one hidden layer with 100 neurons;
- 30-dimensional word vectors;
- no direct input-to-output connections; and
- an equal mixture of its neural prediction with the interpolated MKN trigram model.

Its test perplexity is 252.

MLP10 is useful as a concrete summary of the successful design, but it should not be read as proof that these exact hyperparameters are universally optimal. They are the best settings found in this experiment under this dataset and evaluation procedure.

## 6. Is the improvement caused by the architecture, the representations, or both?

The fairest answer is: **both contribute, and the experiments provide partial evidence about each.**

- Better performance from hidden-layer models supports the value of the neural architecture and its nonlinear interactions.
- Better performance with longer neural contexts supports the way the architecture shares parameters across positions.
- The central motivation and design depend on learned distributed word vectors, which allow related words to share statistical strength.
- Improved results from jointly training the vectors and the network support learning representations as part of the language-model objective.
- Improved performance after mixing shows that the neural model adds information beyond the MKN trigram model.

However, the experiments do not perform a perfectly isolated ablation of every possible factor. For example, changing the word-vector size, hidden-layer size, context length, and mixing strategy can change several things at once. Therefore, the results support the overall neural language-model approach more strongly than they identify one single cause for every improvement.

## 7. What limitations remain?

The experiments also reveal important limitations:

- **Computational cost:** neural models require many matrix operations and are slower to train and evaluate than count-based models.
- **Large output layer:** producing a probability for every vocabulary word is expensive when the vocabulary is large.
- **Fixed context:** the model only directly sees a limited number of previous words.
- **Unknown and rare words:** words outside the fixed vocabulary must be mapped to special symbols, losing their individual identity.
- **Data dependence:** useful word vectors must be learned from enough training data.
- **Optimization and tuning:** performance depends on choices such as hidden-layer size, vector dimension, context length, regularization, and mixing weight.
- **Domain dependence:** the size of the gain differs between Brown and AP News, so the improvement is not identical across domains.

## The experimental argument in one chain

```text
Strong n-gram baselines
        ↓ compare on unseen test text
Neural models achieve lower perplexity
        ↓ inspect controlled comparisons
Longer contexts and hidden layers help neural models
        ↓ combine predictions
Neural + MKN performs even better
        ↓ conclusion
Learned representations and neural computation provide useful
generalization, while n-gram models still provide complementary information.
```

### Mental model

Do not read the results as “the neural model has a smaller number.” Read them as an argument:

> The neural model predicts unseen text better than strong n-gram baselines; longer contexts and hidden layers help it; and mixing it with an n-gram model helps further because the two models capture different kinds of evidence.
