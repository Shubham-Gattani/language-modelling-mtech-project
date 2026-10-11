### 1. What Cross-Entropy Means Exactly

At its core, **cross-entropy** measures **average surprise** or **uncertainty** per word [25, 240–241].

In information theory, cross-entropy quantifies how many **bits of information** a model requires to encode or predict each word in a sequence.

#### The Intuition: "Surprise Bits"

* **High Probability $\implies$ Low Surprise**: If a model predicts a word with $P(\text{"book"} \mid \text{"read a"}) = 0.5$, it is not surprised. The surprise in bits is $-\log_2(0.5) = 1\text{ bit}$.
* **Low Probability $\implies$ High Surprise**: If a model predicts a word with $P(\text{"asbestos"} \mid \text{"read a"}) = 0.001$, it is extremely surprised. The surprise in bits is $-\log_2(0.001) \approx 9.97\text{ bits}$.

**Cross-entropy ($H$) is simply the average of these surprise bits across all words in a text.**

#### Mathematical Formula

For a model $p$ predicting words $w_i$, the cross-entropy $H_p$ is defined as:

$$
H_p = -\frac{1}{W} \sum_{i=1}^{W} \log_2 p(w_i \mid \text{context})
$$


where $W$ is the total number of words.

#### Connection to Perplexity

Cross-entropy ($H$) and **Perplexity ($PP$)** are two sides of the same coin:

$$\
PP = 2^H\
$$

* If cross-entropy is **$3.0\text{ bits/word}$**, perplexity is $2^3 = \mathbf{8}$ (meaning on average, the model is as uncertain as picking uniformly among 8 candidate words).
* **Lower cross-entropy is always better** because fewer bits mean better prediction and less surprise [26–27].

---

### 2. What "Cross-Entropy of a Test Set" Means

The **"cross-entropy of a test set"** measures how well a trained language model generalizes to a completely new, **unseen dataset** $T$.

#### Why Evaluate on a Test Set?

If you evaluate cross-entropy on the *training set*, a raw maximum likelihood model can "cheat" by memorizing observed counts. To measure true prediction quality, we evaluate the model on an independent **test set** $T = (t_1, t_2, \dots, t_{l_T})$ containing $W_T$ total words that were held out during training [24–25, 101].

#### How It Is Calculated Step-by-Step

1. **Calculate Sequence Probability**: The model assigns a probability $p(T)$ to the entire test set by multiplying the conditional probabilities of all words:

$$
p(T) = \prod_{i=1}^{W_T} p(w_i \mid w_{i-n+1}^{i-1})
$$


2. **Convert to Bits**: Take the negative log base 2 of $p(T)$ to get the total bits needed to encode the entire test set:

$$\
\text{Total Bits} = -\log_2 p(T)\
$$

3. **Normalize by Word Count**: Divide total bits by the total number of test tokens $W_T$ (including sentence-ending tokens like `</s>`):

$$
H_p(T) = -\frac{1}{W_T} \log_2 p(T)
= -\frac{1}{W_T} \sum_{i=1}^{W_T} \log_2 p(w_i \mid \text{history})
$$


#### Key Takeaway

* **Cross-entropy of a test set** is the universal **gold standard metric** in language modeling literature.
* Typical cross-entropies for English text range from **6 to 10 bits/word** (corresponding to perplexities between 50 and 1,000).
* A lower test set cross-entropy proves that a smoothing method (like Modified Kneser-Ney) is effectively allocating probability mass to unseen events rather than overfitting.
