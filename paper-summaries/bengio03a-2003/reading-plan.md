I recommend replacing line-by-line reading with a **progressive reconstruction method**:

> First understand what the paper is trying to change. Then reconstruct how the method works. Only afterward study each equation and experiment in detail.

For Bengio et al. (2003), use this sequence.

### Pass 1: Find the paper’s argument

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

### Pass 2: Draw the complete model flow

Before reading mathematical details, draw this:

```text
previous words
      ↓
word representations / embeddings
      ↓
hidden neural network
      ↓
probability distribution over next word
      ↓
training objective: maximize probability of observed next words
```

Then identify what each stage receives and produces.

For example:

- Input: a fixed context such as $(w_{t-3}, w_{t-2}, w_{t-1})$
- Representation: vectors for those words
- Network: combines the vectors
- Output: probabilities for possible $w_t$

If you cannot explain this flow, equations will not help yet.

### Pass 3: Read equations by function, not by order

Group the equations into four questions:

1. **How is a word represented?**
2. **How are several context words combined?**
3. **How is the next-word probability calculated?**
4. **How are the parameters learned?**

For each equation, use this format:

```text
Plain English:
What is this equation calculating?

Symbols:
What does every symbol mean?

Reason:
Why does the model need this calculation?

Tiny example:
What happens with a small context and a small vocabulary?
```

This prevents the paper from becoming a sequence of disconnected formulas.

### Pass 4: Compare it directly with an $n$-gram model

Make a contrast table:

| Question | Traditional $n$-gram | Bengio neural model |
|---|---|---|
| Representation of a word | Discrete identity | Learned vector |
| Similar words | Usually unrelated | Can have similar vectors |
| Generalization | Mostly exact context counts | Shared through representations |
| Probability estimation | Counts and smoothing | Neural network |
| Main challenge | Sparsity | Computational cost and optimization |

The most important conceptual contrast is:

> An $n$-gram model generalizes mainly through overlapping counts. Bengio’s model generalizes through learned representations.

### Pass 5: Understand the experiments as evidence

Do not read the results as just numbers. Ask:

- What hypothesis is this experiment testing?
- What baseline is being used?
- What does lower perplexity demonstrate?
- Is the improvement caused by the neural architecture, the word representations, or both?
- What limitations remain?

This turns the experimental section into an argument rather than a table-reading exercise.

### Pass 6: Reconstruct the paper without looking

Close the PDF and explain:

1. The problem with $n$-gram models.
2. The model’s input and output.
3. Why word vectors help.
4. How the probability of the next word is computed.
5. How training works.
6. What the experiments establish.

Wherever you get stuck, return to that specific section. This is much more efficient than rereading the entire paper line by line.

For this paper, I would use the following study order:

```text
Problem → model diagram → word representations
→ probability equation → training objective
→ comparison with n-grams → experiments
→ limitations and historical significance
```

Line-by-line reading is still useful—but only as a **local debugging tool** after you know the paper’s global structure.

The durable mental model for this paper is:

> Bengio replaces “have I seen this exact word sequence?” with “what useful representation can I learn for these words and their context?”