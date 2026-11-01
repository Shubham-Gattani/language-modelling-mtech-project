# Language Modeling with Gated Convolutional Networks

Paper: Yann N. Dauphin, Angela Fan, Michael Auli, and David Grangier, 2017.

This is a story-first explanation of the paper. It is written to help you understand the main idea, the equations, and the shape of a Python implementation before reading the paper line by line.

## The one-minute story

Suppose the model has read:

```text
The cat sat on the
```

It must predict the next word. A traditional recurrent language model, such as an LSTM, reads the sequence one step at a time. To predict after `the`, it updates a hidden state; then it uses that state to process `cat`; then `sat`; and so on. This is powerful, but the sequence positions depend on one another, so the model cannot process all positions at once.

The paper asks:

> Can we build a language model that looks at preceding words using convolutions, processes all positions in parallel, and still learns which information is useful?

Their answer is a **gated convolutional neural network**, or GCNN:

1. Turn every word into an embedding vector.
2. Apply a causal temporal convolution, so the representation at position `i` can see only positions before `i`.
3. Use a gate to decide which parts of the convolution result should pass upward.
4. Stack many such layers. Each layer sees a slightly wider neighborhood, so the top layers summarize a larger context.
5. Add residual connections so information and gradients can travel through the deep network.
6. Apply an output layer and softmax-like vocabulary prediction.

The central trade-off is:

- an RNN has a theoretically unbounded context but sequential computation;
- a GCNN has a fixed context window but highly parallel computation.

The paper’s experiments argue that a sufficiently large finite context is enough for strong language modeling, while being much faster to score sentence tokens.

## 1. Start with the language-modeling problem

A language model assigns a probability to a sequence by breaking it into next-word predictions:

$$
P(w_0,\ldots,w_N)=P(w_0)\prod_{i=1}^{N}P(w_i\mid w_0,\ldots,w_{i-1}).
$$

Read this as:

> The probability of the whole sentence is the probability of the first word multiplied by the probability of each later word given everything before it.

For each position `i`, the model produces a distribution over the vocabulary. The correct next word receives a probability, and training increases its log-probability, usually through cross-entropy loss.

The neural network does not directly manipulate word identities. It first looks up an embedding for each word:

$$
E=[D_{w_0},D_{w_1},\ldots,D_{w_N}],
$$

where `D` is an embedding table and $D_w$ is the vector for word `w`.

If the sequence has length `N` and the embedding size is `e`, then `E` is roughly an `N x e` matrix. In a PyTorch implementation, the same information may be arranged as `(batch, channels, time)` instead of `(time, channels)`. That layout difference is an implementation detail; the idea is the same.

## 2. Why not simply use an RNN?

An RNN computes a hidden representation like this:

$$
h_i=f(h_{i-1},w_{i-1}).
$$

The representation at position `i` depends on the representation at `i-1`, which depends on `i-2`, and so forth. Therefore the computation is a chain:

```text
w0 -> h1 -> h2 -> h3 -> h4 -> ...
```

This gives RNNs a natural way to carry information for a long time. But it also creates a strict order of computation. Even if a GPU has enough arithmetic units to process every position simultaneously, position `i` must wait for position `i-1`.

The paper’s alternative is to compute representations using local windows:

```text
window 1 -> h1
window 2 -> h2
window 3 -> h3
window 4 -> h4
```

All those windows can be processed at the same time. A convolutional network therefore exchanges unbounded memory for parallelism and a finite receptive field.

## 3. What “convolutional” means here

A one-dimensional temporal convolution slides a small kernel across the sequence. If the kernel width is `k`, the output at position `i` is computed from a local window of `k` neighboring token representations.

For language modeling, the important word is **causal**. The output used to predict the next word must not see the future. If we are predicting after position `i`, the representation must not contain $w_{i+1}, w_{i+2}$, and so on.

The paper enforces this by shifting or padding the input before convolution. At the beginning of the sequence it pads with `k-1` special elements. The first input is treated as a beginning-of-sequence marker, and the model predicts later words rather than predicting that marker.

For a kernel width of `3`, the causal windows can be pictured as:

```text
prediction at position 0: <S>  <S>  w0
prediction at position 1: <S>  w0   w1
prediction at position 2: w0   w1   w2
```

The exact indexing depends on whether the implementation predicts the current token from previous tokens or shifts the labels by one. The invariant is what matters:

> Every prediction position may use the past, never the future.

### Why stacking convolutions gives long context

One convolution with width `3` sees about three positions. A second convolution consumes representations that already summarize three-position windows, so its top-level output can depend on a wider span. Repeating this builds a hierarchical representation:

```text
words -> short phrases -> larger phrases -> broad context -> prediction
```

This is why the paper compares the hierarchy to grammar-like structure: lower layers can capture local patterns, while higher layers combine them into more abstract patterns.

If a layer has kernel width `k` and there are `L` ordinary stacked layers with stride `1`, a rough receptive-field calculation is:

$$
1+L(k-1).
$$

For `k=3` and `L=5`, this gives `1 + 5(2) = 11` positions. Real architectures contain residual blocks and sometimes different kernel widths, so implementation code should be used for the exact receptive field.

The main idea is not that a single convolution understands a paragraph. The idea is that a stack of local operations creates a wider context without requiring one sequential operation per token.

## 4. Why “gated” is in the name

A plain convolution transforms its input and passes the result onward. The paper adds a learned control signal that decides how much of each feature should pass.

The basic Gated Linear Unit, or **GLU**, is:

$$
h_l(X)=(X*W+b)\otimes\sigma(X*V+c).
$$

Here is the equation in plain language:

- `X` is the input sequence representation.
- `X * W + b` is a convolution producing candidate features.
- `X * V + c` is another convolution producing gate values.
- `sigma` is the sigmoid function, so gate values lie between `0` and `1`.
- `otimes` is element-wise multiplication.
- `h_l(X)` is the output of the layer.

Use two names to remember the two branches:

$$
A=X*W+b
$$

is the **content** or candidate branch, and

$$
G=\sigma(X*V+c)
$$

is the **gate** branch. The layer returns:

$$
h=A\otimes G.
$$

For one scalar feature, imagine:

```text
candidate A = 2.0
gate G      = sigmoid(-1.0) ≈ 0.27
output      = 2.0 * 0.27 ≈ 0.54
```

If the gate is close to `1`, the candidate passes almost unchanged. If the gate is close to `0`, the candidate is mostly suppressed. The gate is learned separately for every position and feature channel, so the model can select different information in different contexts.

This is the precise reason for the word **gated**: a sigmoid-controlled multiplicative path regulates the flow of the convolution’s features.

The gate is not a human-interpretable switch saying “this is a noun.” It is a learned, distributed control value. A gate may learn behavior related to useful words, phrase boundaries, or feature combinations, but the paper’s claim is about useful information flow, not a symbolic linguistic annotation.

## 5. Why use a linear branch instead of tanh?

The paper compares the GLU with an LSTM-style convolutional gate, which it calls a gated tanh unit, or GTU:

$$
\operatorname{GTU}(X)=\tanh(X*W+b)\otimes\sigma(X*V+c).
$$

The difference is small to write but important for optimization:

- GTU squashes the candidate through `tanh` before multiplying by the gate.
- GLU leaves the candidate branch linear and places the sigmoid only on the control branch.

Why does that matter? Consider the derivative of the simplified scalar form `x * sigmoid(x)`:

$$
\nabla[x\otimes\sigma(x)]
=\nabla x\otimes\sigma(x)+x\otimes\sigma'(x)\nabla x.
$$

The first term provides a path through the candidate that is not multiplied by `tanh'`. When the gate is open, this gives the gradient a relatively direct route through the layer. The paper describes this as a multiplicative skip connection.

By contrast, the GTU derivative contains both `tanh'` and `sigma'` factors. Sigmoid and tanh derivatives become small when their inputs saturate. Repeated small factors can make gradients vanish as layers are stacked.

The useful mental model is:

> The GLU keeps the information branch simple and uses the nonlinear sigmoid branch as a learned valve.

The model is still nonlinear because the gate depends nonlinearly on `X`. Removing the gate entirely would leave a linear convolutional stack, which is much less expressive.

## 6. A small GLU example

Suppose, for one position, the convolution branches produce three candidate features:

```text
A = [ 2.0, -1.0,  0.5 ]
```

and pre-sigmoid gate values:

```text
z = [ 2.0, -2.0,  0.0 ]
```

The sigmoid values are approximately:

```text
sigmoid(2.0)  ≈ 0.88
sigmoid(-2.0) ≈ 0.12
sigmoid(0.0)  = 0.50
```

Therefore:

```text
output = A * sigmoid(z)
       ≈ [2.0*0.88, -1.0*0.12, 0.5*0.50]
       ≈ [1.76, -0.12, 0.25]
```

The first feature is mostly passed, the second is mostly suppressed, and the third is half passed. In a real layer, these values are produced for every time position and every feature channel.

## 7. The complete architecture

The paper’s Figure 1 can be translated into this implementation flow:

```text
token ids
   |
   v
embedding lookup: E
   |
   v
causal convolution + GLU
   |
   v
residual convolutional blocks
   |
   v
hidden representation H
   |
   v
vocabulary output layer / adaptive softmax
   |
   v
next-word probabilities
```

More formally, the hidden representation is produced by composing layers:

$$
H=h_L\circ\cdots\circ h_1\circ h_0(E).
$$

Each `h_l` is a convolution plus GLU, wrapped in a residual block. The residual idea is:

$$
\operatorname{block}(X)=X+F(X),
$$

where `F` is the convolutional/GLU transformation. This gives the network a path that can carry the input forward and makes optimization easier. It also helps the model learn a useful modification of the input rather than having every layer rebuild the representation from nothing.

The paper describes bottleneck blocks for efficiency. A bottleneck uses narrow `k=1` convolutions before and after the wider temporal convolution:

```text
wide channel representation
        |
        v
1 x 1 convolution: reduce channels
        |
        v
k x 1 temporal convolution: mix nearby positions
        |
        v
1 x 1 convolution: restore channels
```

The wider convolution is expensive because it mixes positions and channels. Performing it in a lower-dimensional channel space reduces computation while preserving much of the modeling capacity.

## 8. How to understand the implementation tensor shapes

When reading Python, identify these objects first:

1. `tokens` or `input`: integer token IDs, often shaped `(batch, time)`.
2. `embedding`: vectors, often `(batch, time, embedding_dim)`.
3. `conv_input`: the same vectors rearranged to the layout expected by the convolution, often `(batch, channels, time)`.
4. `A`: candidate convolution output.
5. `B` or `gate_logits`: second convolution output used to compute the gate.
6. `gate`: `sigmoid(B)`.
7. `H`: `A * gate`, followed perhaps by dropout, residual addition, or normalization.
8. `logits`: vocabulary scores from the final output layer.

In pseudocode, the core layer looks like this:

```python
def gated_conv(x):
    candidate = conv_content(x)
    gate_logits = conv_gate(x)
    gate = sigmoid(gate_logits)
    return candidate * gate
```

An optimized implementation may use one convolution that outputs twice as many channels, then split them:

```python
joined = convolution(x)          # channels = 2 * hidden_size
candidate, gate_logits = split(joined, 2, dim=channel_dimension)
return candidate * sigmoid(gate_logits)
```

This is mathematically equivalent to using separate `W` and `V` convolutions, provided the parameterization and biases are arranged equivalently.

The most important implementation checks are:

- Is the convolution causal?
- Is the time dimension preserved after padding?
- Are the candidate and gate tensors the same shape before element-wise multiplication?
- Is the residual input aligned with the block output?
- Are labels shifted so the model never trains on future information?

## 9. Output prediction and adaptive softmax

After the final convolutional block, the model has a hidden vector for each prediction position. A standard output layer would map each hidden vector to one logit per vocabulary word and apply softmax.

For very large vocabularies, a full softmax is expensive because every prediction touches every word. The paper therefore uses **adaptive softmax**. The basic idea is to spend more capacity and computation on frequent words and less on rare words. This reduces memory and computation while preserving good prediction quality.

This is separate from the central GCNN contribution. When comparing results, remember that a language model’s perplexity depends on both:

- the hidden architecture, such as LSTM versus GCNN; and
- the output approximation, such as full softmax versus adaptive softmax.

The paper tries to make comparisons fair by controlling these choices where possible.

## 10. What experiments did the paper run?

### Datasets

The authors use two large language-modeling datasets:

- **Google Billion Word:** almost one billion tokens, a very large vocabulary, and shuffled sentences. The average sentence is relatively short, about 20 words.
- **WikiText-103:** over 100 million tokens with consecutive text. Sequences can be paragraphs or long document sections, with an average document context around 4000 tokens.

They add beginning-of-sequence and end-of-sequence markers. The model sees the markers as input but is evaluated on predicting the end marker rather than predicting the beginning marker.

They report **perplexity**. Lower perplexity means the model assigns higher probability to the actual test sequence on average.

### Main performance result

On Google Billion Word, the comparable GCNN obtains test perplexity 38.1 versus 39.8 for a comparable LSTM in the paper’s setup. A larger bottleneck GCNN reaches 31.9 using 8 GPUs, close to a much larger LSTM’s 30.6 using 32 GPUs and more training time.

On WikiText-103, the GCNN also beats the listed LSTM baselines: the reported GCNN-8 reaches 44.9 versus 48.7 for the listed LSTM-1024, and the larger GCNN-14 reaches 37.2 using 4 GPUs.

The paper’s important claim is not simply “CNNs always beat LSTMs.” It is:

> A carefully designed non-recurrent model can be competitive with strong recurrent language models while being easier to parallelize.

## 11. What did the ablations teach?

The authors vary individual design choices to discover why the model works.

### GLU versus other activations

They compare GLU with:

- ordinary `tanh`;
- ReLU;
- GTU, the tanh-plus-sigmoid gate;
- linear layers without gating.

GLU converges faster and reaches lower perplexity than the alternatives in their experiments. ReLU also has a direct path through active units, which helps optimization, but GLU performs better because its learned gate provides more flexible context-dependent control.

The result supports two separate conclusions:

1. A good gradient path matters.
2. Nonlinear, learned gating adds useful modeling power beyond a purely linear network.

### Why not only use linear convolutions?

If the layers are only:

$$
h_l(X)=X*W+b,
$$

then stacking them still gives a linear transformation before the final softmax. The network may see a broad context, but it cannot form rich nonlinear interactions.

The paper compares linear, bilinear, and GLU networks. Bilinear layers improve substantially over linear layers, and GLU improves further. The reported Google Billion Word perplexities are approximately:

```text
linear   -> 115
bilinear ->  around 74
GLU      -> 61
```

The exact curves matter less than the mechanism: the model needs both a path that is easy to optimize and nonlinear context-dependent behavior.

### Context size

Larger context generally improves performance, but gains diminish sharply after roughly 20 to 40 words in the reported experiments. Even on WikiText-103, where much longer context is available, strong performance is obtained with a context around 30 tokens.

This supports the paper’s challenge to a common assumption:

> The language model may not need an unlimited context to perform well on these benchmarks.

It does not prove that unlimited context is never useful. It shows that the finite context supplied by the tested GCNN was enough for these tasks.

### Optimization tricks

The paper uses:

- Nesterov momentum;
- gradient clipping;
- weight normalization;
- Kaiming initialization;
- relatively large learning rates selected by validation.

Gradient clipping is not only an RNN technique. The authors report that it helps their non-recurrent network too. Weight normalization and clipping make training faster and more stable, and weight normalization allows much larger learning rates in their experiments.

### Efficiency and responsiveness

The authors distinguish:

- **throughput:** tokens processed per second when many sequences are batched;
- **responsiveness:** how quickly one contiguous sequence is processed.

An RNN can have good GPU throughput when many independent sequences are batched, but it still processes the positions of one sequence sequentially. A GCNN can parallelize both across sequences and across positions within a sequence.

Their bottleneck GCNN achieves similar throughput to the LSTM but about 20 times higher responsiveness in the reported setting. This is especially relevant when a system must score one sentence quickly rather than process a huge batch.

## 12. Limitations and careful interpretation

The paper’s GCNN is not magic and not an exact replacement for every RNN.

1. **The context is finite.** If the receptive field is 30 tokens, information more than 30 tokens away cannot directly affect the prediction unless the architecture is changed.
2. **Depth increases context and cost.** A larger receptive field can require more layers or wider kernels.
3. **The model can overfit small datasets.** The authors note weaker behavior on the small Penn Treebank setup compared with its behavior on larger datasets.
4. **Convolution speed depends on implementation.** The paper notes that one-dimensional convolution kernels were not as optimized as cuDNN’s LSTM implementation at the time. The architectural parallelism and the actual library performance are related but not identical.
5. **Results depend on the output layer and hardware.** Adaptive softmax, GPU count, batch size, and implementation quality affect the comparison.
6. **The paper predates later architectures.** It should be understood as an important step in the history of parallel sequence models, not as a claim about modern Transformer performance.

## 13. A tiny end-to-end mental simulation

Take the input:

```text
<S> the cat sat
```

The training targets are shifted:

```text
input:  <S>   the   cat
target: the   cat   sat
```

The model performs the following:

1. Look up embeddings for `<S>`, `the`, and `cat`.
2. Causal padding makes sure the representation at each position sees only allowed past input.
3. The first convolution mixes local word features.
4. Its gate computes values between `0` and `1` and suppresses or passes the candidate features.
5. Several residual GLU blocks build larger-context representations.
6. At each position, the final hidden vector becomes vocabulary logits.
7. Softmax or adaptive softmax turns the logits into a next-word distribution.
8. Cross-entropy compares that distribution with `the`, `cat`, and `sat` at the corresponding positions.
9. Backpropagation updates the embeddings, convolutions, gates, residual blocks, and output parameters.

The important parallelism is that the representations for all three positions can be computed together during training, because the causal convolution uses a fixed pattern of allowed inputs rather than a hidden state that must be passed one position at a time.

## 14. A reading guide for the paper’s equations

When you read the paper line by line, keep this translation table nearby:

| Paper notation | Plain meaning |
|---|---|
| `X` | input sequence of feature vectors |
| `X * W + b` | candidate temporal convolution |
| `X * V + c` | pre-sigmoid gate convolution |
| `sigma(...)` | learned gate values in `[0, 1]` |
| `otimes` | element-wise multiplication |
| `k` | convolution kernel width |
| `m` | input feature-map/channel count |
| `n` | output feature-map/channel count |
| `H` | hidden representations for all positions |
| `D` | word embedding lookup table |
| `L` | number of layers or blocks, depending on context |
| GLU | gated linear unit: candidate times sigmoid gate |
| GTU | gated tanh unit: tanh candidate times sigmoid gate |
| AdaSoftmax | adaptive softmax for large vocabularies |

The paper’s whole method can be reconstructed from three lines:

$$
E=\operatorname{lookup}(w_0,\ldots,w_N)
$$

$$
h_l=(X*W_l+b_l)\otimes\sigma(X*V_l+c_l)
$$

$$
P(w_i\mid w_{<i})=\operatorname{softmax}(\operatorname{output}(H_i)).
$$

Everything else is about making these lines causal, deep, trainable, efficient, and competitive with recurrent baselines.

## 15. Final mental model

Imagine a team of fast readers examining a sentence. The first team looks at short word neighborhoods. The next team combines those local summaries into larger phrases. At each layer, a learned valve decides which features deserve to continue. Residual paths keep the information moving even when the network is deep. Because every reader can work on its allowed window at the same time, the team is parallel rather than a single reader waiting for the previous reader.

That is a gated convolutional network:

- **convolutional:** local windows are combined across the time dimension and stacked to create a larger finite context;
- **gated:** sigmoid-controlled multipliers decide which candidate features pass through;
- **network:** the blocks are stacked, residualized, and connected to a vocabulary prediction layer.

The paper’s main lesson is that language modeling does not necessarily require a recurrent chain. A deep, causal, gated convolutional hierarchy can capture enough context, learn effectively through a linear gradient path, and exploit modern parallel hardware.

## Source

- Local paper: `research-papers/3-Gated-Convolutional-Networks.pdf`
- Published version: [Dauphin et al. (2017), PMLR](https://proceedings.mlr.press/v70/dauphin17a.html)

# How much RNN/LSTM do I need to know?

No—you do not need to know the exact RNN or LSTM architecture first. Your basic idea is enough to start learning this paper.

You only need these concepts:

1. **RNNs process tokens sequentially**

   ```text
   word 1 -> hidden state 1 -> hidden state 2 -> hidden state 3
   ```

   The next prediction depends on the previous hidden state.

2. **This creates a bottleneck**

   Since token `i` depends on token `i-1`, RNNs cannot easily process all positions in parallel.

3. **RNNs can struggle with long dependencies**

   Information and gradients may weaken as they pass through many steps. LSTMs use gates to preserve or discard information more carefully.

4. **LSTM gates control information flow**

   This helps explain why the paper also uses the word “gated.” The GCNN gate serves a related high-level purpose—controlling what information passes—but its mechanism is much simpler than an LSTM’s.

The key comparison is:

| RNN/LSTM | Gated convolutional network |
|---|---|
| Processes positions sequentially | Processes positions in parallel |
| Uses a recurrent hidden state | Uses convolutional layers |
| Can theoretically see unlimited history | Has a finite receptive field |
| LSTM uses several gates | GCNN uses a simpler GLU gate |

You can understand the paper without memorizing LSTM equations. When the paper compares GLU with LSTM-style gating, you only need to know that LSTM-style gating uses nonlinear transformations such as `tanh` plus sigmoid gates, while GLU uses:

$$
\text{output} = \text{candidate} \times \text{sigmoid gate}
$$

A useful learning order is:

1. Understand the basic RNN computation and its sequential nature.
2. Understand the broad purpose of LSTM gates.
3. Learn one-dimensional causal convolution.
4. Learn the GLU equation.
5. Then study the full GCNN architecture.

So your current background is sufficient. You can learn the exact LSTM architecture later if you want to understand the experimental comparisons more deeply.
