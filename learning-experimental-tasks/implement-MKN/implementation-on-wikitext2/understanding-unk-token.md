# Understanding the `<unk>` Token

This note explains how the `<unk>` token is used when evaluating validation
text. The focus is only on the validation workflow—not on calculating MKN
probabilities.

## 1. The problem: validation may contain new words

The model builds its vocabulary from the training data. Suppose the training
corpus contains:

```text
the cat sat
the dog ran
the <unk> slept
```

The model's vocabulary includes:

```text
the, cat, sat, dog, ran, <unk>, slept, </s>
```

It does not include every word that could ever exist. For example, `dragon`
is not present in this training vocabulary.

Now suppose a validation row is:

```text
the dragon slept
```

The model cannot ask for a probability of a word it does not know as a
vocabulary item:

```text
P(dragon | the)
```

The model has no `dragon` entry in its vocabulary or probability tables.

## 2. Replace an unknown word with `<unk>`

Because `<unk>` is present in the training vocabulary, the validation
preprocessing replaces `dragon` with `<unk>`:

```text
original validation row:
    the dragon slept

after unknown-word replacement:
    the <unk> slept
```

Then the boundary tokens are added:

```text
[<s>, the, <unk>, slept, </s>]
```

The model evaluates this transformed sequence, not the original sequence with
the unknown word `dragon`.

The interpretation is:

```text
<unk> means “some word that was not represented individually in training.”
```

It does not mean that the model believes the real word was literally the word
`<unk>` in the original text. It is a safe vocabulary label used for unknown
words.

## 3. What prediction events are created?

For the transformed validation sequence:

```text
[<s>, the, <unk>, slept, </s>]
```

the model creates these prediction events:

```text
1. predict "the"   using history (<s>,)
2. predict "<unk>"  using history (<s>, the)
3. predict "slept"  using history (the, <unk>)
4. predict "</s>"   using history (<unk>, slept)
```

Notice what happened to `dragon`:

```text
original target:   dragon
model target:      <unk>
```

The model is evaluated on its ability to assign probability to the `<unk>`
category, not to the specific unseen word `dragon`.

## 4. Dummy probabilities for the prediction events

We are not deriving these probabilities here. Assume the already-trained MKN
model returns the following values:

```text
P(the | <s>)             = 5/37
P(<unk> | <s>, the)      = 7/41
P(slept | the, <unk>)    = 3/29
P(</s> | <unk>, slept)   = 11/43
```

The evaluation code simply records these probabilities and includes their log
values in the validation score. It does not try to calculate a separate
probability for `dragon`.

The important mapping is:

```text
validation target dragon
    -> mapped target <unk>
    -> ask the model for P(<unk> | history)
```

## 5. Unknown words can also become part of the history

The unknown token is not only used as a target. After replacement, it can also
appear in later histories.

In our example:

```text
[<s>, the, <unk>, slept, </s>]
```

When predicting `slept`, the history is:

```text
(the, <unk>)
```

So the model asks for:

```text
P(slept | the, <unk>)
```

This is useful because the training data may have learned patterns involving
unknown words, such as:

```text
the <unk> slept
```

The `<unk>` token therefore behaves like an ordinary vocabulary token during
inference. Its special role is that it represents many different real words
that were not kept as individual vocabulary entries.

## 6. Several different unknown words collapse to one token

Suppose the validation set contains:

```text
the dragon slept
the spaceship slept
the unicorn slept
```

If none of `dragon`, `spaceship`, or `unicorn` is in the training vocabulary,
the three rows become:

```text
the <unk> slept
the <unk> slept
the <unk> slept
```

The model does not distinguish which unknown word appeared. All three are
represented by the same symbol:

```text
dragon    -> <unk>
spaceship -> <unk>
unicorn   -> <unk>
```

This is a deliberate loss of detail. The model can say “an unknown word
occurred here,” but it cannot say which unknown word it was.

## 7. Why not add validation words to the vocabulary?

It may seem tempting to do this:

```text
training vocabulary:     the, cat, dog, ...
validation word:         dragon
new vocabulary:          the, cat, dog, dragon, ...
```

But that would create a problem. The model would now contain a word that was
not part of its training vocabulary, but it would still have no reliable
training statistics for that word.

More importantly, the validation data would be influencing the model's
vocabulary. That is a form of information leakage.

The safer rule is:

```text
build vocabulary from training data only
keep it fixed during validation
map unseen validation words to <unk>
```

## 8. What if `<unk>` is not in the training vocabulary?

The current code has a deliberate second branch.

### Case A: `<unk>` is in the vocabulary

```text
unknown validation word -> <unk>
```

The model can evaluate the transformed validation row.

### Case B: `<unk>` is not in the vocabulary

The code cannot safely replace an unknown word, because the replacement token
itself is unknown to the model. It raises an error instead of silently
pretending that the model can score the word.

Conceptually:

```text
unknown validation word + no <unk> token
    -> no valid vocabulary representation
    -> stop with a clear error
```

This is preferable to producing a misleading perplexity value.

## 9. Why WikiText-2 can use this approach

The selected non-raw WikiText-2 configuration already uses `<unk>` for words
outside its word vocabulary. Therefore, `<unk>` is an expected part of the
word-level modeling setup.

The flow is:

```text
WikiText-2 training rows
    -> build vocabulary, including <unk>

WikiText-2 validation row
    -> words known during training stay unchanged
    -> words unknown during training become <unk>
    -> MKN predicts the resulting token sequence
```

## 10. What `<unk>` does and does not mean

`<unk>` does mean:

```text
the original word was not available as an individual model vocabulary item
```

`<unk>` does not mean:

```text
the original text literally contained the characters <unk>
```

It is a modeling category. Many different real words may map to it.

## 11. Connection to perplexity

For the transformed validation row:

```text
[<s>, the, <unk>, slept, </s>]
```

the perplexity calculation counts four prediction events:

```text
P(the | <s>)
P(<unk> | <s>, the)
P(slept | the, <unk>)
P(</s> | <unk>, slept)
```

The unknown word contributes through the second event:

```text
P(<unk> | <s>, the)
```

The original string `dragon` is not passed directly to the model after the
replacement step.

## Final mental model

Think of `<unk>` as a single reserved drawer labelled:

```text
“words that this model does not know individually”
```

During validation:

```text
known word
    -> keep the original word

unknown word
    -> put it in the <unk> drawer

then
    -> evaluate the resulting sequence using the fixed training vocabulary
```

This lets validation contain genuinely new words without changing the model
after training or causing an undefined probability lookup.
