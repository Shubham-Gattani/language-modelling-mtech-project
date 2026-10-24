"""A tiny NumPy implementation of the Bengio et al. (2003) model.

Flow:
    word strings -> word-vector lookup -> concatenated x
    -> z = d + Hx -> h = tanh(z)
    -> y = b + Wx + Uh -> stable softmax probabilities
    -> negative log-likelihood -> manual backpropagation
    -> gradient-descent update of C, H, d, W, U, and b

This is an educational implementation of the Section 2 equations.  It is
deliberately small and explicit; it is not a reproduction of the paper's
large-scale Brown-corpus experiment.

Example:
    $ python3 bengio_numpy_toy.py
    The script prints the first forward-pass values and training progress.
"""

import numpy as np


# Project-specific terms:
# - C: word-feature table; one row is the vector for one vocabulary word.
# - x: concatenation of the context-word rows from C.
# - y: output logits, one value per possible next word.


VOCAB = ["the", "cat", "dog", "sat", "<UNK>"]
WORD_TO_ID = {word: index for index, word in enumerate(VOCAB)}
CONTEXT_SIZE = 2
WORD_VECTOR_SIZE = 3
HIDDEN_SIZE = 4


def make_parameters():
    """Create the deterministic toy parameters used in the worksheet."""

    C = np.array(
        [
            [0.10, 0.20, 0.30],  # the
            [0.20, 0.10, 0.40],  # cat
            [0.30, 0.20, 0.10],  # dog
            [0.40, 0.30, 0.20],  # sat
            [0.00, 0.10, 0.00],  # <UNK>
        ],
        dtype=float,
    )

    H = np.array( # context-to-hidden weights
        [
            [0.10, 0.00, 0.00, 0.10, 0.00, 0.00],
            [0.00, 0.10, 0.00, 0.00, 0.10, 0.00],
            [0.00, 0.00, 0.10, 0.00, 0.00, 0.10],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        ],
        dtype=float,
    )
    d = np.zeros(HIDDEN_SIZE, dtype=float) # hidden bias

    b = np.array([0.010, 0.000, -0.010, 0.020, 0.000], dtype=float) # output bias
    W = np.array( # input-to-output weights
        [
            [0.10, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.10, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.10, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.10, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.10, 0.00],
        ],
        dtype=float,
    )
    U = np.array( # hidden-to-output weights
        [
            [0.10, 0.00, 0.00, 0.00],
            [0.00, 0.10, 0.00, 0.00],
            [0.00, 0.00, 0.10, 0.00],
            [0.00, 0.00, 0.00, 0.10],
            [-0.10, 0.00, 0.00, 0.00],
        ],
        dtype=float,
    )

    return {"C": C, "H": H, "d": d, "b": b, "W": W, "U": U}


def word_id(word):
    """Map an unseen word to the shared ``<UNK>`` row."""

    return WORD_TO_ID.get(word, WORD_TO_ID["<UNK>"])


def stable_softmax(logits):
    """Convert logits to probabilities without exponential overflow."""

    shifted_logits = logits - np.max(logits)
    exponentials = np.exp(shifted_logits)
    return exponentials / np.sum(exponentials)


def forward(parameters, context_words):
    """Run one context through the network and return all useful intermediates."""

    if len(context_words) != CONTEXT_SIZE:
        raise ValueError(f"context must contain exactly {CONTEXT_SIZE} words")

    context_ids = np.array([word_id(word) for word in context_words], dtype=int)
    context_vectors = parameters["C"][context_ids]
    x = np.concatenate(context_vectors)
    z = parameters["d"] + parameters["H"] @ x
    h = np.tanh(z)
    y = parameters["b"] + parameters["W"] @ x + parameters["U"] @ h
    probabilities = stable_softmax(y)

    return {
        "context_ids": context_ids,
        "x": x,
        "z": z,
        "h": h,
        "y": y,
        "probabilities": probabilities,
    }


def loss_and_gradients(parameters, context_words, target_word):
    """Compute NLL(i.e. negative log-likelihood) and every parameter gradient for one training example.

    The backward flow is:
        delta_y -> (b, W, U) -> delta_z -> (d, H) -> dL/dx -> C
    """

    values = forward(parameters, context_words)
    target_id = word_id(target_word)
    probabilities = values["probabilities"]
    loss = -np.log(probabilities[target_id])

    target = np.zeros(len(VOCAB), dtype=float)
    target[target_id] = 1.0
    delta_y = probabilities - target

    gradient_b = delta_y
    gradient_W = np.outer(delta_y, values["x"])
    gradient_U = np.outer(delta_y, values["h"])

    d_loss_dh = parameters["U"].T @ delta_y
    delta_z = d_loss_dh * (1.0 - values["h"] ** 2)
    gradient_d = delta_z
    gradient_H = np.outer(delta_z, values["x"])

    d_loss_dx = parameters["W"].T @ delta_y + parameters["H"].T @ delta_z
    gradient_C = np.zeros_like(parameters["C"])
    for position, context_id in enumerate(values["context_ids"]):
        start = position * WORD_VECTOR_SIZE
        end = start + WORD_VECTOR_SIZE
        gradient_C[context_id] += d_loss_dx[start:end]

    gradients = {
        "C": gradient_C,
        "H": gradient_H,
        "d": gradient_d,
        "b": gradient_b,
        "W": gradient_W,
        "U": gradient_U,
    }
    return loss, values, gradients


def update(parameters, gradients, learning_rate):
    """Apply one ordinary gradient-descent update to every parameter."""

    for name in parameters:
        parameters[name] -= learning_rate * gradients[name]


def check_shapes(parameters):
    """Fail early if an array no longer matches the intended architecture."""

    expected_shapes = {
        "C": (len(VOCAB), WORD_VECTOR_SIZE),
        "H": (HIDDEN_SIZE, CONTEXT_SIZE * WORD_VECTOR_SIZE),
        "d": (HIDDEN_SIZE,),
        "b": (len(VOCAB),),
        "W": (len(VOCAB), CONTEXT_SIZE * WORD_VECTOR_SIZE),
        "U": (len(VOCAB), HIDDEN_SIZE),
    }
    for name, expected_shape in expected_shapes.items():
        assert parameters[name].shape == expected_shape, (
            f"{name} has shape {parameters[name].shape}; "
            f"expected {expected_shape}"
        )


def print_first_forward_pass(parameters):
    """Print the values corresponding to the worksheet's first forward pass."""

    loss, values, gradients = loss_and_gradients(parameters, ["the", "cat"], "sat")
    probabilities = values["probabilities"]

    print("First forward pass: context = ['the', 'cat'], target = 'sat'")
    print("x =", np.round(values["x"], 6))
    print("z =", np.round(values["z"], 6))
    print("h =", np.round(values["h"], 6))
    print("y =", np.round(values["y"], 6))
    print("probabilities =", np.round(probabilities, 6))
    print("P(sat) =", round(probabilities[word_id("sat")], 6))
    print("loss =", round(loss, 6))
    print("gradient_C rows used = the, cat")
    print("gradient_C[the] =", np.round(gradients["C"][word_id("the")], 6))
    print("gradient_C[cat] =", np.round(gradients["C"][word_id("cat")], 6))


def train(parameters, examples, steps=200, learning_rate=0.1):
    """Train on the tiny examples and return loss/probability history."""

    history = []
    for step in range(steps):
        total_loss = 0.0
        for context_words, target_word in examples:
            loss, _, gradients = loss_and_gradients(
                parameters, context_words, target_word
            )
            total_loss += loss
            update(parameters, gradients, learning_rate)

        if step in {0, steps - 1}:
            history.append(total_loss / len(examples))
    return history


def main():
    parameters = make_parameters()
    check_shapes(parameters)
    print_first_forward_pass(parameters)

    examples = [
        (["the", "cat"], "sat"),
        (["the", "dog"], "sat"),
    ]
    before = loss_and_gradients(parameters, *examples[0])[1]["probabilities"][word_id("sat")]
    losses = train(parameters, examples)
    after = loss_and_gradients(parameters, *examples[0])[1]["probabilities"][word_id("sat")]

    assert np.isclose(np.sum(loss_and_gradients(parameters, *examples[0])[1]["probabilities"]), 1.0)
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]
    assert after > before

    print("\nTraining demonstration")
    print("average loss at step 1   =", round(losses[0], 6))
    print("average loss at step 200 =", round(losses[-1], 6))
    print("P(sat | the cat) before =", round(before, 6))
    print("P(sat | the cat) after  =", round(after, 6))
    print("All shape, probability, finiteness, and learning checks passed.")


if __name__ == "__main__":
    main()
