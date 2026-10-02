"""A readable Modified Kneser-Ney experiment on WikiText-2.

Flow:
    locally saved WikiText-2 train rows
        -> keep one non-empty row as one modeling sequence
        -> add sentence boundaries
        -> count n-grams
        -> count distinct continuation histories
        -> estimate or accept D1/D2/D3+ discounts
        -> calculate recursive MKN probabilities
    -> evaluate validation perplexity
    -> print traces and small normalization checks

TODO:
    Use the test split for final perplexity evaluation after the implementation
    and settings have been finalized. Validation is currently used only as a
    learning-time evaluation set.

The code intentionally favors explicit dictionaries and small functions over
performance-oriented abstractions. It is designed to be read while learning.
"""

import argparse
import math

START_TOKEN = "<s>"
END_TOKEN = "</s>"


TOY_SENTENCES = [
    "the cat sat on the mat",
    "the cat sat on the rug",
    "the cat sat near the mat",
    "the dog sat on the mat",
    "the dog sat on the rug",
    "the dog sat near the mat",
    "a cat sat on the mat",
    "a cat sat on the rug",
    "a dog sat on the mat",
    "a dog sat on the rug",
    "birds fly over the mat",
    "birds fly over the rug",
]


def tokenize_sentence(sentence):
    """Return one sentence with explicit boundary tokens.

    Example:
        ``tokenize_sentence("the cat")`` returns
        ``["<s>", "the", "cat", "</s>"]``. 

    WikiText-2's word-level files are already tokenized with spaces. This
    experiment therefore keeps whitespace tokenization so that preprocessing
    remains visible and reproducible.
    """

    words = sentence.split()
    if not words:
        raise ValueError("sentence must contain at least one token")
    if words[0] == START_TOKEN or words[-1] == END_TOKEN:
        raise ValueError("input sentences must not already contain boundary tokens")
    return [START_TOKEN, *words, END_TOKEN]


def estimate_discounts(ngram_counts):
    """Estimate the standard three Modified Kneser-Ney discounts.

    `ngram_counts` contains one count for every distinct n-gram type.   e.g.
    for the bigrams in “the cat sat” and “the cat slept”:

    {
        ("<s>", "the"): 2,
        ("the", "cat"): 2,
        ("cat", "sat"): 1,
        ("sat", "</s>"): 1,
        ("cat", "slept"): 1,
        ("slept", "</s>"): 1,
    }

    ngram_counts would be:

    [2, 2, 1, 1, 1, 1]

    Each number is the frequency of one distinct bigram—not the count of all bigram occurrences combined.
    
    The returned tuple is ``(D1, D2, D3_plus)``. If a tiny corpus does not contain
    enough count classes for one of the formula terms, a clearly documented
    teaching fallback is used for that term.

    n1: number of n-gram types seen exactly once (4 in above example)
    n2: number of n-gram types seen exactly twice (2 in above example)
    n3: number of n-gram types seen exactly three times (0 in above example)
    n4: number of n-gram types seen exactly four times (0 in above example)
    """

    # Count how many n-gram types have each count. For example, if three
    # different n-grams occur once, frequency_of_frequencies[1] becomes 3.
    frequency_of_frequencies = {}
    for count in ngram_counts:
        frequency_of_frequencies[count] = frequency_of_frequencies.get(count, 0) + 1

    n1 = frequency_of_frequencies.get(1, 0)
    n2 = frequency_of_frequencies.get(2, 0)
    n3 = frequency_of_frequencies.get(3, 0)
    n4 = frequency_of_frequencies.get(4, 0)

    if n1 and n2:
        y = n1 / (n1 + 2 * n2)
        d1 = 1 - 2 * y * n2 / n1
    else:
        # Small teaching corpora can lack singleton or doubleton types.
        y = 0.5
        d1 = 0.5

    if n2 and n3:
        d2 = 2 - 3 * y * n3 / n2
    else:
        d2 = 1.0

    if n3 and n4:
        d3_plus = 3 - 4 * y * n4 / n3
    else:
        d3_plus = 1.5

    discounts = (d1, d2, d3_plus)
    if any(discount <= 0 for discount in discounts):
        raise ValueError(f"estimated discounts must be positive: {discounts}")
    return discounts


def discount_for_count(count, discounts):
    """Choose D1, D2, or D3+ for one observed count. e.g. discount_for_count(2, (0.5, 0.75, 1.0)) returns 0.75.`count` here means the number of times a particular n-gram was observed in the training corpus. The `discounts` tuple contains the three discount values (D1, D2, D3+) for the corresponding n-gram order. The function returns the appropriate discount value based on the observed count.
    For example, if an n-gram was observed twice in the training corpus, the function will return the D2 discount value from the `discounts` tuple.
    """

    if count <= 0:
        return 0.0
    if count == 1:
        return discounts[0]
    if count == 2:
        return discounts[1]
    return discounts[2] # For counts of 3 or more, the function returns the D3+ discount value from the `discounts` tuple.


class MKNModel:
    """An in-memory interpolated Modified Kneser-Ney model.

    A model stores counts for orders 1 through ``max_order``. For a query such
    as ``probability("sat", ("the", "cat"))``, it first uses the trigram
    count, then recursively asks the bigram model for ``P(sat | cat)``.
    """

    def __init__(self, sentences, max_order=3, discounts_by_order=None):
        # The constructor receives the training sentences and immediately
        # builds all information the model needs: n-gram counts, vocabulary,
        # continuation histories, and discount values.
        if max_order < 2:
            raise ValueError("max_order must be at least 2")
        if not sentences:
            raise ValueError("at least one sentence is required")

        self.max_order = max_order
        self.sentences = list(sentences)
        self.tokenized_sentences = [tokenize_sentence(sentence) for sentence in sentences]
        self.vocabulary = self._build_vocabulary() # Stores all words in the corpus, excluding the start token.
        self.ngram_counts = self._build_ngram_counts() # Stores counts and actual n-grams for orders 1,2,...max_order.
        self.predecessors_by_word = self._build_predecessor_sets() # Maps each word to the set of distinct words that precede it in the bigrams.
        self.continuation_denominator = len(self.ngram_counts[2]) # The number of distinct bigram types, used as the denominator for continuation probabilities; P_KN(word) = number of distinct preceding words / total number of distinct bigrams
        self.discounts_by_order = self._build_discounts(discounts_by_order) # Stores the D1, D2, D3+ discount values for each order, either estimated from the counts or supplied by the user.

    @classmethod
    def from_sentences(cls, sentences, max_order=3, discounts_by_order=None):
        """Build a model from one whitespace-tokenized sentence per item."""

        # `cls` means MKNModel here. Therefore this line calls:
        # MKNModel(sentences, max_order, discounts_by_order)
        # and that constructor then calls __init__ with the sentences.
        return cls(sentences, max_order, discounts_by_order)

    def _build_ngram_counts(self):
        counts_by_order = {
            order: {} for order in range(1, self.max_order + 1)
        }

        for tokens in self.tokenized_sentences:
            for order in counts_by_order:
                for start in range(len(tokens) - order + 1):
                    ngram = tuple(tokens[start : start + order])
                    counts = counts_by_order[order]
                    counts[ngram] = counts.get(ngram, 0) + 1
        """ Example of "counts_by_order" for the toy corpus:
        counts_by_order = {
                1: {
                    ("the",): 4,
                    ("cat",): 2,
                    ("sat",): 2,
                },
                2: {
                    ("the", "cat"): 2,
                    ("cat", "sat"): 2,
                    ("the", "dog"): 2,
                },
                3: {
                    ("the", "cat", "sat"): 2,
                    ("the", "dog", "sat"): 2,
                }
            }
        """
        return counts_by_order

    def _build_vocabulary(self):
        vocabulary = set()
        for tokens in self.tokenized_sentences:
            vocabulary.update(tokens) # Add all tokens from the sentence to the vocabulary set.
        vocabulary.discard(START_TOKEN) # Remove the start token from the vocabulary, as it is not a word we want to predict.
        return tuple(sorted(vocabulary)) # Return the vocabulary as a sorted tuple for consistent ordering.

    def _build_predecessor_sets(self):
        """
        Build a dictionary mapping each word to the set of distinct words that precede it in the bigrams. The method iterates over all bigrams in the n-gram counts and populates the predecessors dictionary accordingly. e.g.
        predecessors = {
                        "cat": {"the", "a"},
                        "sat": {"cat", "dog"},
                        "dog": {"the"},
                        "slept": {"cat"},
                        }
        """
        predecessors = {}
        for bigram in self.ngram_counts[2]:
            previous_word, word = bigram
            if word not in predecessors:
                predecessors[word] = set()
            predecessors[word].add(previous_word)
        return predecessors

    def _build_discounts(self, supplied_discounts):
        discounts = {}
        supplied_discounts = supplied_discounts or {}

        for order in range(2, self.max_order + 1): # order = 2, 3 in our case since we kept max_order = 3
            if order in supplied_discounts:  # Case 1: The user supplies discounts
                values = tuple(supplied_discounts[order])
                if len(values) != 3 or any(value <= 0 for value in values):
                    # every value must be positive; there must be exactly three values
                    raise ValueError("each discount tuple must contain three positive values")
                discounts[order] = values
            else:
                # Case 2: Estimate discounts from the n-gram counts
                discounts[order] = estimate_discounts(self.ngram_counts[order].values())
        """
        For a trigram model (max_order = 3), an example final value is:
        discounts = {
            2: (0.5, 1.625, 2.0),  # bigram: D1, D2, D3+
            3: (0.1818, 1.7575, 2.2727),   # trigram: D1, D2, D3+
        }
        discounts[3] contains the corresponding discounts for trigrams. Why haven't we included 1 and 4 as the keys in the discounts dictionary? Because the model only needs discounts for n-gram orders 2 through max_order:
            for order in range(2, self.max_order + 1):
                - Key 1 is excluded because unigrams use the continuation probability, not discounting.
                - Key 4 is excluded because this model has max_order = 3; it is only a trigram model.
        """

        return discounts

    def count(self, order, ngram):
        """Return the count of one n-gram. e.g. count(3, ("the", "cat", "sat")) returns 2 for the toy corpus."""

        if order not in self.ngram_counts:
            raise ValueError(f"unsupported n-gram order: {order}")
        ngram = tuple(ngram)
        if len(ngram) != order:
            raise ValueError(f"expected {order} tokens, got {len(ngram)}")
        # A missing n-gram was never observed, so its count is zero.
        return self.ngram_counts[order].get(ngram, 0)

    def continuation_count(self, word):
        """Return the number of distinct words that precede word in the bigrams. e.g. continuation_count("sat") returns 2 for the toy corpus because "cat" and "dog" precede "sat" in the bigrams."""

        self._check_word(word)
        return len(self.predecessors_by_word[word])

    def continuation_probability(self, word):
        """Return the Kneser-Ney continuation probability of ``word``. e.g. continuation_probability("sat") returns 0.5 for the toy corpus because "cat" and "dog" each precede "sat" in the bigrams, and there are 4 bigrams total."""

        self._check_word(word)
        if self.continuation_denominator == 0:
            raise ValueError("cannot calculate continuation probability without bigrams")
        return self.continuation_count(word) / self.continuation_denominator

    def discount_for_count(self, count, order):
        """Return the discount bucket used by one observed n-gram count. e.g. discount_for_count(2, 3) returns the D2 discount for trigrams."""

        if order not in self.discounts_by_order:
            raise ValueError(f"discounts are not defined for order {order}")
        return discount_for_count(count, self.discounts_by_order[order])

    def history_count(self, history):
        """Return the count of a history of length 1 through max_order - 1. e.g. history_count(("the", "cat")) returns 2 for the toy corpus because the bigram ("the", "cat") occurs twice."""

        history = tuple(history)
        if not history:
            return 0
        return self.count(len(history), history)

    def lambda_for_history(self, history):
        """Return the total probability mass removed after an observed history. e.g. lambda_for_history(("the", "cat")) returns 0.5 for the toy corpus because the bigram ("the", "cat") is discounted by 0.5."""

        history = tuple(history)
        if not history:
            raise ValueError("the empty history has no higher-order lambda")
        order = len(history) + 1
        if order > self.max_order:
            raise ValueError("history is longer than the model supports")

        history_count = self.history_count(history) # Count of the history e.g. history_count(("the", "cat")) returns 2 for the toy corpus because the bigram ("the", "cat") occurs twice.
        if history_count == 0: # If the history was never observed, the model backs off to a lower order and assigns all probability mass to that lower order. Therefore, the lambda for an unseen history is 1.0.
            return 1.0

        removed_count = 0.0
        for ngram, count in self.ngram_counts[order].items():
            if ngram[:-1] == history:
                removed_count += self.discount_for_count(count, order)
        return removed_count / history_count

    def probability(self, word, history=(), return_trace=False):
        """Calculate P(word | history) and optionally return a trace.

        Example:
            ``model.probability("sat", ("the", "cat"))`` calculates a
            trigram probability and recursively uses a bigram if needed.
        """

        self._check_word(word)
        history = tuple(history)
        if len(history) >= self.max_order:
            raise ValueError("history is longer than the supported n-gram order")

        trace = []
        probability = self._probability(word, history, trace)
        if return_trace:
            return probability, trace
        return probability

    def _probability(self, word, history, trace):
        if not history:
            probability = self.continuation_probability(word)
            trace.append(
                {
                    "order": 1,
                    "history": (),
                    "event": "continuation_probability",
                    "direct_contribution": 0.0,
                    "lambda": 1.0,
                    "lower_order_probability": probability,
                    "probability": probability,
                }
            )
            return probability

        order = len(history) + 1
        history_count = self.history_count(history)
        if history_count == 0:
            lower_history = history[1:]
            step = {
                "order": order,
                "history": history,
                "event": "unseen_history_backoff",
                "direct_contribution": 0.0,
                "lambda": 1.0,
            }
            trace.append(step)
            lower_probability = self._probability(word, lower_history, trace)
            step["lower_order_probability"] = lower_probability
            step["probability"] = lower_probability
            return lower_probability

        ngram = history + (word,)
        count = self.count(order, ngram)
        selected_discount = self.discount_for_count(count, order)
        direct_contribution = 0.0
        if count > 0:
            direct_contribution = max(count - selected_discount, 0.0) / history_count

        interpolation_weight = self.lambda_for_history(history)
        event = "seen_history_seen_ngram" if count > 0 else "seen_history_unseen_ngram"
        step = {
            "order": order,
            "history": history,
            "event": event,
            "ngram": ngram,
            "count": count,
            "selected_discount": selected_discount,
            "history_count": history_count,
            "direct_contribution": direct_contribution,
            "lambda": interpolation_weight,
        }
        trace.append(step)

        lower_probability = self._probability(word, history[1:], trace)
        probability = direct_contribution + interpolation_weight * lower_probability
        step["lower_order_probability"] = lower_probability
        step["probability"] = probability
        return probability

    def normalization_total(self, history=()):
        """Sum probabilities over the model vocabulary for one history."""

        return sum(self.probability(word, history) for word in self.vocabulary)

    def _check_word(self, word):
        if word not in self.vocabulary:
            raise ValueError(f"word is not in the vocabulary: {word!r}")


def load_saved_huggingface_splits(dataset_path, dataset_loader=None):
    """Load a locally saved Hugging Face DatasetDict.

    ``dataset_loader`` is optional so tests can provide a small fake dataset
    without needing a local Hugging Face download. Normal use imports and
    calls ``datasets.load_from_disk`` here.
    """

    if dataset_loader is None:
        try:
            from datasets import load_from_disk
        except ImportError as error:
            raise RuntimeError(
                "the local Hugging Face loader requires the 'datasets' package; "
                "install it with: python3 -m pip install -r requirements.txt"
            ) from error
        dataset_loader = load_from_disk

    dataset = dataset_loader(dataset_path)
    required_splits = ("train", "validation", "test")
    for split_name in required_splits:
        if split_name not in dataset:
            raise ValueError(
                f"saved Hugging Face dataset is missing the {split_name!r} split"
            )

    splits = {}
    for split_name in required_splits:
        sentences = []
        for row in dataset[split_name]:
            if "text" not in row:
                raise ValueError(
                    f"row in {split_name!r} split has no 'text' field"
                )
            text = row["text"].strip()
            if text:
                sentences.append(text)
        if not sentences:
            raise ValueError(
                f"saved Hugging Face {split_name!r} split contains no text"
            )
        splits[split_name] = sentences

    return splits


def take_first_lines(sentences, limit):
    """Return all lines when ``limit`` is zero, otherwise return a prefix."""

    if limit < 0:
        raise ValueError("line limit must be zero or positive")
    if limit == 0:
        return sentences
    return sentences[:limit]


def evaluation_words(sentence, vocabulary):
    """Replace validation words unseen in training with ``<unk>``.

    The non-raw WikiText files use ``<unk>`` for word-level unknowns. Keeping
    this replacement explicit prevents validation data from expanding the
    training vocabulary.
    """

    words = sentence.split()
    if "<unk>" not in vocabulary:
        unknown_words = [word for word in words if word not in vocabulary]
        if unknown_words:
            raise ValueError(
                "validation contains out-of-vocabulary words, but the training "
                "vocabulary has no <unk> token"
            )
        return words
    return [word if word in vocabulary else "<unk>" for word in words]


def validation_perplexity(model, sentences):
    """Calculate word-level perplexity using only the model's vocabulary.

    The first real word in each line is predicted from ``<s>``. The end token
    is also evaluated, so every non-empty modeling line contributes one
    boundary prediction.
    """

    total_log_probability = 0.0
    predicted_tokens = 0

    for sentence in sentences:
        words = evaluation_words(sentence, model.vocabulary)
        if not words:
            continue
        tokens = [START_TOKEN, *words, END_TOKEN]
        for position in range(1, len(tokens)):
            history_start = max(0, position - (model.max_order - 1))
            history = tuple(tokens[history_start:position])
            word = tokens[position]
            probability = model.probability(word, history)
            if probability <= 0.0:
                raise ValueError(f"model assigned zero probability to {word!r}")
            total_log_probability += math.log(probability)
            predicted_tokens += 1

    if predicted_tokens == 0:
        raise ValueError("validation corpus contains no predictable tokens")
    return math.exp(-total_log_probability / predicted_tokens), predicted_tokens


def print_demo(model, validation_sentences):
    """Print a learner-facing report for WikiText-2 and selected queries."""

    print("Modified Kneser-Ney WikiText-2 experiment")
    print("=========================================")
    print(f"sentences: {len(model.sentences)}")
    print(f"vocabulary size (excluding <s>): {len(model.vocabulary)}")
    print(f"distinct bigram types: {model.continuation_denominator}")
    print()

    print("discounts by order:")
    for order, discounts in model.discounts_by_order.items():
        print(f"  order {order}: D1={discounts[0]:.6f}, D2={discounts[1]:.6f}, D3+={discounts[2]:.6f}")
    print()

    selected_counts = [(2, ("the", "cat")), (3, ("the", "cat", "sat"))]
    print("selected counts:")
    for order, ngram in selected_counts:
        print(f"  c({ngram}) = {model.count(order, ngram)}")
    print()

    example_word = "the" if "the" in model.vocabulary else model.vocabulary[0]
    queries = [(example_word, ()), (example_word, (example_word,))]
    for word, history in queries:
        probability, trace = model.probability(word, history, return_trace=True)
        print(f"P({word} | {history}) = {probability:.10f}")
        for step in trace:
            print(f"  {step}")
        print()

    perplexity, predicted_tokens = validation_perplexity(model, validation_sentences)
    print(f"validation predicted tokens: {predicted_tokens}")
    print(f"validation perplexity: {perplexity:.6f}")
    print()

    print("normalization checks on selected histories:")
    histories = [(), ("the",), ("of",), ("of", "the")]
    for history in histories:
        if any(token not in model.vocabulary for token in history):
            continue
        total = model.normalization_total(history)
        status = "PASS" if math.isclose(total, 1.0, abs_tol=1e-9) else "FAIL"
        print(f"  {history}: total={total:.10f} [{status}]")


def main():
    parser = argparse.ArgumentParser(
        description="Run MKN on a locally saved WikiText-2 dataset."
    )
    parser.add_argument(
        "--dataset-path",
        default="data/wikitext-2",
        help="path created by download_wikitext2.py",
    )
    parser.add_argument(
        "--max-train-lines",
        type=int,
        default=5000,
        help="number of training lines; use 0 for the full training file",
    )
    parser.add_argument(
        "--max-valid-lines",
        type=int,
        default=500,
        help="number of validation lines; use 0 for the full validation file",
    )
    args = parser.parse_args()

    dataset_splits = load_saved_huggingface_splits(args.dataset_path)
    train_sentences = take_first_lines(
        dataset_splits["train"], args.max_train_lines
    )
    validation_sentences = take_first_lines(
        dataset_splits["validation"], args.max_valid_lines
    )

    # This creates the model and immediately builds counts, predecessor sets,
    # discounts, and recursive probability information.
    model = MKNModel.from_sentences(train_sentences, max_order=3)
    print_demo(model, validation_sentences)


if __name__ == "__main__":
    main()
