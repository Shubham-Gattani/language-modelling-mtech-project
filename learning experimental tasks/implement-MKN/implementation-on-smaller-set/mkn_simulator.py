"""A small, readable Modified Kneser-Ney language-model simulator.

Flow:
    sentences
        -> add sentence boundaries
        -> count n-grams
        -> count distinct continuation histories
        -> estimate or accept D1/D2/D3+ discounts
        -> calculate recursive MKN probabilities
        -> print traces and normalization checks

The code intentionally favors explicit dictionaries and small functions over
performance-oriented abstractions. It is designed to be read while learning.
"""

from pathlib import Path
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

    The toy corpus uses whitespace tokenization. A later WikiText version
    should replace this function with a tokenizer appropriate for WikiText.
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

        history_count = self.history_count(history)
        if history_count == 0:
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


def load_corpus(path):
    """Load one non-empty sentence per line from a corpus file."""

    path = Path(path)
    sentences = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not sentences:
        raise ValueError(f"corpus file is empty: {path}")
    return sentences


def print_demo(model):
    """Print a learner-facing report for selected toy-corpus queries."""

    print("Modified Kneser-Ney toy simulation")
    print("==================================")
    print(f"sentences: {len(model.sentences)}")
    print(f"vocabulary size (excluding <s>): {len(model.vocabulary)}")
    print(f"distinct bigram types: {model.continuation_denominator}")
    print()

    print("discounts by order:")
    for order, discounts in model.discounts_by_order.items():
        print(f"  order {order}: D1={discounts[0]:.6f}, D2={discounts[1]:.6f}, D3+={discounts[2]:.6f}")
    print()

    selected_counts = [
        (2, ("the", "cat")),
        (3, ("the", "cat", "sat")),
        (2, ("cat", "fly")),
    ]
    print("selected counts:")
    for order, ngram in selected_counts:
        print(f"  c({ngram}) = {model.count(order, ngram)}")
    print()

    queries = [
        ("sat", ("the", "cat")),
        ("fly", ("cat",)),
    ]
    for word, history in queries:
        probability, trace = model.probability(word, history, return_trace=True)
        print(f"P({word} | {history}) = {probability:.10f}")
        for step in trace:
            print(f"  {step}")
        print()

    print("normalization checks:")
    histories = [(), ("the",), ("cat",), ("the", "cat")]
    for history in histories:
        total = model.normalization_total(history)
        status = "PASS" if math.isclose(total, 1.0, abs_tol=1e-9) else "FAIL"
        print(f"  {history}: total={total:.10f} [{status}]")


def main():
    corpus_path = Path(__file__).with_name("toy_corpus.txt") # Use the toy corpus file in the same directory as this script.

    # Read the file into a list such as:
    # ["the cat sat on the mat", "the cat sat on the rug", ...]
    sentences = load_corpus(corpus_path)

    # This class-method call creates the model. Internally, it calls
    # MKNModel(sentences, max_order=3), which invokes __init__.
    model = MKNModel.from_sentences(sentences, max_order=3)
    print_demo(model)


if __name__ == "__main__":
    main()
