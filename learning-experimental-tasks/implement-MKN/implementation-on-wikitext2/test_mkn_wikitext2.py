import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from mkn_wikitext2 import (
    END_TOKEN,
    START_TOKEN,
    MKNModel,
    estimate_discounts,
    evaluation_words,
    load_saved_huggingface_splits,
    take_first_lines,
    tokenize_sentence,
    validation_perplexity,
)


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


TEST_DISCOUNTS = {
    2: (0.5, 0.75, 1.0),
    3: (0.5, 0.75, 1.0),
}


class MKNModelTests(unittest.TestCase):
    def setUp(self):
        self.model = MKNModel.from_sentences(
            TOY_SENTENCES,
            max_order=3,
            discounts_by_order=TEST_DISCOUNTS,
        )

    def test_tokenizer_adds_sentence_boundaries(self):
        tokens = tokenize_sentence("the cat sat")

        self.assertEqual(tokens, [START_TOKEN, "the", "cat", "sat", END_TOKEN])

    def test_counts_repeated_bigram_and_trigram_types(self):
        self.assertEqual(self.model.count(2, ("the", "cat")), 3)
        self.assertEqual(self.model.count(3, ("the", "cat", "sat")), 3)
        self.assertEqual(self.model.count(2, ("cat", "sat")), 5)

    def test_continuation_count_uses_distinct_predecessors(self):
        self.assertEqual(self.model.continuation_count("mat"), 1)
        self.assertEqual(self.model.continuation_count("rug"), 1)
        self.assertEqual(self.model.continuation_count("cat"), 2)

    def test_frequency_of_frequencies_discount_estimation(self):
        counts = [1] * 10 + [2] * 4 + [3] * 2 + [4]
        discounts = estimate_discounts(counts)

        self.assertAlmostEqual(discounts[0], 0.5555555555, places=6)
        self.assertAlmostEqual(discounts[1], 1.1666666666, places=6)
        self.assertAlmostEqual(discounts[2], 1.8888888888, places=6)

    def test_count_one_uses_d1_and_count_two_uses_d2(self):
        d1 = self.model.discount_for_count(1, order=2)
        d2 = self.model.discount_for_count(2, order=2)

        self.assertEqual(d1, 0.5)
        self.assertEqual(d2, 0.75)
        self.assertNotEqual(d1, d2)

    def test_seen_bigram_probability_has_direct_and_lower_order_parts(self):
        probability, trace = self.model.probability(
            "cat",
            ("the",),
            return_trace=True,
        )

        history_count = self.model.history_count(("the",))
        direct = (3 - self.model.discount_for_count(3, order=2)) / history_count
        continuation = self.model.continuation_probability("cat")
        expected_lambda = self.model.lambda_for_history(("the",))
        expected = direct + expected_lambda * continuation

        self.assertEqual(history_count, 18)
        self.assertAlmostEqual(probability, expected)
        self.assertEqual(trace[0]["order"], 2)
        self.assertAlmostEqual(trace[0]["direct_contribution"], direct)
        self.assertAlmostEqual(trace[0]["lambda"], expected_lambda)

    def test_unseen_bigram_recurses_to_continuation_probability(self):
        probability, trace = self.model.probability(
            "fly",
            ("cat",),
            return_trace=True,
        )

        self.assertEqual(self.model.count(2, ("cat", "fly")), 0)
        self.assertGreater(probability, 0.0)
        self.assertEqual(trace[0]["direct_contribution"], 0.0)
        self.assertEqual(trace[0]["event"], "seen_history_unseen_ngram")

    def test_seen_trigram_recurses_to_bigram(self):
        probability, trace = self.model.probability(
            "sat",
            ("the", "cat"),
            return_trace=True,
        )

        self.assertGreater(probability, 0.0)
        self.assertEqual(trace[0]["order"], 3)
        self.assertEqual(trace[1]["order"], 2)

    def test_each_observed_history_normalizes(self):
        histories_to_check = [
            (),
            ("the",),
            ("cat",),
            ("the", "cat"),
            ("cat", "sat"),
        ]

        for history in histories_to_check:
            total = self.model.normalization_total(history)
            self.assertTrue(
                math.isclose(total, 1.0, abs_tol=1e-9),
                msg=f"history={history!r} total={total}",
            )

    def test_unknown_word_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.probability("not-in-vocabulary", ())

    def test_history_longer_than_model_order_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.probability("cat", ("the", "cat", "sat"))

    def test_line_limit_zero_means_full_input(self):
        sentences = ["one", "two", "three"]

        self.assertEqual(take_first_lines(sentences, 0), sentences)
        self.assertEqual(take_first_lines(sentences, 2), ["one", "two"])

    def test_saved_huggingface_rows_become_non_empty_sentence_lists(self):
        fake_dataset = {
            "train": [{"text": "first row"}, {"text": "   "}, {"text": "second row"}],
            "validation": [{"text": "validation row"}],
            "test": [{"text": "test row"}],
        }

        def fake_loader(dataset_path):
            self.assertEqual(dataset_path, "data/wikitext-2")
            return fake_dataset

        splits = load_saved_huggingface_splits(
            "data/wikitext-2", dataset_loader=fake_loader
        )

        self.assertEqual(splits["train"], ["first row", "second row"])
        self.assertEqual(splits["validation"], ["validation row"])
        self.assertEqual(splits["test"], ["test row"])

    def test_download_script_saves_the_dataset_to_the_requested_directory(self):
        script_path = Path(__file__).with_name("download_wikitext2.py")
        spec = importlib.util.spec_from_file_location("download_wikitext2", script_path)
        download_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(download_module)

        class FakeDataset:
            def __init__(self):
                self.saved_path = None

            def save_to_disk(self, path):
                self.saved_path = path

        fake_dataset = FakeDataset()

        def fake_loader(dataset_name, config_name):
            self.assertEqual(dataset_name, "Salesforce/wikitext")
            self.assertEqual(config_name, "wikitext-2-v1")
            return fake_dataset

        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "wikitext-2"
            download_module.download_and_save(output_path, fake_loader)

        self.assertEqual(fake_dataset.saved_path, str(output_path))

    def test_validation_unknown_words_use_unk(self):
        vocabulary = ("known", "<unk>")

        self.assertEqual(
            evaluation_words("known unseen", vocabulary), ["known", "<unk>"]
        )

    def test_validation_perplexity_is_positive(self):
        perplexity, predicted_tokens = validation_perplexity(
            self.model, ["the cat sat"]
        )

        self.assertGreater(perplexity, 0.0)
        self.assertEqual(predicted_tokens, 4)


if __name__ == "__main__":
    unittest.main()
