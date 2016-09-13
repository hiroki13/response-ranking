import unittest
from ..utils import load_dataset, load_init_emb
from ..ranking.preprocessor import convert_word_into_id
from ..ranking.sample import Sample


class TestPreProc(unittest.TestCase):

    def setUp(self):
        train_dataset, word_set = load_dataset(fn='text.txt')
        train_samples, vocab_words = load_init_emb(None, word_set)
        train_samples = convert_word_into_id(train_dataset, vocab_words)
        self.sample = Sample(context, spk_id, adr_id, responses, label, n_agents_in_ctx, max_n_agents, max_n_words, pad)

    def test_sample(self):
        self.assertEqual()


if __name__ == '__main__':
    unittest.main()
