from types import SimpleNamespace

from datasets.dataset import VerdictDataset
from datasets.dataloader import verdictDataLoader


class FakeTokenizer:
    def encode(self, text):
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_verdict_dataset_builds_correct_windows():
    config = SimpleNamespace(context_length=4, stride=2)
    dataset = VerdictDataset("dummy", FakeTokenizer(), config)

    assert len(dataset) == 3
    assert dataset[0][0].tolist() == [0, 1, 2, 3]
    assert dataset[0][1].tolist() == [1, 2, 3, 4]


def test_verdict_dataloader_wrapper_is_available():
    dataloader = verdictDataLoader("dummy", batch_size=2, shuffle=False, context_length=4, stride=2, tokenizer=FakeTokenizer())

    assert dataloader.batch_size == 2
