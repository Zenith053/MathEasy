"""The shortlister saves its name vectors once and reuses them; a changed name list or adapter recomputes."""

import json

import pytest

torch = pytest.importorskip("torch")
model = pytest.importorskip("matheasy.shortlister.model")
if not (model.DATA / "raw" / "misconception_mapping.csv").exists():
    pytest.skip("no Eedi data on this machine", allow_module_level=True)


class FakeModel:
    device = "cpu"


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16}))
    (tmp_path / "results.json").write_text(json.dumps({"model": "Qwen/Qwen2.5-3B-Instruct"}))
    monkeypatch.setattr(model, "_load", lambda adapter: (None, FakeModel()))
    calls = []

    def embed(self, texts, max_len, size):
        calls.append(len(texts))
        return torch.ones(len(texts), 4)

    monkeypatch.setattr(model.ModelShortlister, "_embed", embed)
    return tmp_path, calls


def test_vectors_are_computed_once_then_read_back(adapter):
    path, calls = adapter
    first = model.ModelShortlister(path, candidates={1, 4, 6})
    assert calls == [3] and len(list(path.glob("name_vectors_*.pt"))) == 1
    second = model.ModelShortlister(path, candidates={1, 4, 6})
    assert calls == [3]                                   # read from the file, not recomputed
    assert torch.equal(first.vectors, second.vectors) and second.ids == first.ids


def test_a_different_name_list_or_adapter_recomputes(adapter):
    path, calls = adapter
    model.ModelShortlister(path, candidates={1, 4, 6})
    model.ModelShortlister(path, candidates={1, 4})       # other names: another file
    (path / "results.json").write_text(json.dumps({"model": "Qwen/Qwen2.5-3B-Instruct", "epoch": 9}))
    model.ModelShortlister(path, candidates={1, 4, 6})    # retrained adapter: recomputed
    assert calls == [3, 2, 3]
