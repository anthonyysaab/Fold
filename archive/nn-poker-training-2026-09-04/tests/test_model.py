import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pokerkit")

data_module = pytest.importorskip("poker_nn_training.data")
model_module = pytest.importorskip("poker_nn_training.model")
FEATURE_NAMES = data_module.FEATURE_NAMES
LABELS = data_module.LABELS
TinyPolicy = model_module.TinyPolicy
mask_illegal_logits = model_module.mask_illegal_logits


def test_tiny_policy_shape() -> None:
    model = TinyPolicy(hidden_size=8)
    features = torch.zeros((4, len(FEATURE_NAMES)))

    assert model(features).shape == (4, len(LABELS))


def test_illegal_actions_are_masked() -> None:
    features = torch.zeros((1, len(FEATURE_NAMES)))
    features[0, FEATURE_NAMES.index("legal_check_call")] = 1
    logits = torch.tensor([[100.0, 0.0, 100.0]])

    masked = mask_illegal_logits(logits, features)

    assert masked.argmax(dim=1).item() == LABELS.index("check_call")
