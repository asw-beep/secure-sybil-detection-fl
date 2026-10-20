import numpy as np
import torch

from helpers import synthetic_digits as _synthetic_digits
from sybil_fl.models import SmallCNN, build_model, get_flat_params, num_params, set_flat_params
from sybil_fl.training import evaluate_accuracy, train_epochs


# ---------- architecture ----------

def test_forward_shape_and_finite():
    out = SmallCNN()(torch.randn(8, 1, 28, 28))
    assert out.shape == (8, 10)
    assert torch.isfinite(out).all()


def test_parameter_count_matches_design_decision():
    # docs/DECISIONS.md D5: d = 46,730
    assert num_params(SmallCNN()) == 46_730


def test_build_model_is_reproducible_from_seed():
    a, b, c = build_model(seed=1), build_model(seed=1), build_model(seed=2)
    assert torch.equal(get_flat_params(a), get_flat_params(b))
    assert not torch.equal(get_flat_params(a), get_flat_params(c))


# ---------- flatten / unflatten ----------

def test_flat_params_round_trip_preserves_function():
    src, dst = build_model(seed=1), build_model(seed=2)
    set_flat_params(dst, get_flat_params(src))
    x = torch.randn(4, 1, 28, 28)
    assert torch.equal(src(x), dst(x))


def test_get_flat_params_returns_detached_copy():
    m = build_model(seed=1)
    vec = get_flat_params(m)
    vec.add_(1.0)
    assert not torch.equal(vec, get_flat_params(m))
    assert not vec.requires_grad


def test_set_flat_params_rejects_wrong_length():
    m = build_model(seed=1)
    try:
        set_flat_params(m, torch.zeros(10))
    except ValueError as e:
        assert "46730" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_set_flat_params_rejects_non_finite():
    m = build_model(seed=1)
    bad = get_flat_params(m)
    bad[0] = float("nan")
    try:
        set_flat_params(m, bad)
    except ValueError as e:
        assert "finite" in str(e)
    else:
        raise AssertionError("expected ValueError")


# ---------- training / evaluation ----------

def test_training_improves_accuracy_on_learnable_data():
    x, y = _synthetic_digits()
    m = build_model(seed=0)
    before = evaluate_accuracy(m, x, y)
    idx = np.arange(len(y))
    train_epochs(m, x, y, idx, epochs=3, batch_size=32, lr=0.05, momentum=0.0, seed=0)
    after = evaluate_accuracy(m, x, y)
    assert 0.0 <= before <= 1.0
    assert after > 0.9 and after > before + 0.5


def test_training_only_uses_the_given_indices():
    # Train only on class-3 samples: the model should then predict 3 almost everywhere.
    x, y = _synthetic_digits()
    m = build_model(seed=0)
    idx = np.flatnonzero(y.numpy() == 3)
    train_epochs(m, x, y, idx, epochs=2, batch_size=16, lr=0.05, momentum=0.0, seed=0)
    with torch.no_grad():
        preds = m(x).argmax(1)
    assert (preds == 3).float().mean() > 0.9


def test_training_is_reproducible_with_same_seed():
    x, y = _synthetic_digits(n_per_class=20)
    idx = np.arange(len(y))
    a, b = build_model(seed=0), build_model(seed=0)
    train_epochs(a, x, y, idx, epochs=1, batch_size=16, lr=0.05, momentum=0.9, seed=5)
    train_epochs(b, x, y, idx, epochs=1, batch_size=16, lr=0.05, momentum=0.9, seed=5)
    assert torch.equal(get_flat_params(a), get_flat_params(b))
    assert torch.isfinite(get_flat_params(a)).all()


def test_training_on_empty_index_set_raises():
    x, y = _synthetic_digits(n_per_class=2)
    try:
        train_epochs(build_model(0), x, y, np.array([], dtype=np.int64), 1, 8, 0.05, 0.0, seed=0)
    except ValueError as e:
        assert "empty" in str(e)
    else:
        raise AssertionError("expected ValueError")
