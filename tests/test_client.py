import numpy as np
import torch

from helpers import synthetic_digits
from sybil_fl.client import local_update
from sybil_fl.config import TrainConfig
from sybil_fl.models import build_model, get_flat_params, num_params

TRAIN = TrainConfig(batch_size=16, lr=0.05, momentum=0.0)


def _update(model, idx, seed=0, client_id=0):
    x, y = synthetic_digits(n_per_class=20)
    return local_update(model, x, y, idx, client_id=client_id, train_cfg=TRAIN, local_epochs=1, seed=seed)


def test_update_is_finite_flat_vector_of_model_size():
    m = build_model(seed=0)
    u = _update(m, np.arange(100))
    assert u.delta.shape == (num_params(m),)
    assert torch.isfinite(u.delta).all()
    assert u.delta.abs().sum() > 0  # training actually moved the weights


def test_update_reports_client_id_sample_count_and_loss():
    u = _update(build_model(seed=0), np.arange(37), client_id=4)
    assert u.client_id == 4
    assert u.num_samples == 37
    assert np.isfinite(u.train_loss)


def test_global_model_is_not_modified():
    m = build_model(seed=0)
    before = get_flat_params(m)
    _update(m, np.arange(100))
    assert torch.equal(get_flat_params(m), before)


def test_delta_is_local_minus_global():
    # Applying the delta to the global weights must reproduce the locally trained weights:
    # train the same way by hand and compare.
    from sybil_fl.training import train_epochs

    x, y = synthetic_digits(n_per_class=20)
    idx = np.arange(100)
    m = build_model(seed=0)
    u = local_update(m, x, y, idx, client_id=0, train_cfg=TRAIN, local_epochs=1, seed=3)
    by_hand = build_model(seed=0)
    train_epochs(by_hand, x, y, idx, epochs=1, batch_size=16, lr=0.05, momentum=0.0, seed=3)
    assert torch.allclose(get_flat_params(m) + u.delta, get_flat_params(by_hand), atol=1e-6)


def test_same_seed_same_update_different_seed_different():
    m = build_model(seed=0)
    a, b, c = _update(m, np.arange(100), seed=1), _update(m, np.arange(100), seed=1), _update(m, np.arange(100), seed=2)
    assert torch.equal(a.delta, b.delta)
    assert not torch.equal(a.delta, c.delta)
