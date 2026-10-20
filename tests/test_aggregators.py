import pytest
import torch

from sybil_fl.aggregators import aggregate, coordinate_median, fedavg


def test_fedavg_matches_hand_calculation():
    # (1*[1,2] + 3*[3,4]) / 4 = [2.5, 3.5]
    out = fedavg([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])], num_samples=[1, 3])
    assert torch.equal(out, torch.tensor([2.5, 3.5]))


def test_fedavg_equal_weights_is_plain_mean():
    deltas = [torch.tensor([1.0, -1.0, 0.0]), torch.tensor([3.0, 1.0, 2.0]), torch.tensor([2.0, 0.0, 4.0])]
    out = fedavg(deltas, num_samples=[5, 5, 5])
    assert torch.allclose(out, torch.stack(deltas).mean(dim=0))


def test_fedavg_single_client_returns_its_update():
    d = torch.tensor([0.5, -2.0, 7.0])
    assert torch.equal(fedavg([d], num_samples=[42]), d)


def test_fedavg_does_not_modify_inputs_and_keeps_dtype():
    a, b = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    a_copy, b_copy = a.clone(), b.clone()
    out = fedavg([a, b], num_samples=[1, 1])
    assert torch.equal(a, a_copy) and torch.equal(b, b_copy)
    assert out.dtype == a.dtype


def test_fedavg_rejects_empty_input():
    with pytest.raises(ValueError, match="no updates"):
        fedavg([], num_samples=[])


def test_fedavg_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="total weight"):
        fedavg([torch.zeros(2), torch.ones(2)], num_samples=[0, 0])


def test_fedavg_rejects_negative_weight():
    with pytest.raises(ValueError, match="negative"):
        fedavg([torch.zeros(2), torch.ones(2)], num_samples=[3, -1])


def test_fedavg_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        fedavg([torch.zeros(2), torch.ones(3)], num_samples=[1, 1])


def test_fedavg_rejects_count_mismatch():
    with pytest.raises(ValueError, match="num_samples"):
        fedavg([torch.zeros(2), torch.ones(2)], num_samples=[1])


def test_fedavg_rejects_non_finite_update():
    with pytest.raises(ValueError, match="non-finite"):
        fedavg([torch.zeros(2), torch.tensor([1.0, float("nan")])], num_samples=[1, 1])


# ---------- coordinate-wise median ----------


def test_median_odd_count_hand_case():
    out = coordinate_median([torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0]), torch.tensor([10.0, 0.0])])
    assert torch.equal(out, torch.tensor([3.0, 2.0]))


def test_median_even_count_averages_the_middle_two():
    out = coordinate_median([torch.tensor([v]) for v in (1.0, 2.0, 3.0, 10.0)])
    assert torch.equal(out, torch.tensor([2.5]))


def test_median_rejects_empty_and_non_finite():
    with pytest.raises(ValueError, match="no updates"):
        coordinate_median([])
    with pytest.raises(ValueError, match="non-finite"):
        coordinate_median([torch.tensor([float("inf")]), torch.tensor([1.0])])


def test_aggregate_dispatches_on_name():
    deltas, n = [torch.tensor([0.0]), torch.tensor([3.0]), torch.tensor([9.0])], [1, 1, 2]
    assert torch.equal(aggregate("fedavg", deltas, n), fedavg(deltas, n))
    assert torch.equal(aggregate("median", deltas, n), torch.tensor([3.0]))  # median ignores sample counts
    with pytest.raises(ValueError, match="aggregator"):
        aggregate("krum", deltas, n)
