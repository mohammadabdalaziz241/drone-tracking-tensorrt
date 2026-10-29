"""Tests for the C++ vs Python detection-agreement metric in scripts/parity_check.py.

Detections are rows of (x1, y1, x2, y2, confidence, class). Ultralytics is only imported inside
parity_check.main(), so these tests need numpy alone.
"""

import numpy as np
import pytest

from parity_check import MATCH_IOUS, agreement, iou_matrix, match, to_array


def det(x1, y1, x2, y2, conf=0.9, cls=0):
    return [x1, y1, x2, y2, conf, cls]


def arr(*rows):
    return np.array(rows, dtype=float).reshape(-1, 6)


# --------------------------------------------------------------------------- iou_matrix


def test_iou_matrix_known_values():
    a = np.array([[0, 0, 2, 2], [0, 0, 10, 10]], dtype=float)
    b = np.array([[1, 0, 3, 2], [0, 0, 2, 2], [20, 20, 30, 30]], dtype=float)
    m = iou_matrix(a, b)
    assert m.shape == (2, 3)
    assert m[0, 0] == pytest.approx(1 / 3)  # half overlap: 2 / (4 + 4 - 2)
    assert m[0, 1] == pytest.approx(1.0)
    assert m[0, 2] == 0.0
    assert m[1, 1] == pytest.approx(4 / 100)  # small box inside a large one


@pytest.mark.parametrize("na, nb", [(0, 3), (2, 0), (0, 0)])
def test_iou_matrix_empty_inputs(na, nb):
    m = iou_matrix(np.zeros((na, 4)), np.zeros((nb, 4)))
    assert m.shape == (na, nb)


# --------------------------------------------------------------------------- match


def test_match_identical_sets_match_fully():
    ref = arr(det(0, 0, 10, 10), det(20, 20, 30, 30, cls=1))
    assert match(ref, ref.copy(), 0.9) == 2


def test_match_requires_same_class():
    ref = arr(det(0, 0, 10, 10, cls=0))
    other = arr(det(0, 0, 10, 10, cls=1))
    assert match(ref, other, 0.5) == 0


def test_match_threshold_boundary():
    ref = arr(det(0, 0, 2, 2))
    other = arr(det(1, 0, 3, 2))  # IoU 1/3 (minus the 1e-9 guard in the denominator)
    assert match(ref, other, 0.333) == 1
    assert match(ref, other, 0.34) == 0


def test_match_is_one_to_one():
    ref = arr(det(0, 0, 10, 10, conf=0.9), det(0, 0, 10, 10, conf=0.8))
    other = arr(det(0, 0, 10, 10))
    assert match(ref, other, 0.5) == 1


@pytest.mark.parametrize("ref, other", [(arr(), arr(det(0, 0, 1, 1))), (arr(det(0, 0, 1, 1)), arr())])
def test_match_with_an_empty_side_is_zero(ref, other):
    assert match(ref, other, 0.5) == 0


# --------------------------------------------------------------------------- to_array


class _Tensor:
    """Minimal stand-in for a torch tensor: .cpu().numpy() returns the array."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class _Boxes:
    """Minimal stand-in for ultralytics.engine.results.Boxes."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Tensor(xyxy), _Tensor(conf), _Tensor(cls)

    def __len__(self):
        return len(self.conf.numpy())


def test_to_array_stacks_boxes_confidence_and_class():
    out = to_array(_Boxes([[1, 2, 3, 4], [5, 6, 7, 8]], [0.9, 0.5], [0, 3]))
    assert out.shape == (2, 6)
    np.testing.assert_allclose(out[0], [1, 2, 3, 4, 0.9, 0], rtol=1e-6)
    np.testing.assert_allclose(out[1], [5, 6, 7, 8, 0.5, 3], rtol=1e-6)


def test_to_array_empty():
    out = to_array(_Boxes(np.zeros((0, 4)), [], []))
    assert out.shape == (0, 6)


# --------------------------------------------------------------------------- agreement


def test_agreement_perfect():
    a = arr(det(0, 0, 10, 10), det(20, 20, 30, 30, cls=2))
    out = agreement([(a, a.copy()), (a, a.copy())])
    assert out["reference_detections"] == 4
    assert out["other_detections"] == 4
    for thr in MATCH_IOUS:
        assert out[f"recall@iou{thr}"] == 1.0
        assert out[f"precision@iou{thr}"] == 1.0


def test_agreement_separates_jitter_from_missing_boxes():
    # One box shifted by 1 px (IoU 0.82 for a 10x10 box) and one box missing entirely.
    ref = arr(det(0, 0, 10, 10), det(50, 50, 60, 60))
    other = arr(det(1, 0, 11, 10))
    out = agreement([(ref, other)])
    assert out["recall@iou0.5"] == 0.5  # the missing box
    assert out["recall@iou0.75"] == 0.5
    assert out["recall@iou0.9"] == 0.0  # the jittered box also fails the strict threshold
    assert out["precision@iou0.5"] == 1.0
    assert out["precision@iou0.9"] == 0.0


def test_agreement_with_no_detections_does_not_divide_by_zero():
    out = agreement([(arr(), arr())])
    assert out["reference_detections"] == 0
    assert all(out[f"recall@iou{thr}"] == 0.0 for thr in MATCH_IOUS)
