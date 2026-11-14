import numpy as np

from truthlens.calibration import (
    expected_calibration_error,
    fit_temperature,
    softmax_with_temperature,
)


def test_ece_is_zero_for_certain_correct_predictions():
    probs = np.array([[1.0, 0.0], [0.0, 1.0]])

    assert expected_calibration_error(probs, np.array([0, 1])) == 0.0


def test_temperature_above_one_softens_probabilities():
    softened = softmax_with_temperature(np.array([[0.9, 0.1]]), 2.0)

    assert softened[0, 0] < 0.9
    assert abs(softened.sum() - 1.0) < 1e-9


def test_fit_temperature_softens_overconfident_wrong_predictions():
    probs = np.array([[0.99, 0.01]] * 10)
    labels = np.array([0] * 5 + [1] * 5)

    assert fit_temperature(probs, labels) > 1.0
