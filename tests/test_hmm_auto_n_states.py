import numpy as np
import pytest

pytest.importorskip("hmmlearn")
from hmmlearn.hmm import GaussianHMM

from brainnet.dynamic.config import DynamicConfig
from brainnet.dynamic.hmm import hmm_analysis


def _generate_sample(n_samples: int = 300, random_state: int = 0) -> np.ndarray:
    rng = np.random.RandomState(random_state)
    model = GaussianHMM(n_components=2, covariance_type="full", random_state=random_state)
    model.startprob_ = np.array([0.6, 0.4])
    model.transmat_ = np.array([[0.7, 0.3], [0.3, 0.7]])
    model.means_ = np.array([[0.0], [3.0]])
    model.covars_ = np.array([[[0.5]], [[0.5]]])
    sample, _ = model.sample(n_samples, random_state=rng)
    return sample


@pytest.mark.parametrize("criterion", ["bic", "aic"])
def test_auto_n_states_hmm_selects_correct_number(criterion):
    data = _generate_sample()
    cfg = DynamicConfig(
        window_length=1,
        step=1,
        n_states=4,
        auto_n_states=True,
        method="hmm",
        random_state=0,
        n_states_criterion=criterion,
    )
    model = hmm_analysis(data, cfg)
    assert model.n_states == 2
