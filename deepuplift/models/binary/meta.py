from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseBinaryUpliftModel, OutcomeEstimator


def _append_treatment(x: pd.DataFrame, treatment: int) -> pd.DataFrame:
    result = x.copy()
    result["__treatment__"] = float(treatment)
    return result


class SLearner(BaseBinaryUpliftModel):
    name = "S-Learner"

    def _fit(self, x, y, treatment, assignment_type):
        self.outcome = OutcomeEstimator(self.task, self.random_state).fit(_append_treatment(x, 0).assign(__treatment__=treatment), y)
        self.propensity = float(np.mean(treatment))

    def _predict_effects(self, x):
        y0 = self.outcome.predict(_append_treatment(x, 0))
        y1 = self.outcome.predict(_append_treatment(x, 1))
        return y0, y1, np.full(len(x), self.propensity, dtype="float64")


class TLearner(BaseBinaryUpliftModel):
    name = "T-Learner"

    def _fit(self, x, y, treatment, assignment_type):
        if np.sum(treatment == 0) < 2 or np.sum(treatment == 1) < 2:
            raise ValueError("T-Learner needs at least two rows in each treatment arm.")
        self.control = OutcomeEstimator(self.task, self.random_state).fit(x[treatment == 0], y[treatment == 0])
        self.treated = OutcomeEstimator(self.task, self.random_state).fit(x[treatment == 1], y[treatment == 1])
        self.propensity = float(np.mean(treatment))

    def _predict_effects(self, x):
        return self.control.predict(x), self.treated.predict(x), np.full(len(x), self.propensity, dtype="float64")


class XLearner(BaseBinaryUpliftModel):
    name = "X-Learner"

    def _fit(self, x, y, treatment, assignment_type):
        self.control = OutcomeEstimator(self.task, self.random_state).fit(x[treatment == 0], y[treatment == 0])
        self.treated = OutcomeEstimator(self.task, self.random_state).fit(x[treatment == 1], y[treatment == 1])
        tau_treated = y[treatment == 1] - self.control.predict(x[treatment == 1])
        tau_control = self.treated.predict(x[treatment == 0]) - y[treatment == 0]
        self.effect_treated = OutcomeEstimator("regression", self.random_state).fit(x[treatment == 1], tau_treated)
        self.effect_control = OutcomeEstimator("regression", self.random_state).fit(x[treatment == 0], tau_control)
        self.propensity = float(np.mean(treatment))

    def _predict_effects(self, x):
        y0 = self.control.predict(x)
        y1 = self.treated.predict(x)
        tau = self.propensity * self.effect_control.predict(x) + (1.0 - self.propensity) * self.effect_treated.predict(x)
        return y0, y0 + tau, np.full(len(x), self.propensity, dtype="float64")


class DRLearner(BaseBinaryUpliftModel):
    name = "DR-Learner"

    def _fit(self, x, y, treatment, assignment_type):
        self.control = OutcomeEstimator(self.task, self.random_state).fit(x[treatment == 0], y[treatment == 0])
        self.treated = OutcomeEstimator(self.task, self.random_state).fit(x[treatment == 1], y[treatment == 1])
        if assignment_type == "randomized":
            self.propensity = float(np.mean(treatment))
            self.propensity_model = None
        else:
            from sklearn.linear_model import LogisticRegression

            self.propensity_model = LogisticRegression(max_iter=500, random_state=self.random_state).fit(x, treatment)
            self.propensity = None
        m0 = self.control.predict(x)
        m1 = self.treated.predict(x)
        e = self._propensity(x, treatment)
        mu = np.where(treatment == 1, m1, m0)
        pseudo = (m1 - m0) + (treatment - e) / np.clip(e * (1.0 - e), 0.02, None) * (y - mu)
        self.effect = OutcomeEstimator("regression", self.random_state).fit(x, pseudo)

    def _propensity(self, x, treatment=None):
        if self.propensity_model is not None:
            return np.clip(self.propensity_model.predict_proba(x)[:, 1], 0.02, 0.98)
        return np.full(len(x), self.propensity if self.propensity is not None else 0.5, dtype="float64")

    def _predict_effects(self, x):
        y0 = self.control.predict(x)
        return y0, y0 + self.effect.predict(x), self._propensity(x)
