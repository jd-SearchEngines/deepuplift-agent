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

    def _fit(self, x, y, treatment, assignment_type, nuisance=None):
        self.outcome = OutcomeEstimator(self.task, self.random_state, self.outcome_model).fit(_append_treatment(x, 0).assign(__treatment__=treatment), y)
        self.propensity = float(np.mean(treatment))

    def _predict_effects(self, x):
        y0 = self.outcome.predict(_append_treatment(x, 0))
        y1 = self.outcome.predict(_append_treatment(x, 1))
        return y0, y1, self._prediction_propensity(x, self.propensity)


class TLearner(BaseBinaryUpliftModel):
    name = "T-Learner"

    def _fit(self, x, y, treatment, assignment_type, nuisance=None):
        if np.sum(treatment == 0) < 2 or np.sum(treatment == 1) < 2:
            raise ValueError("T-Learner needs at least two rows in each treatment arm.")
        self.control = OutcomeEstimator(self.task, self.random_state, self.outcome_model).fit(x[treatment == 0], y[treatment == 0])
        self.treated = OutcomeEstimator(self.task, self.random_state, self.outcome_model).fit(x[treatment == 1], y[treatment == 1])
        self.propensity = float(np.mean(treatment))

    def _predict_effects(self, x):
        return self.control.predict(x), self.treated.predict(x), self._prediction_propensity(x, self.propensity)


class XLearner(BaseBinaryUpliftModel):
    name = "X-Learner"

    def _fit(self, x, y, treatment, assignment_type, nuisance=None):
        self.control = OutcomeEstimator(self.task, self.random_state, self.outcome_model).fit(x[treatment == 0], y[treatment == 0])
        self.treated = OutcomeEstimator(self.task, self.random_state, self.outcome_model).fit(x[treatment == 1], y[treatment == 1])
        tau_treated = y[treatment == 1] - self.control.predict(x[treatment == 1])
        tau_control = self.treated.predict(x[treatment == 0]) - y[treatment == 0]
        self.effect_treated = OutcomeEstimator("regression", self.random_state, self.outcome_model).fit(x[treatment == 1], tau_treated)
        self.effect_control = OutcomeEstimator("regression", self.random_state, self.outcome_model).fit(x[treatment == 0], tau_control)
        self.propensity = float(np.mean(treatment))

    def _predict_effects(self, x):
        y0 = self.control.predict(x)
        y1 = self.treated.predict(x)
        tau = self.propensity * self.effect_control.predict(x) + (1.0 - self.propensity) * self.effect_treated.predict(x)
        return y0, y0 + tau, self._prediction_propensity(x, self.propensity)


class DRLearner(BaseBinaryUpliftModel):
    name = "DR-Learner"

    def _fit(self, x, y, treatment, assignment_type, nuisance=None):
        self.control = OutcomeEstimator(self.task, self.random_state, self.outcome_model).fit(x[treatment == 0], y[treatment == 0])
        self.treated = OutcomeEstimator(self.task, self.random_state, self.outcome_model).fit(x[treatment == 1], y[treatment == 1])
        self.propensity = float(np.mean(treatment))
        m0 = nuisance.outcome_control_oof if nuisance is not None else self.control.predict(x)
        m1 = nuisance.outcome_treated_oof if nuisance is not None else self.treated.predict(x)
        e = nuisance.propensity_scores if nuisance is not None else self._propensity(x, treatment)
        mu = np.where(treatment == 1, m1, m0)
        pseudo = (m1 - m0) + (treatment - e) / np.clip(e * (1.0 - e), 0.02, None) * (y - mu)
        fit_mask = getattr(nuisance, "trim_mask", np.ones(len(x), dtype=bool)) if nuisance is not None else np.ones(len(x), dtype=bool)
        self.effect = OutcomeEstimator("regression", self.random_state, self.outcome_model).fit(x[fit_mask], pseudo[fit_mask])

    def _propensity(self, x, treatment=None):
        return self._prediction_propensity(x, self.propensity if self.propensity is not None else 0.5)

    def _predict_effects(self, x):
        y0 = self.control.predict(x)
        return y0, y0 + self.effect.predict(x), self._propensity(x)


class RLearner(BaseBinaryUpliftModel):
    name = "R-Learner"

    def _fit(self, x, y, treatment, assignment_type, nuisance=None):
        if nuisance is None:
            raise ValueError("R-Learner requires the unified nuisance result; call fit(dataset, nuisance=result).")
        e = np.clip(nuisance.propensity_scores, 1e-3, 1 - 1e-3)
        mu = np.where(treatment == 1, nuisance.outcome_treated_oof, nuisance.outcome_control_oof)
        residual_t = treatment - e
        denominator = np.where(np.abs(residual_t) > 0.05, residual_t, np.where(residual_t >= 0, 0.05, -0.05))
        pseudo = (y - mu) / denominator
        mask = nuisance.trim_mask & np.isfinite(pseudo)
        self.effect = OutcomeEstimator("regression", self.random_state, self.outcome_model).fit(x[mask], pseudo[mask], sample_weight=np.square(residual_t[mask]))
        self.propensity = float(np.mean(treatment))

    def _predict_effects(self, x):
        y0 = np.zeros(len(x), dtype="float64")
        return y0, self.effect.predict(x), self._prediction_propensity(x, self.propensity)


class IPWLearner(BaseBinaryUpliftModel):
    name = "IPW-Learner"

    def _fit(self, x, y, treatment, assignment_type, nuisance=None):
        if nuisance is None:
            raise ValueError("IPW-Learner requires the unified nuisance result; call fit(dataset, nuisance=result).")
        e = np.clip(nuisance.propensity_scores, 1e-3, 1 - 1e-3)
        pseudo = y * np.where(treatment == 1, 1 / e, -1 / (1 - e))
        mask = nuisance.trim_mask & np.isfinite(pseudo)
        self.effect = OutcomeEstimator("regression", self.random_state, self.outcome_model).fit(x[mask], pseudo[mask], sample_weight=nuisance.sample_weights[mask])
        self.propensity = float(np.mean(treatment))

    def _predict_effects(self, x):
        y0 = np.zeros(len(x), dtype="float64")
        return y0, self.effect.predict(x), self._prediction_propensity(x, self.propensity)
