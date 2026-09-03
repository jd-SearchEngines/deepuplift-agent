from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split


def _as_array(values: Any) -> np.ndarray:
    if isinstance(values, (pd.DataFrame, pd.Series)):
        return values.to_numpy()
    return np.asarray(values)


def _as_frame(values: Any) -> pd.DataFrame:
    if isinstance(values, pd.DataFrame):
        return values.copy()
    return pd.DataFrame(values)


def _append_treatment(x: Any, treatment_value: float) -> pd.DataFrame:
    frame = _as_frame(x)
    enriched = frame.copy()
    enriched["__treatment__"] = treatment_value
    return enriched


def _clip_propensity(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.02, 0.98)


def _array_summary(values: Any) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    summary: Dict[str, Any] = {"count": int(arr.size), "finite_count": int(finite.size)}
    if finite.size == 0:
        summary.update(
            {
                "mean": None,
                "std": None,
                "min": None,
                "p01": None,
                "p05": None,
                "median": None,
                "p95": None,
                "p99": None,
                "max": None,
            }
        )
        return summary
    summary.update(
        {
            "mean": float(np.mean(finite)),
            "std": float(np.std(finite)),
            "min": float(np.min(finite)),
            "p01": float(np.quantile(finite, 0.01)),
            "p05": float(np.quantile(finite, 0.05)),
            "median": float(np.median(finite)),
            "p95": float(np.quantile(finite, 0.95)),
            "p99": float(np.quantile(finite, 0.99)),
            "max": float(np.max(finite)),
        }
    )
    return summary


def _weight_effective_sample_size(values: Any) -> float | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr) & (arr >= 0)]
    denom = float(np.sum(finite**2))
    if finite.size == 0 or denom <= 0:
        return None
    return float((np.sum(finite) ** 2) / denom)


def _predict_outcome(model: Any, x: Any, task: str) -> np.ndarray:
    if task == "classification":
        proba = model.predict_proba(x)
        if proba.shape[1] == 1:
            return np.full(len(_as_frame(x)), float(model.classes_[0]))
        class_index = list(model.classes_).index(1) if 1 in model.classes_ else -1
        return proba[:, class_index]
    return model.predict(x)


def _fit_estimator(estimator: Any, x: Any, y: Any, task: str, sample_weight: Any | None = None):
    y_array = _as_array(y).reshape(-1)
    if len(y_array) == 0:
        raise ValueError("Cannot fit estimator on an empty treatment group.")
    if task == "classification" and len(np.unique(y_array)) < 2:
        model = DummyClassifier(strategy="constant", constant=int(y_array[0]))
    elif task == "regression" and len(np.unique(y_array)) < 2:
        model = DummyRegressor(strategy="constant", constant=float(y_array[0]))
    else:
        model = clone(estimator)
    if sample_weight is None:
        return model.fit(x, y_array)
    return model.fit(x, y_array, sample_weight=_as_array(sample_weight).reshape(-1))


def _default_outcome_estimator(task: str, params: Dict[str, Any], forest: bool = False):
    if forest and task == "classification":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            max_depth=params.get("max_depth"),
            n_jobs=params.get("n_jobs", -1),
            random_state=params.get("random_state", 42),
        )
    if forest:
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            max_depth=params.get("max_depth"),
            n_jobs=params.get("n_jobs", -1),
            random_state=params.get("random_state", 42),
        )
    if task == "classification":
        return HistGradientBoostingClassifier(
            max_iter=params.get("max_iter", 80),
            learning_rate=params.get("learning_rate", 0.05),
            max_leaf_nodes=params.get("max_leaf_nodes", 31),
            random_state=params.get("random_state", 42),
        )
    return HistGradientBoostingRegressor(
        max_iter=params.get("max_iter", 80),
        learning_rate=params.get("learning_rate", 0.05),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        random_state=params.get("random_state", 42),
    )


def _lightgbm_outcome_estimator(task: str, params: Dict[str, Any]):
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except ImportError as exc:
        raise ImportError("LightGBM learners require optional package 'lightgbm'.") from exc

    common = {
        "n_estimators": params.get("n_estimators", params.get("max_iter", 120)),
        "learning_rate": params.get("learning_rate", 0.05),
        "num_leaves": params.get("num_leaves", 31),
        "min_child_samples": params.get("min_child_samples", 20),
        "subsample": params.get("subsample", 0.9),
        "colsample_bytree": params.get("colsample_bytree", 0.9),
        "reg_alpha": params.get("reg_alpha", 0.0),
        "reg_lambda": params.get("reg_lambda", 0.0),
        "random_state": params.get("random_state", 42),
        "n_jobs": params.get("n_jobs", 1),
        "verbose": params.get("verbose", -1),
    }
    if task == "classification":
        return LGBMClassifier(**common)
    return LGBMRegressor(**common)


def _lightgbm_effect_estimator(params: Dict[str, Any]):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("LightGBM learners require optional package 'lightgbm'.") from exc

    return LGBMRegressor(
        n_estimators=params.get("effect_n_estimators", params.get("n_estimators", params.get("max_iter", 120))),
        learning_rate=params.get("effect_learning_rate", params.get("learning_rate", 0.05)),
        num_leaves=params.get("effect_num_leaves", params.get("num_leaves", 31)),
        min_child_samples=params.get("effect_min_child_samples", params.get("min_child_samples", 20)),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 0.0),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", 1),
        verbose=params.get("verbose", -1),
    )


def _xgboost_outcome_estimator(task: str, params: Dict[str, Any]):
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as exc:
        raise ImportError("XGBoost learners require optional package 'xgboost'.") from exc

    common = {
        "n_estimators": params.get("n_estimators", params.get("max_iter", 120)),
        "learning_rate": params.get("learning_rate", 0.05),
        "max_depth": params.get("max_depth", 4),
        "min_child_weight": params.get("min_child_weight", 1.0),
        "subsample": params.get("subsample", 0.9),
        "colsample_bytree": params.get("colsample_bytree", 0.9),
        "reg_alpha": params.get("reg_alpha", 0.0),
        "reg_lambda": params.get("reg_lambda", 1.0),
        "random_state": params.get("random_state", 42),
        "n_jobs": params.get("n_jobs", 1),
        "verbosity": params.get("verbosity", 0),
        "tree_method": params.get("tree_method", "hist"),
    }
    if task == "classification":
        return XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    return XGBRegressor(objective="reg:squarederror", **common)


def _xgboost_effect_estimator(params: Dict[str, Any]):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("XGBoost learners require optional package 'xgboost'.") from exc

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=params.get("effect_n_estimators", params.get("n_estimators", params.get("max_iter", 120))),
        learning_rate=params.get("effect_learning_rate", params.get("learning_rate", 0.05)),
        max_depth=params.get("effect_max_depth", params.get("max_depth", 4)),
        min_child_weight=params.get("effect_min_child_weight", params.get("min_child_weight", 1.0)),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        reg_alpha=params.get("reg_alpha", 0.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        random_state=params.get("random_state", 42),
        n_jobs=params.get("n_jobs", 1),
        verbosity=params.get("verbosity", 0),
        tree_method=params.get("tree_method", "hist"),
    )


def _catboost_outcome_estimator(task: str, params: Dict[str, Any]):
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        raise ImportError("CatBoost learners require optional package 'catboost'.") from exc

    common = {
        "iterations": params.get("iterations", params.get("n_estimators", params.get("max_iter", 120))),
        "learning_rate": params.get("learning_rate", 0.05),
        "depth": params.get("depth", params.get("max_depth", 6)),
        "l2_leaf_reg": params.get("l2_leaf_reg", params.get("reg_lambda", 3.0)),
        "random_seed": params.get("random_state", 42),
        "thread_count": params.get("thread_count", params.get("n_jobs", 1)),
        "verbose": params.get("verbose", False),
        "allow_writing_files": params.get("allow_writing_files", False),
    }
    if task == "classification":
        return CatBoostClassifier(loss_function=params.get("loss_function", "Logloss"), **common)
    return CatBoostRegressor(loss_function=params.get("loss_function", "RMSE"), **common)


def _catboost_effect_estimator(params: Dict[str, Any]):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("CatBoost learners require optional package 'catboost'.") from exc

    return CatBoostRegressor(
        loss_function=params.get("effect_loss_function", params.get("loss_function", "RMSE")),
        iterations=params.get("effect_iterations", params.get("iterations", params.get("n_estimators", params.get("max_iter", 120)))),
        learning_rate=params.get("effect_learning_rate", params.get("learning_rate", 0.05)),
        depth=params.get("effect_depth", params.get("depth", params.get("max_depth", 6))),
        l2_leaf_reg=params.get("effect_l2_leaf_reg", params.get("l2_leaf_reg", params.get("reg_lambda", 3.0))),
        random_seed=params.get("random_state", 42),
        thread_count=params.get("thread_count", params.get("n_jobs", 1)),
        verbose=params.get("verbose", False),
        allow_writing_files=params.get("allow_writing_files", False),
    )


def _default_effect_estimator(params: Dict[str, Any], forest: bool = False):
    if forest:
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            max_depth=params.get("max_depth"),
            n_jobs=params.get("n_jobs", -1),
            random_state=params.get("random_state", 42),
        )
    return HistGradientBoostingRegressor(
        max_iter=params.get("max_iter", 80),
        learning_rate=params.get("learning_rate", 0.05),
        max_leaf_nodes=params.get("max_leaf_nodes", 31),
        random_state=params.get("random_state", 42),
    )


@dataclass
class SklearnUpliftModel:
    learner_type: str
    task: str = "classification"
    params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.params = self.params or {}
        lightgbm_learners = {
            "s_lgbm",
            "t_lgbm",
            "x_lgbm",
            "dr_lgbm",
            "r_lgbm",
            "transformed_outcome_lgbm",
            "ipw_lgbm",
            "domain_adaptation_lgbm",
            "ggbm_lgbm",
            "pav_dr_lgbm",
            "pav_r_lgbm",
        }
        xgboost_learners = {
            "s_xgb",
            "t_xgb",
            "x_xgb",
            "dr_xgb",
            "r_xgb",
            "transformed_outcome_xgb",
            "ipw_xgb",
            "domain_adaptation_xgb",
            "ggbm_xgb",
            "pav_dr_xgb",
            "pav_r_xgb",
        }
        catboost_learners = {
            "s_cat",
            "t_cat",
            "x_cat",
            "dr_cat",
            "r_cat",
            "transformed_outcome_cat",
            "ipw_cat",
            "domain_adaptation_cat",
            "ggbm_cat",
            "pav_dr_cat",
            "pav_r_cat",
        }
        forest_outcome_learners = {
            "s_forest",
            "t_forest",
            "x_forest",
            "dr_forest",
            "r_forest",
            "causal_forest",
            "pav_causal_forest",
            "cvt_forest",
        }
        forest_effect_learners = {
            "x_forest",
            "dr_forest",
            "r_forest",
            "causal_forest",
            "pav_causal_forest",
            "orthogonal_random_forest",
            "transformed_outcome_forest",
            "ipw_forest",
            "cvt_forest",
        }
        if self.learner_type in lightgbm_learners:
            self.outcome_estimator = _lightgbm_outcome_estimator(self.task, self.params)
            self.effect_estimator = _lightgbm_effect_estimator(self.params)
        elif self.learner_type in xgboost_learners:
            self.outcome_estimator = _xgboost_outcome_estimator(self.task, self.params)
            self.effect_estimator = _xgboost_effect_estimator(self.params)
        elif self.learner_type in catboost_learners:
            self.outcome_estimator = _catboost_outcome_estimator(self.task, self.params)
            self.effect_estimator = _catboost_effect_estimator(self.params)
        else:
            self.outcome_estimator = _default_outcome_estimator(
                self.task,
                self.params,
                forest=self.learner_type in forest_outcome_learners,
            )
            self.effect_estimator = _default_effect_estimator(
                self.params,
                forest=self.learner_type in forest_effect_learners,
            )
        self.propensity_estimator = LogisticRegression(max_iter=1000)
        self.models_: Dict[str, Any] = {}
        self.history_: list[dict] = []
        self.nuisance_diagnostics_: Dict[str, Any] = {}

    def fit(
        self,
        x,
        y,
        t,
        batch_size: int = 64,
        epochs: int = 10,
        learning_rate: float = 1e-4,
        valid_perc: float | None = None,
        loss_f=None,
        tensorboard: bool = False,
        callback=None,
    ):
        x_frame = _as_frame(x)
        y_array = _as_array(y).reshape(-1)
        t_array = _as_array(t).reshape(-1).astype(int)
        self.nuisance_diagnostics_ = {}

        if valid_perc:
            stratify = t_array if len(np.unique(t_array)) == 2 and min(np.bincount(t_array)) >= 2 else None
            x_train, x_valid, y_train, y_valid, t_train, t_valid = train_test_split(
                x_frame,
                y_array,
                t_array,
                test_size=valid_perc,
                random_state=self.params.get("random_state", 42),
                stratify=stratify,
            )
        else:
            x_train, y_train, t_train = x_frame, y_array, t_array
            x_valid = y_valid = t_valid = None

        fitters = {
            "s": self._fit_s_learner,
            "s_forest": self._fit_s_learner,
            "s_lgbm": self._fit_s_learner,
            "s_xgb": self._fit_s_learner,
            "s_cat": self._fit_s_learner,
            "t": self._fit_t_learner,
            "t_forest": self._fit_t_learner,
            "t_lgbm": self._fit_t_learner,
            "t_xgb": self._fit_t_learner,
            "t_cat": self._fit_t_learner,
            "x": self._fit_x_learner,
            "x_forest": self._fit_x_learner,
            "x_lgbm": self._fit_x_learner,
            "x_xgb": self._fit_x_learner,
            "x_cat": self._fit_x_learner,
            "dr": self._fit_dr_learner,
            "dr_forest": self._fit_dr_learner,
            "dr_lgbm": self._fit_dr_learner,
            "dr_xgb": self._fit_dr_learner,
            "dr_cat": self._fit_dr_learner,
            "r": self._fit_r_learner,
            "r_forest": self._fit_r_learner,
            "r_lgbm": self._fit_r_learner,
            "r_xgb": self._fit_r_learner,
            "r_cat": self._fit_r_learner,
            "orthogonal_dml": self._fit_r_learner,
            "orthogonal_random_forest": self._fit_r_learner,
            "causal_forest": self._fit_dr_learner,
            "transformed_outcome": self._fit_transformed_outcome,
            "transformed_outcome_forest": self._fit_transformed_outcome,
            "transformed_outcome_lgbm": self._fit_transformed_outcome,
            "transformed_outcome_xgb": self._fit_transformed_outcome,
            "transformed_outcome_cat": self._fit_transformed_outcome,
            "ipw": self._fit_ipw_learner,
            "ipw_forest": self._fit_ipw_learner,
            "ipw_lgbm": self._fit_ipw_learner,
            "ipw_xgb": self._fit_ipw_learner,
            "ipw_cat": self._fit_ipw_learner,
            "domain_adaptation": self._fit_domain_adaptation_learner,
            "domain_adaptation_lgbm": self._fit_domain_adaptation_learner,
            "domain_adaptation_xgb": self._fit_domain_adaptation_learner,
            "domain_adaptation_cat": self._fit_domain_adaptation_learner,
            "pav_dr": self._fit_pav_dr_learner,
            "pav_dr_lgbm": self._fit_pav_dr_learner,
            "pav_dr_xgb": self._fit_pav_dr_learner,
            "pav_dr_cat": self._fit_pav_dr_learner,
            "pav_r": self._fit_pav_r_learner,
            "pav_r_lgbm": self._fit_pav_r_learner,
            "pav_r_xgb": self._fit_pav_r_learner,
            "pav_r_cat": self._fit_pav_r_learner,
            "pav_causal_forest": self._fit_pav_causal_forest,
            "ggbm": self._fit_ggbm_learner,
            "ggbm_lgbm": self._fit_ggbm_learner,
            "ggbm_xgb": self._fit_ggbm_learner,
            "ggbm_cat": self._fit_ggbm_learner,
            "cvt": self._fit_cvt_learner,
            "cvt_forest": self._fit_cvt_learner,
            "pav_cvt": self._fit_pav_cvt_learner,
        }
        fitters[self.learner_type](x_train, y_train, t_train)

        train_loss = self._observed_loss(x_train, y_train, t_train)
        metrics = {"epoch": 0, "train_loss": train_loss, "train_outcome_loss": train_loss, "train_treatment_loss": 0.0}
        if x_valid is not None:
            valid_loss = self._observed_loss(x_valid, y_valid, t_valid)
            metrics.update({"valid_loss": valid_loss, "valid_outcome_loss": valid_loss, "valid_treatment_loss": 0.0})

        self.history_ = [metrics]
        if callback:
            callback(metrics)
        return self.history_

    def _fit_s_learner(self, x, y, t) -> None:
        self.models_["s"] = _fit_estimator(self.outcome_estimator, _append_observed_treatment(x, t), y, self.task)

    def _fit_t_learner(self, x, y, t) -> None:
        self.models_["m0"] = _fit_estimator(self.outcome_estimator, x[t == 0], y[t == 0], self.task)
        self.models_["m1"] = _fit_estimator(self.outcome_estimator, x[t == 1], y[t == 1], self.task)

    def _fit_x_learner(self, x, y, t) -> None:
        self._fit_t_learner(x, y, t)
        m0_on_treated = _predict_outcome(self.models_["m0"], x[t == 1], self.task)
        m1_on_control = _predict_outcome(self.models_["m1"], x[t == 0], self.task)
        tau_treated = y[t == 1] - m0_on_treated
        tau_control = m1_on_control - y[t == 0]
        self.models_["tau1"] = _fit_estimator(self.effect_estimator, x[t == 1], tau_treated, "regression")
        self.models_["tau0"] = _fit_estimator(self.effect_estimator, x[t == 0], tau_control, "regression")
        self.models_["propensity"] = _fit_estimator(self.propensity_estimator, x, t, "classification")

    def _fit_dr_learner(self, x, y, t) -> None:
        x_frame = _as_frame(x)
        y_array = _as_array(y).reshape(-1)
        t_array = _as_array(t).reshape(-1).astype(int)
        folds = self._effective_nuisance_folds(t_array)
        fold_rows = []
        if folds >= 2:
            m0, m1, raw_e, e, fold_rows = self._cross_fit_dr_nuisance(x_frame, y_array, t_array, folds)
            self._fit_t_learner(x_frame, y_array, t_array)
            self.models_["propensity"] = _fit_estimator(self.propensity_estimator, x_frame, t_array, "classification")
        else:
            self._fit_t_learner(x_frame, y_array, t_array)
            self.models_["propensity"] = _fit_estimator(self.propensity_estimator, x_frame, t_array, "classification")
            m0 = _predict_outcome(self.models_["m0"], x_frame, self.task)
            m1 = _predict_outcome(self.models_["m1"], x_frame, self.task)
            raw_e = _predict_outcome(self.models_["propensity"], x_frame, "classification")
            e = _clip_propensity(raw_e)
        pseudo = (m1 - m0) + t_array * (y_array - m1) / e - (1 - t_array) * (y_array - m0) / (1 - e)
        self.models_["tau"] = _fit_estimator(self.effect_estimator, x_frame, pseudo, "regression")
        pseudo_weights = np.clip(e * (1 - e), 0.02, 0.25)
        self.nuisance_diagnostics_ = self._build_nuisance_diagnostics(
            method="dr",
            treatment=t_array,
            propensity_raw=raw_e,
            propensity_clipped=e,
            pseudo=pseudo,
            weights=pseudo_weights,
            cross_fit_folds=folds,
            fold_rows=fold_rows,
            extra={
                "m0": _array_summary(m0),
                "m1": _array_summary(m1),
                "tau_target": "doubly_robust_pseudo_effect",
            },
        )

    def _fit_pav_dr_learner(self, x, y, t) -> None:
        self._fit_dr_learner(x, y, t)
        self._fit_isotonic_tau_calibrator(x, y, t, base="dr")

    def _fit_pav_causal_forest(self, x, y, t) -> None:
        self._fit_dr_learner(x, y, t)
        self._fit_isotonic_tau_calibrator(x, y, t, base="dr")

    def _fit_ggbm_learner(self, x, y, t) -> None:
        self._fit_dr_learner(x, y, t)
        self._fit_isotonic_tau_calibrator(x, y, t, base="dr")

    def _fit_r_learner(self, x, y, t) -> None:
        x_frame = _as_frame(x)
        y_array = _as_array(y).reshape(-1)
        t_array = _as_array(t).reshape(-1).astype(int)
        folds = self._effective_nuisance_folds(t_array)
        fold_rows = []
        if folds >= 2:
            m, raw_e, e, fold_rows = self._cross_fit_r_nuisance(x_frame, y_array, t_array, folds)
            self.models_["m"] = _fit_estimator(self.outcome_estimator, x_frame, y_array, self.task)
            self.models_["propensity"] = _fit_estimator(self.propensity_estimator, x_frame, t_array, "classification")
        else:
            self.models_["m"] = _fit_estimator(self.outcome_estimator, x_frame, y_array, self.task)
            self.models_["propensity"] = _fit_estimator(self.propensity_estimator, x_frame, t_array, "classification")
            m = _predict_outcome(self.models_["m"], x_frame, self.task)
            raw_e = _predict_outcome(self.models_["propensity"], x_frame, "classification")
            e = _clip_propensity(raw_e)
        t_residual = t_array - e
        y_residual = y_array - m
        denom = np.where(np.abs(t_residual) < 0.05, np.sign(t_residual + 1e-12) * 0.05, t_residual)
        pseudo = y_residual / denom
        weights = np.clip(t_residual**2, 1e-4, None)
        self.models_["tau"] = _fit_estimator(self.effect_estimator, x_frame, pseudo, "regression", sample_weight=weights)
        self.nuisance_diagnostics_ = self._build_nuisance_diagnostics(
            method="r",
            treatment=t_array,
            propensity_raw=raw_e,
            propensity_clipped=e,
            pseudo=pseudo,
            weights=weights,
            residuals={"t_residual": t_residual, "y_residual": y_residual},
            cross_fit_folds=folds,
            fold_rows=fold_rows,
            extra={
                "m": _array_summary(m),
                "tau_target": "orthogonal_residual_ratio",
            },
        )

    def _fit_pav_r_learner(self, x, y, t) -> None:
        self._fit_r_learner(x, y, t)
        self._fit_isotonic_tau_calibrator(x, y, t, base="r")

    def _fit_transformed_outcome(self, x, y, t) -> None:
        treatment_rate = float(np.clip(np.mean(t), 0.02, 0.98))
        pseudo = y * (t - treatment_rate) / (treatment_rate * (1 - treatment_rate))
        self.models_["m"] = _fit_estimator(self.outcome_estimator, x, y, self.task)
        self.models_["tau"] = _fit_estimator(self.effect_estimator, x, pseudo, "regression")
        self.models_["constant_propensity"] = treatment_rate

    def _fit_cvt_learner(self, x, y, t) -> None:
        if self.task != "classification":
            raise ValueError("Class-variable-transform uplift supports classification only.")
        self.models_["m"] = _fit_estimator(self.outcome_estimator, x, y, self.task)
        self.models_["propensity"] = _fit_estimator(self.propensity_estimator, x, t, "classification")
        e = _clip_propensity(_predict_outcome(self.models_["propensity"], x, "classification"))
        transformed = np.where(t == 1, y, 1 - y).astype(int)
        weights = np.where(t == 1, 0.5 / e, 0.5 / (1 - e))
        weights = np.clip(weights, 0.1, np.quantile(weights, 0.95))
        self.models_["z"] = _fit_estimator(self.outcome_estimator, x, transformed, "classification", sample_weight=weights)

    def _fit_pav_cvt_learner(self, x, y, t) -> None:
        self._fit_cvt_learner(x, y, t)
        self._fit_isotonic_tau_calibrator(x, y, t, base="cvt")

    def _fit_ipw_learner(self, x, y, t) -> None:
        x_frame = _as_frame(x)
        y_array = _as_array(y).reshape(-1)
        t_array = _as_array(t).reshape(-1).astype(int)
        self.models_["m"] = _fit_estimator(self.outcome_estimator, x_frame, y_array, self.task)
        self.models_["propensity"] = _fit_estimator(self.propensity_estimator, x_frame, t_array, "classification")
        raw_e = _predict_outcome(self.models_["propensity"], x_frame, "classification")
        e = _clip_propensity(raw_e)
        pseudo = t_array * y_array / e - (1 - t_array) * y_array / (1 - e)
        weights = np.where(t_array == 1, 1 / e, 1 / (1 - e))
        weights = np.clip(weights, 0.1, np.quantile(weights, 0.95))
        self.models_["tau"] = _fit_estimator(self.effect_estimator, x_frame, pseudo, "regression", sample_weight=weights)
        self.nuisance_diagnostics_ = self._build_nuisance_diagnostics(
            method="ipw",
            treatment=t_array,
            propensity_raw=raw_e,
            propensity_clipped=e,
            pseudo=pseudo,
            weights=weights,
            cross_fit_folds=1,
            fold_rows=[],
            extra={
                "m": _array_summary(_predict_outcome(self.models_["m"], x_frame, self.task)),
                "tau_target": "inverse_probability_weighted_pseudo_effect",
            },
        )

    def _fit_domain_adaptation_learner(self, x, y, t) -> None:
        self.models_["propensity"] = _fit_estimator(self.propensity_estimator, x, t, "classification")
        e = _clip_propensity(_predict_outcome(self.models_["propensity"], x, "classification"))
        control_mask = t == 0
        treated_mask = t == 1
        control_weights = np.clip(e[control_mask] / (1 - e[control_mask]), 0.05, 20.0)
        treated_weights = np.clip((1 - e[treated_mask]) / e[treated_mask], 0.05, 20.0)
        self.models_["m0"] = _fit_estimator(
            self.outcome_estimator,
            x[control_mask],
            y[control_mask],
            self.task,
            sample_weight=control_weights,
        )
        self.models_["m1"] = _fit_estimator(
            self.outcome_estimator,
            x[treated_mask],
            y[treated_mask],
            self.task,
            sample_weight=treated_weights,
        )
        m0_on_treated = _predict_outcome(self.models_["m0"], x[treated_mask], self.task)
        m1_on_control = _predict_outcome(self.models_["m1"], x[control_mask], self.task)
        tau_treated = y[treated_mask] - m0_on_treated
        tau_control = m1_on_control - y[control_mask]
        tau_x = pd.concat([x[control_mask], x[treated_mask]], axis=0)
        tau_y = np.concatenate([tau_control, tau_treated])
        self.models_["tau"] = _fit_estimator(self.effect_estimator, tau_x, tau_y, "regression")

    def _effective_nuisance_folds(self, t) -> int:
        requested = int(self.params.get("nuisance_folds", self.params.get("cross_fit_folds", 1)) or 1)
        if bool(self.params.get("cross_fit", False)) and requested < 2:
            requested = 2
        if self.learner_type in {"orthogonal_dml", "orthogonal_random_forest"} and requested < 2:
            requested = 2
        if requested < 2:
            return 1
        t_array = _as_array(t).reshape(-1).astype(int)
        counts = np.bincount(t_array, minlength=2)
        max_folds = int(counts.min()) if counts.size >= 2 else 0
        if max_folds < 2:
            return 1
        return int(min(requested, max_folds))

    def _cross_fit_dr_nuisance(self, x, y, t, folds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        x_frame = _as_frame(x).reset_index(drop=True)
        y_array = _as_array(y).reshape(-1)
        t_array = _as_array(t).reshape(-1).astype(int)
        m0 = np.full(len(x_frame), np.nan)
        m1 = np.full(len(x_frame), np.nan)
        raw_e = np.full(len(x_frame), np.nan)
        fold_rows: list[dict] = []
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.params.get("random_state", 42))
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_frame, t_array), start=1):
            x_train = x_frame.iloc[train_idx]
            y_train = y_array[train_idx]
            t_train = t_array[train_idx]
            x_valid = x_frame.iloc[valid_idx]
            control_mask = t_train == 0
            treated_mask = t_train == 1
            model0 = _fit_estimator(self.outcome_estimator, x_train[control_mask], y_train[control_mask], self.task)
            model1 = _fit_estimator(self.outcome_estimator, x_train[treated_mask], y_train[treated_mask], self.task)
            propensity = _fit_estimator(self.propensity_estimator, x_train, t_train, "classification")
            m0[valid_idx] = _predict_outcome(model0, x_valid, self.task)
            m1[valid_idx] = _predict_outcome(model1, x_valid, self.task)
            raw_e[valid_idx] = _predict_outcome(propensity, x_valid, "classification")
            fold_rows.append(
                {
                    "fold": fold,
                    "train_rows": int(len(train_idx)),
                    "valid_rows": int(len(valid_idx)),
                    "train_treatment_rate": float(np.mean(t_train)),
                    "valid_treatment_rate": float(np.mean(t_array[valid_idx])),
                }
            )
        if not (np.isfinite(m0).all() and np.isfinite(m1).all() and np.isfinite(raw_e).all()):
            raise ValueError("Cross-fitted DR nuisance predictions contain non-finite values.")
        return m0, m1, raw_e, _clip_propensity(raw_e), fold_rows

    def _cross_fit_r_nuisance(self, x, y, t, folds: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        x_frame = _as_frame(x).reset_index(drop=True)
        y_array = _as_array(y).reshape(-1)
        t_array = _as_array(t).reshape(-1).astype(int)
        m = np.full(len(x_frame), np.nan)
        raw_e = np.full(len(x_frame), np.nan)
        fold_rows: list[dict] = []
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.params.get("random_state", 42))
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_frame, t_array), start=1):
            x_train = x_frame.iloc[train_idx]
            model_m = _fit_estimator(self.outcome_estimator, x_train, y_array[train_idx], self.task)
            propensity = _fit_estimator(self.propensity_estimator, x_train, t_array[train_idx], "classification")
            x_valid = x_frame.iloc[valid_idx]
            m[valid_idx] = _predict_outcome(model_m, x_valid, self.task)
            raw_e[valid_idx] = _predict_outcome(propensity, x_valid, "classification")
            fold_rows.append(
                {
                    "fold": fold,
                    "train_rows": int(len(train_idx)),
                    "valid_rows": int(len(valid_idx)),
                    "train_treatment_rate": float(np.mean(t_array[train_idx])),
                    "valid_treatment_rate": float(np.mean(t_array[valid_idx])),
                }
            )
        if not (np.isfinite(m).all() and np.isfinite(raw_e).all()):
            raise ValueError("Cross-fitted R nuisance predictions contain non-finite values.")
        return m, raw_e, _clip_propensity(raw_e), fold_rows

    def _build_nuisance_diagnostics(
        self,
        *,
        method: str,
        treatment: Any,
        propensity_raw: Any,
        propensity_clipped: Any,
        pseudo: Any,
        weights: Any | None = None,
        residuals: Dict[str, Any] | None = None,
        cross_fit_folds: int = 1,
        fold_rows: list[dict] | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        t_array = _as_array(treatment).reshape(-1).astype(int)
        raw_e = np.asarray(propensity_raw, dtype=float).reshape(-1)
        e = np.asarray(propensity_clipped, dtype=float).reshape(-1)
        ipw_weights = np.where(t_array == 1, 1 / e, 1 / (1 - e))
        diagnostics: Dict[str, Any] = {
            "schema_version": 1,
            "learner_type": self.learner_type,
            "method": method,
            "task": self.task,
            "cross_fit": {
                "enabled": int(cross_fit_folds) >= 2,
                "folds": int(cross_fit_folds),
                "fold_rows": fold_rows or [],
            },
            "sample": {
                "rows": int(len(t_array)),
                "treatment_rate": float(np.mean(t_array)) if len(t_array) else None,
                "treated_rows": int(np.sum(t_array == 1)),
                "control_rows": int(np.sum(t_array == 0)),
            },
            "propensity": {
                "raw": _array_summary(raw_e),
                "clipped": _array_summary(e),
                "clip_low_rate": float(np.mean(raw_e < 0.02)) if len(raw_e) else None,
                "clip_high_rate": float(np.mean(raw_e > 0.98)) if len(raw_e) else None,
                "weak_overlap_rate": float(np.mean((e < 0.05) | (e > 0.95))) if len(e) else None,
                "ipw_ess": _weight_effective_sample_size(ipw_weights),
            },
            "pseudo_outcome": _array_summary(pseudo),
            "nuisance_models": sorted(self.models_.keys()),
        }
        if weights is not None:
            diagnostics["weights"] = {
                "summary": _array_summary(weights),
                "ess": _weight_effective_sample_size(weights),
            }
        if residuals:
            diagnostics["residuals"] = {name: _array_summary(values) for name, values in residuals.items()}
        if extra:
            diagnostics["extra"] = extra
        return diagnostics

    def _fit_isotonic_tau_calibrator(self, x, y, t, base: str) -> None:
        raw_tau = self._raw_tau_for_calibration(x, base)
        pseudo, weights = self._pseudo_effect_for_calibration(x, y, t, base)
        finite = np.isfinite(raw_tau) & np.isfinite(pseudo) & np.isfinite(weights)
        if finite.sum() < 10 or len(np.unique(raw_tau[finite])) < 2:
            return
        calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
        calibrator.fit(raw_tau[finite], pseudo[finite], sample_weight=np.clip(weights[finite], 1e-4, None))
        self.models_["isotonic_tau"] = calibrator
        self.models_["isotonic_base"] = base

    def _raw_tau_for_calibration(self, x, base: str) -> np.ndarray:
        x_frame = _as_frame(x)
        if base == "dr":
            return np.asarray(self.models_["tau"].predict(x_frame)).reshape(-1)
        if base == "r":
            return np.asarray(self.models_["tau"].predict(x_frame)).reshape(-1)
        if base == "cvt":
            z_prob = _predict_outcome(self.models_["z"], x_frame, "classification")
            return 2 * np.asarray(z_prob).reshape(-1) - 1
        raise ValueError(f"Unsupported calibration base: {base}")

    def _pseudo_effect_for_calibration(self, x, y, t, base: str) -> tuple[np.ndarray, np.ndarray]:
        x_frame = _as_frame(x)
        y_array = _as_array(y).reshape(-1)
        t_array = _as_array(t).reshape(-1).astype(int)
        e = _clip_propensity(_predict_outcome(self.models_["propensity"], x_frame, "classification"))

        if base == "dr" and {"m0", "m1"}.issubset(self.models_):
            m0 = _predict_outcome(self.models_["m0"], x_frame, self.task)
            m1 = _predict_outcome(self.models_["m1"], x_frame, self.task)
            pseudo = (m1 - m0) + t_array * (y_array - m1) / e - (1 - t_array) * (y_array - m0) / (1 - e)
            weights = np.clip(e * (1 - e), 0.02, 0.25)
            return pseudo, weights

        if base == "r" and "m" in self.models_:
            m = _predict_outcome(self.models_["m"], x_frame, self.task)
            t_residual = t_array - e
            denom = np.where(np.abs(t_residual) < 0.05, np.sign(t_residual + 1e-12) * 0.05, t_residual)
            pseudo = (y_array - m) / denom
            weights = np.clip(t_residual**2, 1e-4, None)
            return pseudo, weights

        if base == "cvt":
            z = np.where(t_array == 1, y_array, 1 - y_array)
            pseudo = 2 * z - 1
            weights = np.where(t_array == 1, 0.5 / e, 0.5 / (1 - e))
            weights = np.clip(weights, 0.1, np.quantile(weights, 0.95))
            return pseudo, weights

        treatment_rate = float(np.clip(np.mean(t_array), 0.02, 0.98))
        pseudo = y_array * (t_array - treatment_rate) / (treatment_rate * (1 - treatment_rate))
        weights = np.ones_like(pseudo)
        return pseudo, weights

    def predict(self, x, t_true=None):
        x_frame = _as_frame(x)
        if self.learner_type in {"s", "s_forest", "s_lgbm", "s_xgb", "s_cat"}:
            y0 = _predict_outcome(self.models_["s"], _append_treatment(x_frame, 0), self.task)
            y1 = _predict_outcome(self.models_["s"], _append_treatment(x_frame, 1), self.task)
        elif self.learner_type in {"t", "t_forest", "t_lgbm", "t_xgb", "t_cat"}:
            y0, y1 = self._predict_t(x_frame)
        elif self.learner_type in {"x", "x_forest", "x_lgbm", "x_xgb", "x_cat"}:
            y0_base, y1_base = self._predict_t(x_frame)
            e = _clip_propensity(_predict_outcome(self.models_["propensity"], x_frame, "classification"))
            tau0 = self.models_["tau0"].predict(x_frame)
            tau1 = self.models_["tau1"].predict(x_frame)
            tau = e * tau0 + (1 - e) * tau1
            y0, y1 = y0_base, y0_base + tau
        elif self.learner_type in {"dr", "dr_forest", "dr_lgbm", "dr_xgb", "dr_cat", "pav_dr", "pav_dr_lgbm", "pav_dr_xgb", "pav_dr_cat", "ggbm", "ggbm_lgbm", "ggbm_xgb", "ggbm_cat"}:
            y0_base, y1_base = self._predict_t(x_frame)
            tau = self.models_["tau"].predict(x_frame)
            y0, y1 = y0_base, y0_base + tau
        elif self.learner_type in {"causal_forest", "pav_causal_forest"}:
            y0_base, _ = self._predict_t(x_frame)
            tau = self.models_["tau"].predict(x_frame)
            y0, y1 = y0_base, y0_base + tau
        elif self.learner_type in {"r", "r_forest", "r_lgbm", "r_xgb", "r_cat", "pav_r", "pav_r_lgbm", "pav_r_xgb", "pav_r_cat", "orthogonal_dml", "orthogonal_random_forest"}:
            m = _predict_outcome(self.models_["m"], x_frame, self.task)
            e = _clip_propensity(_predict_outcome(self.models_["propensity"], x_frame, "classification"))
            tau = self.models_["tau"].predict(x_frame)
            y0, y1 = m - e * tau, m + (1 - e) * tau
        elif self.learner_type in {"domain_adaptation", "domain_adaptation_lgbm", "domain_adaptation_xgb", "domain_adaptation_cat"}:
            y0_base, _ = self._predict_t(x_frame)
            tau = self.models_["tau"].predict(x_frame)
            y0, y1 = y0_base, y0_base + tau
        elif self.learner_type in {"transformed_outcome", "transformed_outcome_forest", "transformed_outcome_lgbm", "transformed_outcome_xgb", "transformed_outcome_cat"}:
            m = _predict_outcome(self.models_["m"], x_frame, self.task)
            e = np.full(len(x_frame), float(self.models_["constant_propensity"]))
            tau = self.models_["tau"].predict(x_frame)
            y0, y1 = m - e * tau, m + (1 - e) * tau
        elif self.learner_type in {"cvt", "cvt_forest", "pav_cvt"}:
            m = _predict_outcome(self.models_["m"], x_frame, self.task)
            e = _clip_propensity(_predict_outcome(self.models_["propensity"], x_frame, "classification"))
            z_prob = _predict_outcome(self.models_["z"], x_frame, "classification")
            tau = 2 * np.asarray(z_prob).reshape(-1) - 1
            y0, y1 = m - e * tau, m + (1 - e) * tau
        elif self.learner_type in {"ipw", "ipw_forest", "ipw_lgbm", "ipw_xgb", "ipw_cat"}:
            m = _predict_outcome(self.models_["m"], x_frame, self.task)
            e = _clip_propensity(_predict_outcome(self.models_["propensity"], x_frame, "classification"))
            tau = self.models_["tau"].predict(x_frame)
            y0, y1 = m - e * tau, m + (1 - e) * tau
        else:
            raise ValueError(f"Unknown learner type: {self.learner_type}")

        if "isotonic_tau" in self.models_:
            raw_tau = np.asarray(y1).reshape(-1) - np.asarray(y0).reshape(-1)
            calibrated_tau = self.models_["isotonic_tau"].predict(raw_tau)
            y1 = np.asarray(y0).reshape(-1) + calibrated_tau
            if self.task == "classification":
                y0 = np.clip(y0, 1e-6, 1 - 1e-6)
                y1 = np.clip(y1, 1e-6, 1 - 1e-6)

        return None, [np.asarray(y0).reshape(-1), np.asarray(y1).reshape(-1)]

    def _predict_t(self, x_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        y0 = _predict_outcome(self.models_["m0"], x_frame, self.task)
        y1 = _predict_outcome(self.models_["m1"], x_frame, self.task)
        return y0, y1

    def _observed_loss(self, x, y, t) -> float:
        _, y_preds = self.predict(x, t)
        observed = (1 - t) * y_preds[0] + t * y_preds[1]
        if self.task == "classification":
            return float(log_loss(y, np.clip(observed, 1e-6, 1 - 1e-6), labels=[0, 1]))
        return float(mean_squared_error(y, observed))


def _append_observed_treatment(x: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    enriched = x.copy()
    enriched["__treatment__"] = t
    return enriched
