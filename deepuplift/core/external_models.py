from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_squared_error

from .sklearn_models import (
    _as_array,
    _as_frame,
    _clip_propensity,
    _default_effect_estimator,
    _default_outcome_estimator,
    _fit_estimator,
    _lightgbm_effect_estimator,
    _lightgbm_outcome_estimator,
    _predict_outcome,
)


def _missing_dependency(package: str, model_name: str) -> ImportError:
    return ImportError(
        f"{model_name} requires optional package '{package}'. "
        f"Install it in the app environment before training this model."
    )


def _flatten_effect(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1:
        return array.reshape(-1)
    if array.ndim == 2:
        return array[:, 0].reshape(-1)
    return array.reshape(array.shape[0], -1)[:, 0]


@dataclass
class ExternalUpliftModel:
    backend: str
    task: str = "classification"
    params: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.params = self.params or {}
        self.models_: Dict[str, Any] = {}
        self.history_: list[dict] = []

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

        self.models_["m"] = _fit_estimator(
            _default_outcome_estimator(self.task, self.params),
            x_frame,
            y_array,
            self.task,
        )
        self.models_["propensity"] = _fit_estimator(LogisticRegression(max_iter=1000), x_frame, t_array, "classification")

        if self.backend.startswith("econml_"):
            self._fit_econml(x_frame, y_array, t_array)
        elif self.backend.startswith("causalml_"):
            self._fit_causalml(x_frame, y_array, t_array)
        elif self.backend.startswith("sklift_"):
            self._fit_sklift(x_frame, y_array, t_array)
        elif self.backend.startswith("utboost_"):
            self._fit_utboost(x_frame, y_array, t_array)
        else:
            raise ValueError(f"Unknown external backend: {self.backend}")

        train_loss = self._observed_loss(x_frame, y_array, t_array)
        metrics = {"epoch": 0, "train_loss": train_loss, "train_outcome_loss": train_loss, "train_treatment_loss": 0.0}
        self.history_ = [metrics]
        if callback:
            callback(metrics)
        return self.history_

    def _fit_econml(self, x, y, t) -> None:
        dml_model_y = _default_effect_estimator(self.params, forest=False)
        dr_model_y = _default_outcome_estimator(self.task, self.params)
        model_t = LogisticRegression(max_iter=1000)
        model_final = _default_effect_estimator(self.params, forest=self.params.get("forest_final", True))
        cv = int(self.params.get("cv", 3))
        random_state = int(self.params.get("random_state", 42))

        if self.backend == "econml_causal_forest_dml":
            try:
                from econml.dml import CausalForestDML
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLCausalForestDML") from exc
            estimator = CausalForestDML(
                model_y=dml_model_y,
                model_t=model_t,
                discrete_treatment=True,
                n_estimators=int(self.params.get("n_estimators", 200)),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 10)),
                cv=cv,
                random_state=random_state,
            )
        elif self.backend == "econml_linear_dml":
            try:
                from econml.dml import LinearDML
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLLinearDML") from exc
            estimator = LinearDML(
                model_y=dml_model_y,
                model_t=model_t,
                discrete_treatment=True,
                cv=cv,
                random_state=random_state,
            )
        elif self.backend == "econml_sparse_linear_dml":
            try:
                from econml.dml import SparseLinearDML
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLSparseLinearDML") from exc
            estimator = SparseLinearDML(
                model_y=dml_model_y,
                model_t=model_t,
                discrete_outcome=self.task == "classification",
                discrete_treatment=True,
                cv=cv,
                random_state=random_state,
                n_jobs=int(self.params.get("n_jobs", 1)),
            )
        elif self.backend == "econml_kernel_dml":
            try:
                from econml.dml import KernelDML
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLKernelDML") from exc
            estimator = KernelDML(
                model_y=dml_model_y,
                model_t=model_t,
                discrete_outcome=self.task == "classification",
                discrete_treatment=True,
                dim=int(self.params.get("dim", 20)),
                bw=float(self.params.get("bw", 1.0)),
                cv=cv,
                random_state=random_state,
            )
        elif self.backend == "econml_nonparam_dml":
            try:
                from econml.dml import NonParamDML
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLNonParamDML") from exc
            estimator = NonParamDML(
                model_y=dml_model_y,
                model_t=model_t,
                model_final=model_final,
                discrete_treatment=True,
                cv=cv,
                random_state=random_state,
            )
        elif self.backend == "econml_dr_learner":
            try:
                from econml.dr import DRLearner
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLDRLearner") from exc
            estimator = DRLearner(
                model_regression=dr_model_y,
                model_propensity=model_t,
                model_final=model_final,
                discrete_outcome=self.task == "classification",
                cv=cv,
                random_state=random_state,
            )
        elif self.backend == "econml_linear_dr_learner":
            try:
                from econml.dr import LinearDRLearner
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLLinearDRLearner") from exc
            estimator = LinearDRLearner(
                model_regression=dr_model_y,
                model_propensity=model_t,
                discrete_outcome=self.task == "classification",
                cv=cv,
                random_state=random_state,
            )
        elif self.backend == "econml_sparse_linear_dr_learner":
            try:
                from econml.dr import SparseLinearDRLearner
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLSparseLinearDRLearner") from exc
            estimator = SparseLinearDRLearner(
                model_regression=dr_model_y,
                model_propensity=model_t,
                discrete_outcome=self.task == "classification",
                cv=cv,
                random_state=random_state,
                n_jobs=int(self.params.get("n_jobs", 1)),
            )
        elif self.backend == "econml_forest_dr_learner":
            try:
                from econml.dr import ForestDRLearner
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLForestDRLearner") from exc
            estimator = ForestDRLearner(
                model_regression=dr_model_y,
                model_propensity=model_t,
                discrete_outcome=self.task == "classification",
                n_estimators=int(self.params.get("n_estimators", 200)),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 10)),
                cv=cv,
                random_state=random_state,
            )
        elif self.backend == "econml_dml_ortho_forest":
            try:
                from econml.orf import DMLOrthoForest
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLDMLOrthoForest") from exc
            estimator = DMLOrthoForest(
                n_trees=int(self.params.get("n_estimators", self.params.get("n_trees", 100))),
                min_leaf_size=int(self.params.get("min_samples_leaf", self.params.get("min_leaf_size", 10))),
                max_depth=int(self.params.get("max_depth", 10)),
                model_T=model_t,
                model_Y=_default_effect_estimator(self.params, forest=False),
                discrete_treatment=True,
                n_jobs=int(self.params.get("n_jobs", 1)),
                random_state=random_state,
                verbose=int(self.params.get("verbose", 0)),
            )
        elif self.backend == "econml_dr_ortho_forest":
            try:
                from econml.orf import DROrthoForest
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLDROrthoForest") from exc
            estimator = DROrthoForest(
                n_trees=int(self.params.get("n_estimators", self.params.get("n_trees", 100))),
                min_leaf_size=int(self.params.get("min_samples_leaf", self.params.get("min_leaf_size", 10))),
                max_depth=int(self.params.get("max_depth", 10)),
                propensity_model=model_t,
                model_Y=_default_effect_estimator(self.params, forest=False),
                n_jobs=int(self.params.get("n_jobs", 1)),
                random_state=random_state,
                verbose=int(self.params.get("verbose", 0)),
            )
        elif self.backend == "econml_grf_causal_forest":
            try:
                from econml.grf import CausalForest
            except ImportError as exc:
                raise _missing_dependency("econml", "EconMLGRFCausalForest") from exc
            estimator = CausalForest(
                n_estimators=int(self.params.get("n_estimators", 100)),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 5)),
                max_depth=self.params.get("max_depth"),
                n_jobs=int(self.params.get("n_jobs", 1)),
                random_state=random_state,
                verbose=int(self.params.get("verbose", 0)),
            )
        else:
            raise ValueError(f"Unsupported EconML backend: {self.backend}")

        if self.backend == "econml_grf_causal_forest":
            estimator.fit(x, t, y)
        else:
            estimator.fit(y, t, X=x)
        self.models_["external"] = estimator

    def _fit_causalml(self, x, y, t) -> None:
        if self.backend.startswith("causalml_meta_"):
            self._fit_causalml_meta(x, y, t)
            return

        if self.backend in {"causalml_uplift_tree", "causalml_uplift_rf"} and self.task != "classification":
            raise ValueError(f"{self.backend} supports classification outcomes only.")

        if self.backend == "causalml_uplift_tree":
            try:
                from causalml.inference.tree import UpliftTreeClassifier
            except ImportError as exc:
                raise _missing_dependency("causalml", "CausalMLUpliftTree") from exc
            estimator = UpliftTreeClassifier(control_name=0, max_depth=self.params.get("max_depth", 5))
            estimator.fit(x.to_numpy(), treatment=t, y=y)
        elif self.backend == "causalml_uplift_rf":
            try:
                from causalml.inference.tree import UpliftRandomForestClassifier
            except ImportError as exc:
                raise _missing_dependency("causalml", "CausalMLUpliftRandomForest") from exc
            estimator = UpliftRandomForestClassifier(
                control_name=0,
                n_estimators=int(self.params.get("n_estimators", 100)),
                max_depth=self.params.get("max_depth", None),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 100)),
                random_state=int(self.params.get("random_state", 42)),
            )
            estimator.fit(x.to_numpy(), treatment=t, y=y)
        elif self.backend == "causalml_causal_forest":
            try:
                from causalml.inference.tree import CausalRandomForestRegressor
            except ImportError as exc:
                raise _missing_dependency("causalml", "CausalMLCausalRandomForest") from exc
            estimator = CausalRandomForestRegressor(
                control_name=0,
                n_estimators=int(self.params.get("n_estimators", 100)),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 100)),
                random_state=int(self.params.get("random_state", 42)),
            )
            estimator.fit(x.to_numpy(), treatment=t, y=y)
        else:
            raise ValueError(f"Unsupported CausalML backend: {self.backend}")

        self.models_["external"] = estimator

    def _fit_causalml_meta(self, x, y, t) -> None:
        try:
            from causalml.inference.meta import (
                BaseRClassifier,
                BaseRRegressor,
                BaseSClassifier,
                BaseSRegressor,
                BaseTClassifier,
                BaseTRegressor,
                BaseXClassifier,
                BaseXRegressor,
            )
        except ImportError as exc:
            raise _missing_dependency("causalml", "CausalML meta-learners") from exc

        learner_map = {
            ("s", "classification"): BaseSClassifier,
            ("s", "regression"): BaseSRegressor,
            ("t", "classification"): BaseTClassifier,
            ("t", "regression"): BaseTRegressor,
            ("x", "classification"): BaseXClassifier,
            ("x", "regression"): BaseXRegressor,
            ("r", "classification"): BaseRClassifier,
            ("r", "regression"): BaseRRegressor,
        }
        learner_key = self.backend.replace("causalml_meta_", "").split("_", 1)[0]
        learner_cls = learner_map[(learner_key, self.task)]
        base = "lgbm" if self.backend.endswith("_lgbm") else "gbm"
        base_learner = self._sklift_outcome_estimator(base, self.task)
        estimator = learner_cls(learner=base_learner, control_name="control")
        treatment = np.where(t == 1, "treatment", "control")
        kwargs = {"X": x.to_numpy(), "treatment": treatment, "y": y}
        if learner_key in {"x", "r"}:
            kwargs["p"] = _clip_propensity(_predict_outcome(self.models_["propensity"], x, "classification"))
        cate = estimator.fit_predict(**kwargs)
        self.models_["external"] = estimator
        self.models_["train_tau"] = _flatten_effect(cate)

    def _sklift_outcome_estimator(self, base: str, task: str):
        if base == "lgbm":
            return _lightgbm_outcome_estimator(task, self.params)
        return _default_outcome_estimator(task, self.params)

    def _sklift_effect_estimator(self, base: str):
        if base == "lgbm":
            return _lightgbm_effect_estimator(self.params)
        return _default_effect_estimator(self.params, forest=False)

    def _fit_sklift(self, x, y, t) -> None:
        try:
            from sklift.models import ClassTransformation, ClassTransformationReg, SoloModel, TwoModels
        except ImportError as exc:
            raise _missing_dependency("scikit-uplift", "scikit-uplift adapters") from exc

        base = "lgbm" if self.backend.endswith("_lgbm") else "gbm"
        if self.backend.startswith("sklift_solo_interaction"):
            estimator = SoloModel(self._sklift_outcome_estimator(base, self.task), method="treatment_interaction")
        elif self.backend.startswith("sklift_solo"):
            estimator = SoloModel(self._sklift_outcome_estimator(base, self.task), method="dummy")
        elif self.backend.startswith("sklift_two_ddr_control"):
            estimator = TwoModels(
                estimator_trmnt=self._sklift_outcome_estimator(base, self.task),
                estimator_ctrl=self._sklift_outcome_estimator(base, self.task),
                method="ddr_control",
            )
        elif self.backend.startswith("sklift_two_ddr_treatment"):
            estimator = TwoModels(
                estimator_trmnt=self._sklift_outcome_estimator(base, self.task),
                estimator_ctrl=self._sklift_outcome_estimator(base, self.task),
                method="ddr_treatment",
            )
        elif self.backend.startswith("sklift_two"):
            estimator = TwoModels(
                estimator_trmnt=self._sklift_outcome_estimator(base, self.task),
                estimator_ctrl=self._sklift_outcome_estimator(base, self.task),
                method="vanilla",
            )
        elif self.backend.startswith("sklift_class_transformation"):
            if self.task != "classification":
                raise ValueError("scikit-uplift ClassTransformation supports classification outcomes only.")
            estimator = ClassTransformation(self._sklift_outcome_estimator(base, "classification"))
        elif self.backend.startswith("sklift_transformed_outcome"):
            estimator = ClassTransformationReg(
                estimator=self._sklift_effect_estimator(base),
                propensity_estimator=LogisticRegression(max_iter=1000),
            )
        else:
            raise ValueError(f"Unsupported scikit-uplift backend: {self.backend}")

        estimator.fit(x, y, t)
        self.models_["external"] = estimator

    def _fit_utboost(self, x, y, t) -> None:
        try:
            from utboost import UTBClassifier, UTBRegressor
        except ImportError as exc:
            raise _missing_dependency("utboost", "UTBoostGBM") from exc

        estimator_cls = UTBClassifier if self.task == "classification" else UTBRegressor
        estimator = estimator_cls(
            ensemble_type=self.params.get("ensemble_type", "boosting"),
            criterion=self.params.get("criterion", "gbm"),
            iterations=int(self.params.get("iterations", self.params.get("n_estimators", 80))),
            max_depth=int(self.params.get("max_depth", 4)),
            learning_rate=float(self.params.get("learning_rate", 0.05)),
        )
        estimator.fit(X=x, ti=t.astype(int), y=y)
        self.models_["external"] = estimator

    def predict(self, x, t_true=None):
        x_frame = _as_frame(x)
        if self.backend.startswith("sklift_"):
            y0, y1 = self._predict_sklift_outcomes(x_frame)
            return None, [y0, y1]
        if self.backend.startswith("utboost_"):
            y0, y1 = self._predict_utboost_outcomes(x_frame)
            return None, [y0, y1]
        tau = self._predict_effect(x_frame)
        m = _predict_outcome(self.models_["m"], x_frame, self.task)
        e = _clip_propensity(_predict_outcome(self.models_["propensity"], x_frame, "classification"))
        y0 = m - e * tau
        y1 = m + (1 - e) * tau
        return None, [np.asarray(y0).reshape(-1), np.asarray(y1).reshape(-1)]

    def _predict_sklift_outcomes(self, x) -> tuple[np.ndarray, np.ndarray]:
        estimator = self.models_["external"]
        tau = np.asarray(estimator.predict(x)).reshape(-1)
        ctrl = getattr(estimator, "ctrl_preds_", None)
        trmnt = getattr(estimator, "trmnt_preds_", None)
        if ctrl is not None and trmnt is not None:
            y0 = np.asarray(ctrl).reshape(-1)
            y1 = np.asarray(trmnt).reshape(-1)
        else:
            m = _predict_outcome(self.models_["m"], x, self.task)
            e = _clip_propensity(_predict_outcome(self.models_["propensity"], x, "classification"))
            y0 = m - e * tau
            y1 = m + (1 - e) * tau
        if self.task == "classification":
            y0 = np.clip(y0, 1e-6, 1 - 1e-6)
            y1 = np.clip(y1, 1e-6, 1 - 1e-6)
        return np.asarray(y0).reshape(-1), np.asarray(y1).reshape(-1)

    def _predict_utboost_outcomes(self, x) -> tuple[np.ndarray, np.ndarray]:
        estimator = self.models_["external"]
        prediction = np.asarray(estimator.predict(x))
        if prediction.ndim == 1:
            tau = prediction.reshape(-1)
            m = _predict_outcome(self.models_["m"], x, self.task)
            e = _clip_propensity(_predict_outcome(self.models_["propensity"], x, "classification"))
            y0 = m - e * tau
            y1 = m + (1 - e) * tau
        elif prediction.shape[1] >= 2:
            y0 = prediction[:, 0]
            y1 = prediction[:, 1]
        else:
            raise ValueError("UTBoost prediction must return uplift or at least two potential-outcome columns.")
        if self.task == "classification":
            y0 = np.clip(y0, 1e-6, 1 - 1e-6)
            y1 = np.clip(y1, 1e-6, 1 - 1e-6)
        return np.asarray(y0).reshape(-1), np.asarray(y1).reshape(-1)

    def _predict_effect(self, x) -> np.ndarray:
        estimator = self.models_["external"]
        if self.backend.startswith("econml_"):
            if self.backend == "econml_grf_causal_forest":
                return _flatten_effect(estimator.predict(x))
            return _flatten_effect(estimator.effect(x))
        if self.backend.startswith("sklift_"):
            return _flatten_effect(estimator.predict(x))
        if self.backend.startswith("causalml_meta_"):
            return _flatten_effect(estimator.predict(x.to_numpy()))
        if self.backend == "causalml_causal_forest":
            prediction = estimator.predict(x.to_numpy(), with_outcomes=True)
            array = np.asarray(prediction)
            if array.ndim == 2 and array.shape[1] >= 3:
                return array[:, -1].reshape(-1)
            return _flatten_effect(array)
        prediction = estimator.predict(x.to_numpy())
        return _flatten_effect(prediction)

    def _observed_loss(self, x, y, t) -> float:
        _, y_preds = self.predict(x, t)
        observed = (1 - t) * y_preds[0] + t * y_preds[1]
        if self.task == "classification":
            return float(log_loss(y, np.clip(observed, 1e-6, 1 - 1e-6), labels=[0, 1]))
        return float(mean_squared_error(y, observed))
