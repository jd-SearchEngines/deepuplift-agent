from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Callable, Dict, Iterable, Tuple


@dataclass(frozen=True)
class ModelSpec:
    name: str
    supported_tasks: Tuple[str, ...]
    factory: Callable[[int, str, Dict[str, Any]], Tuple[Any, Callable[..., Any]]]
    dependencies: Tuple[str, ...] = ()


def _task_error(name: str, task: str, supported: Iterable[str]) -> ValueError:
    supported_text = ", ".join(supported)
    return ValueError(f"{name} supports task(s): {supported_text}; got '{task}'.")


def _build_tarnet(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.TarNet import TarNet, tarnet_loss

    model = TarNet(
        input_dim=input_dim,
        share_dim=params.get("share_dim", 12),
        share_hidden_dims=params.get("share_hidden_dims", [64, 64, 64]),
        base_hidden_dims=params.get("base_hidden_dims", [64, 64]),
        task=task,
    )
    return model, partial(tarnet_loss, task=task)


def _build_cfrnet(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.CFRNet import CFRNet, cfrnet_loss

    model = CFRNet(
        input_dim=input_dim,
        share_dim=params.get("share_dim", 12),
        share_hidden_dims=params.get("share_hidden_dims", [64, 64, 64]),
        base_hidden_dims=params.get("base_hidden_dims", [64, 64]),
        task=task,
    )
    return model, partial(
        cfrnet_loss,
        alpha=params.get("alpha", 1.0),
        task=task,
        ipm_mode=params.get("ipm_mode", "mmd_rbf"),
        ipm_sigma=params.get("ipm_sigma"),
    )


def _build_contrastive_uplift_net(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.ContrastiveUpliftNet import ContrastiveUpliftNet, contrastive_uplift_loss

    model = ContrastiveUpliftNet(
        input_dim=input_dim,
        share_dim=params.get("share_dim", 12),
        share_hidden_dims=params.get("share_hidden_dims", [64, 64]),
        base_hidden_dims=params.get("base_hidden_dims", [64]),
        task=task,
    )
    return model, partial(
        contrastive_uplift_loss,
        task=task,
        lambda_contrastive=params.get("lambda_contrastive", 0.05),
        propensity=params.get("propensity", 0.5),
        temperature=params.get("temperature", 0.2),
    )


def _build_dragonnet(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.DragonNet import DragonNet, dragonnet_loss

    model = DragonNet(
        input_dim=input_dim,
        share_dim=params.get("share_dim", 32),
        share_hidden_dims=params.get("share_hidden_dims", [64, 64, 64]),
        base_hidden_dims=params.get("base_hidden_dims", [64, 64]),
        task=task,
    )
    loss_f = partial(
        dragonnet_loss,
        alpha=params.get("alpha", 1.0),
        beta=params.get("beta", 1.0),
        tarreg=params.get("tarreg", True),
        task=task,
    )
    return model, loss_f


def _build_dragondeepfm(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.DragonDeepFM import DragonDeepFM
    from deepuplift.models.DragonNet import dragonnet_loss

    model = DragonDeepFM(
        input_dim=input_dim,
        num_continuous=params.get("num_continuous", input_dim),
        share_dim=params.get("share_dim", 32),
        embedding_size=params.get("embedding_size", 8),
        task=task,
        num_treatments=2,
    )
    loss_f = partial(
        dragonnet_loss,
        alpha=params.get("alpha", 1.0),
        beta=params.get("beta", 1.0),
        tarreg=params.get("tarreg", True),
        task=task,
    )
    return model, loss_f


def _build_efin(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.EFIN import EFIN, efin_loss

    model = EFIN(
        input_dim=input_dim,
        hc_dim=params.get("hc_dim", 64),
        hu_dim=params.get("hu_dim", 16),
        is_self=params.get("is_self", True),
        task=task,
        use_interaction_attention=params.get("use_interaction_attention", True),
        uplift_tau_scale=params.get("uplift_tau_scale", 1.0),
    )
    return model, partial(efin_loss, task=task)


def _build_descn(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.DESCN import ESX, esx_loss

    if task != "classification":
        raise _task_error("DESCN", task, ("classification",))
    model = ESX(
        input_dim=input_dim,
        share_dim=params.get("share_dim", 12),
        base_dim=params.get("base_dim", 12),
        task="classification",
        do_rate=params.get("do_rate", 0.2),
        use_batch_norm1d=params.get("use_batch_norm1d", True),
        normalization=params.get("normalization", "divide"),
    )
    return model, partial(
        esx_loss,
        task="classification",
        prpsy_w=params.get("prpsy_w", 1.0),
        escvr1_w=params.get("escvr1_w", 1.0),
        escvr0_w=params.get("escvr0_w", 1.0),
        h1_w=params.get("h1_w", 1.0),
        h0_w=params.get("h0_w", 1.0),
        mu1hat_w=params.get("mu1hat_w", 1.0),
        mu0hat_w=params.get("mu0hat_w", 1.0),
        imb_dist_w=params.get("imb_dist_w", 0.0),
        imb_dist=params.get("imb_dist", "wass"),
    )


def _build_euen(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.EUEN import EUEN, euen_loss

    if task != "regression":
        raise _task_error("EUEN", task, ("regression",))
    model = EUEN(
        input_dim=input_dim,
        hc_dim=params.get("hc_dim", 64),
        hu_dim=params.get("hu_dim", 64),
        is_self=params.get("is_self", False),
        task="regression",
    )
    return model, partial(euen_loss, task="regression")


def _build_eeuen(input_dim: int, task: str, params: Dict[str, Any]):
    from deepuplift.models.EEUEN import EEUEN, eeuen_loss

    model = EEUEN(
        input_dim=input_dim,
        hc_dim=params.get("hc_dim", 64),
        hu_dim=params.get("hu_dim", 64),
        he_dim=params.get("he_dim", 64),
        is_self=params.get("is_self", True),
        task=task,
    )
    loss_f = partial(
        eeuen_loss,
        alpha=params.get("alpha", 1.0),
        beta=params.get("beta", 1.0),
        task=task,
    )
    return model, loss_f


def _build_sklearn_uplift(learner_type: str):
    def factory(input_dim: int, task: str, params: Dict[str, Any]):
        from deepuplift.core.sklearn_models import SklearnUpliftModel

        return SklearnUpliftModel(learner_type=learner_type, task=task, params=params), None

    return factory


def _build_external_uplift(backend: str):
    def factory(input_dim: int, task: str, params: Dict[str, Any]):
        from deepuplift.core.external_models import ExternalUpliftModel

        return ExternalUpliftModel(backend=backend, task=task, params=params), None

    return factory


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "SLearnerGBM": ModelSpec("SLearnerGBM", ("classification", "regression"), _build_sklearn_uplift("s")),
    "SLearnerRF": ModelSpec("SLearnerRF", ("classification", "regression"), _build_sklearn_uplift("s_forest")),
    "SLearnerLightGBM": ModelSpec("SLearnerLightGBM", ("classification", "regression"), _build_sklearn_uplift("s_lgbm"), ("lightgbm",)),
    "SLearnerXGBoost": ModelSpec("SLearnerXGBoost", ("classification", "regression"), _build_sklearn_uplift("s_xgb"), ("xgboost",)),
    "SLearnerCatBoost": ModelSpec("SLearnerCatBoost", ("classification", "regression"), _build_sklearn_uplift("s_cat"), ("catboost",)),
    "TLearnerGBM": ModelSpec("TLearnerGBM", ("classification", "regression"), _build_sklearn_uplift("t")),
    "TLearnerRF": ModelSpec("TLearnerRF", ("classification", "regression"), _build_sklearn_uplift("t_forest")),
    "TLearnerLightGBM": ModelSpec("TLearnerLightGBM", ("classification", "regression"), _build_sklearn_uplift("t_lgbm"), ("lightgbm",)),
    "TLearnerXGBoost": ModelSpec("TLearnerXGBoost", ("classification", "regression"), _build_sklearn_uplift("t_xgb"), ("xgboost",)),
    "TLearnerCatBoost": ModelSpec("TLearnerCatBoost", ("classification", "regression"), _build_sklearn_uplift("t_cat"), ("catboost",)),
    "XLearnerGBM": ModelSpec("XLearnerGBM", ("classification", "regression"), _build_sklearn_uplift("x")),
    "XLearnerForest": ModelSpec("XLearnerForest", ("classification", "regression"), _build_sklearn_uplift("x_forest")),
    "XLearnerLightGBM": ModelSpec("XLearnerLightGBM", ("classification", "regression"), _build_sklearn_uplift("x_lgbm"), ("lightgbm",)),
    "XLearnerXGBoost": ModelSpec("XLearnerXGBoost", ("classification", "regression"), _build_sklearn_uplift("x_xgb"), ("xgboost",)),
    "XLearnerCatBoost": ModelSpec("XLearnerCatBoost", ("classification", "regression"), _build_sklearn_uplift("x_cat"), ("catboost",)),
    "DRLearnerGBM": ModelSpec("DRLearnerGBM", ("classification", "regression"), _build_sklearn_uplift("dr")),
    "DRLearnerForest": ModelSpec("DRLearnerForest", ("classification", "regression"), _build_sklearn_uplift("dr_forest")),
    "DRLearnerLightGBM": ModelSpec("DRLearnerLightGBM", ("classification", "regression"), _build_sklearn_uplift("dr_lgbm"), ("lightgbm",)),
    "DRLearnerXGBoost": ModelSpec("DRLearnerXGBoost", ("classification", "regression"), _build_sklearn_uplift("dr_xgb"), ("xgboost",)),
    "DRLearnerCatBoost": ModelSpec("DRLearnerCatBoost", ("classification", "regression"), _build_sklearn_uplift("dr_cat"), ("catboost",)),
    "DRForest": ModelSpec("DRForest", ("classification", "regression"), _build_sklearn_uplift("dr_forest")),
    "RLearnerGBM": ModelSpec("RLearnerGBM", ("classification", "regression"), _build_sklearn_uplift("r")),
    "RLearnerForest": ModelSpec("RLearnerForest", ("classification", "regression"), _build_sklearn_uplift("r_forest")),
    "RLearnerLightGBM": ModelSpec("RLearnerLightGBM", ("classification", "regression"), _build_sklearn_uplift("r_lgbm"), ("lightgbm",)),
    "RLearnerXGBoost": ModelSpec("RLearnerXGBoost", ("classification", "regression"), _build_sklearn_uplift("r_xgb"), ("xgboost",)),
    "RLearnerCatBoost": ModelSpec("RLearnerCatBoost", ("classification", "regression"), _build_sklearn_uplift("r_cat"), ("catboost",)),
    "OrthogonalDMLGBM": ModelSpec("OrthogonalDMLGBM", ("classification", "regression"), _build_sklearn_uplift("orthogonal_dml")),
    "OrthogonalRandomForest": ModelSpec("OrthogonalRandomForest", ("classification", "regression"), _build_sklearn_uplift("orthogonal_random_forest")),
    "CausalForest": ModelSpec("CausalForest", ("classification", "regression"), _build_sklearn_uplift("causal_forest")),
    "TransformedOutcomeGBM": ModelSpec("TransformedOutcomeGBM", ("classification", "regression"), _build_sklearn_uplift("transformed_outcome")),
    "TransformedOutcomeForest": ModelSpec("TransformedOutcomeForest", ("classification", "regression"), _build_sklearn_uplift("transformed_outcome_forest")),
    "TransformedOutcomeLightGBM": ModelSpec("TransformedOutcomeLightGBM", ("classification", "regression"), _build_sklearn_uplift("transformed_outcome_lgbm"), ("lightgbm",)),
    "TransformedOutcomeXGBoost": ModelSpec("TransformedOutcomeXGBoost", ("classification", "regression"), _build_sklearn_uplift("transformed_outcome_xgb"), ("xgboost",)),
    "TransformedOutcomeCatBoost": ModelSpec("TransformedOutcomeCatBoost", ("classification", "regression"), _build_sklearn_uplift("transformed_outcome_cat"), ("catboost",)),
    "ClassVariableTransformGBM": ModelSpec("ClassVariableTransformGBM", ("classification",), _build_sklearn_uplift("cvt")),
    "ClassVariableTransformForest": ModelSpec("ClassVariableTransformForest", ("classification",), _build_sklearn_uplift("cvt_forest")),
    "PAVCalibratedClassVariableTransformGBM": ModelSpec("PAVCalibratedClassVariableTransformGBM", ("classification",), _build_sklearn_uplift("pav_cvt")),
    "IPWLearnerGBM": ModelSpec("IPWLearnerGBM", ("classification", "regression"), _build_sklearn_uplift("ipw")),
    "IPWForest": ModelSpec("IPWForest", ("classification", "regression"), _build_sklearn_uplift("ipw_forest")),
    "IPWLightGBM": ModelSpec("IPWLightGBM", ("classification", "regression"), _build_sklearn_uplift("ipw_lgbm"), ("lightgbm",)),
    "IPWXGBoost": ModelSpec("IPWXGBoost", ("classification", "regression"), _build_sklearn_uplift("ipw_xgb"), ("xgboost",)),
    "IPWCatBoost": ModelSpec("IPWCatBoost", ("classification", "regression"), _build_sklearn_uplift("ipw_cat"), ("catboost",)),
    "DomainAdaptationLearner": ModelSpec("DomainAdaptationLearner", ("classification", "regression"), _build_sklearn_uplift("domain_adaptation")),
    "DomainAdaptationLightGBM": ModelSpec("DomainAdaptationLightGBM", ("classification", "regression"), _build_sklearn_uplift("domain_adaptation_lgbm"), ("lightgbm",)),
    "DomainAdaptationXGBoost": ModelSpec("DomainAdaptationXGBoost", ("classification", "regression"), _build_sklearn_uplift("domain_adaptation_xgb"), ("xgboost",)),
    "DomainAdaptationCatBoost": ModelSpec("DomainAdaptationCatBoost", ("classification", "regression"), _build_sklearn_uplift("domain_adaptation_cat"), ("catboost",)),
    "GGBMUpliftGBM": ModelSpec("GGBMUpliftGBM", ("classification", "regression"), _build_sklearn_uplift("ggbm")),
    "GGBMUpliftLightGBM": ModelSpec("GGBMUpliftLightGBM", ("classification", "regression"), _build_sklearn_uplift("ggbm_lgbm"), ("lightgbm",)),
    "GGBMUpliftXGBoost": ModelSpec("GGBMUpliftXGBoost", ("classification", "regression"), _build_sklearn_uplift("ggbm_xgb"), ("xgboost",)),
    "GGBMUpliftCatBoost": ModelSpec("GGBMUpliftCatBoost", ("classification", "regression"), _build_sklearn_uplift("ggbm_cat"), ("catboost",)),
    "PAVCalibratedDRLearnerGBM": ModelSpec("PAVCalibratedDRLearnerGBM", ("classification", "regression"), _build_sklearn_uplift("pav_dr")),
    "PAVCalibratedDRLearnerLightGBM": ModelSpec("PAVCalibratedDRLearnerLightGBM", ("classification", "regression"), _build_sklearn_uplift("pav_dr_lgbm"), ("lightgbm",)),
    "PAVCalibratedDRLearnerXGBoost": ModelSpec("PAVCalibratedDRLearnerXGBoost", ("classification", "regression"), _build_sklearn_uplift("pav_dr_xgb"), ("xgboost",)),
    "PAVCalibratedDRLearnerCatBoost": ModelSpec("PAVCalibratedDRLearnerCatBoost", ("classification", "regression"), _build_sklearn_uplift("pav_dr_cat"), ("catboost",)),
    "PAVCalibratedRLearnerGBM": ModelSpec("PAVCalibratedRLearnerGBM", ("classification", "regression"), _build_sklearn_uplift("pav_r")),
    "PAVCalibratedRLearnerLightGBM": ModelSpec("PAVCalibratedRLearnerLightGBM", ("classification", "regression"), _build_sklearn_uplift("pav_r_lgbm"), ("lightgbm",)),
    "PAVCalibratedRLearnerXGBoost": ModelSpec("PAVCalibratedRLearnerXGBoost", ("classification", "regression"), _build_sklearn_uplift("pav_r_xgb"), ("xgboost",)),
    "PAVCalibratedRLearnerCatBoost": ModelSpec("PAVCalibratedRLearnerCatBoost", ("classification", "regression"), _build_sklearn_uplift("pav_r_cat"), ("catboost",)),
    "PAVCalibratedCausalForest": ModelSpec("PAVCalibratedCausalForest", ("classification", "regression"), _build_sklearn_uplift("pav_causal_forest")),
    "TarNet": ModelSpec("TarNet", ("classification", "regression"), _build_tarnet),
    "CFRNet": ModelSpec("CFRNet", ("classification", "regression"), _build_cfrnet),
    "ContrastiveUpliftNet": ModelSpec("ContrastiveUpliftNet", ("classification", "regression"), _build_contrastive_uplift_net),
    "DragonNet": ModelSpec("DragonNet", ("classification", "regression"), _build_dragonnet),
    "DragonDeepFM": ModelSpec("DragonDeepFM", ("classification", "regression"), _build_dragondeepfm),
    "EFIN": ModelSpec("EFIN", ("classification", "regression"), _build_efin),
    "DESCN": ModelSpec("DESCN", ("classification",), _build_descn),
    "ESX": ModelSpec("ESX", ("classification",), _build_descn),
    "EUEN": ModelSpec("EUEN", ("regression",), _build_euen),
    "EEUEN": ModelSpec("EEUEN", ("classification", "regression"), _build_eeuen),
    "EconMLCausalForestDML": ModelSpec(
        "EconMLCausalForestDML",
        ("classification", "regression"),
        _build_external_uplift("econml_causal_forest_dml"),
        ("econml",),
    ),
    "EconMLLinearDML": ModelSpec(
        "EconMLLinearDML",
        ("classification", "regression"),
        _build_external_uplift("econml_linear_dml"),
        ("econml",),
    ),
    "EconMLSparseLinearDML": ModelSpec(
        "EconMLSparseLinearDML",
        ("classification", "regression"),
        _build_external_uplift("econml_sparse_linear_dml"),
        ("econml",),
    ),
    "EconMLKernelDML": ModelSpec(
        "EconMLKernelDML",
        ("classification", "regression"),
        _build_external_uplift("econml_kernel_dml"),
        ("econml",),
    ),
    "EconMLNonParamDML": ModelSpec(
        "EconMLNonParamDML",
        ("classification", "regression"),
        _build_external_uplift("econml_nonparam_dml"),
        ("econml",),
    ),
    "EconMLDRLearner": ModelSpec(
        "EconMLDRLearner",
        ("classification", "regression"),
        _build_external_uplift("econml_dr_learner"),
        ("econml",),
    ),
    "EconMLLinearDRLearner": ModelSpec(
        "EconMLLinearDRLearner",
        ("classification", "regression"),
        _build_external_uplift("econml_linear_dr_learner"),
        ("econml",),
    ),
    "EconMLSparseLinearDRLearner": ModelSpec(
        "EconMLSparseLinearDRLearner",
        ("classification", "regression"),
        _build_external_uplift("econml_sparse_linear_dr_learner"),
        ("econml",),
    ),
    "EconMLForestDRLearner": ModelSpec(
        "EconMLForestDRLearner",
        ("classification", "regression"),
        _build_external_uplift("econml_forest_dr_learner"),
        ("econml",),
    ),
    "EconMLDMLOrthoForest": ModelSpec(
        "EconMLDMLOrthoForest",
        ("classification", "regression"),
        _build_external_uplift("econml_dml_ortho_forest"),
        ("econml",),
    ),
    "EconMLDROrthoForest": ModelSpec(
        "EconMLDROrthoForest",
        ("classification", "regression"),
        _build_external_uplift("econml_dr_ortho_forest"),
        ("econml",),
    ),
    "EconMLGRFCausalForest": ModelSpec(
        "EconMLGRFCausalForest",
        ("classification", "regression"),
        _build_external_uplift("econml_grf_causal_forest"),
        ("econml",),
    ),
    "CausalMLUpliftTree": ModelSpec(
        "CausalMLUpliftTree",
        ("classification",),
        _build_external_uplift("causalml_uplift_tree"),
        ("causalml",),
    ),
    "CausalMLUpliftRandomForest": ModelSpec(
        "CausalMLUpliftRandomForest",
        ("classification",),
        _build_external_uplift("causalml_uplift_rf"),
        ("causalml",),
    ),
    "CausalMLCausalRandomForest": ModelSpec(
        "CausalMLCausalRandomForest",
        ("classification", "regression"),
        _build_external_uplift("causalml_causal_forest"),
        ("causalml",),
    ),
    "CausalMLSLearnerLightGBM": ModelSpec(
        "CausalMLSLearnerLightGBM",
        ("classification", "regression"),
        _build_external_uplift("causalml_meta_s_lgbm"),
        ("causalml", "lightgbm"),
    ),
    "CausalMLTLearnerLightGBM": ModelSpec(
        "CausalMLTLearnerLightGBM",
        ("classification", "regression"),
        _build_external_uplift("causalml_meta_t_lgbm"),
        ("causalml", "lightgbm"),
    ),
    "CausalMLXLearnerLightGBM": ModelSpec(
        "CausalMLXLearnerLightGBM",
        ("classification", "regression"),
        _build_external_uplift("causalml_meta_x_lgbm"),
        ("causalml", "lightgbm"),
    ),
    "CausalMLRLearnerLightGBM": ModelSpec(
        "CausalMLRLearnerLightGBM",
        ("classification", "regression"),
        _build_external_uplift("causalml_meta_r_lgbm"),
        ("causalml", "lightgbm"),
    ),
    "SkLiftSoloGBM": ModelSpec(
        "SkLiftSoloGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_solo_gbm"),
        ("sklift",),
    ),
    "SkLiftSoloLightGBM": ModelSpec(
        "SkLiftSoloLightGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_solo_lgbm"),
        ("sklift", "lightgbm"),
    ),
    "SkLiftSoloInteractionGBM": ModelSpec(
        "SkLiftSoloInteractionGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_solo_interaction_gbm"),
        ("sklift",),
    ),
    "SkLiftSoloInteractionLightGBM": ModelSpec(
        "SkLiftSoloInteractionLightGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_solo_interaction_lgbm"),
        ("sklift", "lightgbm"),
    ),
    "SkLiftTwoModelsGBM": ModelSpec(
        "SkLiftTwoModelsGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_two_gbm"),
        ("sklift",),
    ),
    "SkLiftTwoModelsLightGBM": ModelSpec(
        "SkLiftTwoModelsLightGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_two_lgbm"),
        ("sklift", "lightgbm"),
    ),
    "SkLiftTwoModelsDDRControlGBM": ModelSpec(
        "SkLiftTwoModelsDDRControlGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_two_ddr_control_gbm"),
        ("sklift",),
    ),
    "SkLiftTwoModelsDDRControlLightGBM": ModelSpec(
        "SkLiftTwoModelsDDRControlLightGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_two_ddr_control_lgbm"),
        ("sklift", "lightgbm"),
    ),
    "SkLiftTwoModelsDDRTreatmentGBM": ModelSpec(
        "SkLiftTwoModelsDDRTreatmentGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_two_ddr_treatment_gbm"),
        ("sklift",),
    ),
    "SkLiftTwoModelsDDRTreatmentLightGBM": ModelSpec(
        "SkLiftTwoModelsDDRTreatmentLightGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_two_ddr_treatment_lgbm"),
        ("sklift", "lightgbm"),
    ),
    "SkLiftClassTransformationGBM": ModelSpec(
        "SkLiftClassTransformationGBM",
        ("classification",),
        _build_external_uplift("sklift_class_transformation_gbm"),
        ("sklift",),
    ),
    "SkLiftClassTransformationLightGBM": ModelSpec(
        "SkLiftClassTransformationLightGBM",
        ("classification",),
        _build_external_uplift("sklift_class_transformation_lgbm"),
        ("sklift", "lightgbm"),
    ),
    "SkLiftTransformedOutcomeGBM": ModelSpec(
        "SkLiftTransformedOutcomeGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_transformed_outcome_gbm"),
        ("sklift",),
    ),
    "SkLiftTransformedOutcomeLightGBM": ModelSpec(
        "SkLiftTransformedOutcomeLightGBM",
        ("classification", "regression"),
        _build_external_uplift("sklift_transformed_outcome_lgbm"),
        ("sklift", "lightgbm"),
    ),
    "UTBoostGBM": ModelSpec(
        "UTBoostGBM",
        ("classification", "regression"),
        _build_external_uplift("utboost_gbm"),
        ("utboost",),
    ),
}


def _short_dependency_error(exc: Exception, limit: int = 180) -> str:
    text = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    return text if len(text) <= limit else text[: limit - 3] + "..."


@lru_cache(maxsize=None)
def _dependency_import_issue(dependency: str) -> str | None:
    if importlib.util.find_spec(dependency) is None:
        return dependency
    try:
        importlib.import_module(dependency)
    except Exception as exc:
        return f"{dependency} import failed: {_short_dependency_error(exc)}"
    return None


def missing_dependencies(model_name: str) -> list[str]:
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        return []
    missing = [issue for dep in spec.dependencies if (issue := _dependency_import_issue(dep))]
    if "xgboost" in spec.dependencies and os.environ.get("DEEPUPLIFT_ENABLE_XGBOOST") != "1":
        missing.append("set DEEPUPLIFT_ENABLE_XGBOOST=1 (experimental; disabled by default)")
    if "causalml" in spec.dependencies and sys.version_info < (3, 11):
        missing.append("python>=3.11 for causalml>=0.16")
    return missing


def is_model_available(model_name: str) -> bool:
    return not missing_dependencies(model_name)


def available_models(task: str | None = None, include_external: bool = True, only_available: bool = False):
    def allowed(name: str, spec: ModelSpec) -> bool:
        if task is not None and task not in spec.supported_tasks:
            return False
        if not include_external and spec.dependencies:
            return False
        if only_available and not is_model_available(name):
            return False
        return True

    if task is None:
        return sorted(name for name, spec in MODEL_REGISTRY.items() if allowed(name, spec))
    return sorted(name for name, spec in MODEL_REGISTRY.items() if allowed(name, spec))


def build_model(model_name: str, input_dim: int, task: str, model_params: Dict[str, Any] | None = None):
    if model_name not in MODEL_REGISTRY:
        options = ", ".join(available_models())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {options}.")
    spec = MODEL_REGISTRY[model_name]
    if task not in spec.supported_tasks:
        raise _task_error(spec.name, task, spec.supported_tasks)
    return spec.factory(input_dim, task, model_params or {})
