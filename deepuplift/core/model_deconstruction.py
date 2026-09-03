from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ExternalSource:
    name: str
    url: str
    source_type: str
    license: str
    license_risk: str
    adoption: str


@dataclass(frozen=True)
class ModelDeconstruction:
    model_id: str
    display_name: str
    family: str
    role: str
    current_status: str
    code_paths: tuple[str, ...]
    formulas: tuple[str, ...]
    call_chain: tuple[str, ...]
    data_flow: tuple[str, ...]
    train_flow: tuple[str, ...]
    predict_flow: tuple[str, ...]
    eval_metrics: tuple[str, ...]
    smoke_evidence: tuple[str, ...]
    source_notes: tuple[str, ...]
    external_sources: tuple[ExternalSource, ...]
    license_policy: str
    when_to_use: tuple[str, ...]
    failure_modes: tuple[str, ...]
    implementation_gaps: tuple[str, ...]
    next_tasks: tuple[str, ...]
    interview_line: str


EXTERNAL_SOURCES: dict[str, ExternalSource] = {
    "meta_learners_pnas": ExternalSource(
        name="Metalearners for estimating heterogeneous treatment effects using machine learning",
        url="https://doi.org/10.1073/pnas.1804597116",
        source_type="paper",
        license="paper reference",
        license_risk="No code copied; use the S/T/X learner idea and cite the paper.",
        adoption="reference formulas and evaluation framing only",
    ),
    "r_learner": ExternalSource(
        name="Quasi-Oracle Estimation of Heterogeneous Treatment Effects",
        url="https://arxiv.org/abs/1712.04912",
        source_type="paper",
        license="paper reference",
        license_risk="No code copied; local implementation is a clean sklearn-style residual learner.",
        adoption="reference orthogonal residual objective",
    ),
    "dml": ExternalSource(
        name="Double/debiased machine learning for treatment and structural parameters",
        url="https://academic.oup.com/ectj/article/21/1/C1/5056401",
        source_type="paper",
        license="paper reference",
        license_risk="No code copied; use Neyman-orthogonal/cross-fitting design ideas.",
        adoption="reference nuisance cross-fitting and orthogonal score design",
    ),
    "econml": ExternalSource(
        name="EconML",
        url="https://github.com/py-why/EconML",
        source_type="open-source framework",
        license="MIT plus BSD-3-Clause portions",
        license_risk="Permissive but preserve notices if code is copied; current repo uses guarded adapters, not copied source.",
        adoption="adapter benchmark target and model-card reference",
    ),
    "causalml": ExternalSource(
        name="CausalML",
        url="https://github.com/uber/causalml",
        source_type="open-source framework",
        license="Apache-2.0",
        license_risk="Permissive with notice/patent terms; current repo keeps it optional/guarded.",
        adoption="uplift tree, causal forest, multi-treatment and industrial API reference",
    ),
    "scikit_uplift": ExternalSource(
        name="scikit-uplift",
        url="https://github.com/maks-sh/scikit-uplift",
        source_type="open-source framework",
        license="MIT",
        license_risk="Permissive if notices are preserved; current repo wraps it as optional backend.",
        adoption="two-model/class-transformation baseline reference",
    ),
    "cfrnet": ExternalSource(
        name="clinicalml/cfrnet",
        url="https://github.com/clinicalml/cfrnet",
        source_type="official research code",
        license="MIT",
        license_risk="Permissive; local PyTorch version is not a TensorFlow code copy.",
        adoption="architecture/reference only; local implementation keeps PyTorch + registry contract",
    ),
    "shalit": ExternalSource(
        name="Estimating individual treatment effect: generalization bounds and algorithms",
        url="https://proceedings.mlr.press/v70/shalit17a.html",
        source_type="paper",
        license="paper reference",
        license_risk="No code copied; cite for TARNet/CFRNet objective and balance rationale.",
        adoption="reference TARNet/CFRNet theory",
    ),
    "dragonnet": ExternalSource(
        name="Adapting neural networks for the estimation of treatment effects",
        url="https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects",
        source_type="paper",
        license="paper reference",
        license_risk="No code copied; local DragonNet extends local TarNet implementation.",
        adoption="reference propensity head and targeted regularization",
    ),
    "efin_paper": ExternalSource(
        name="Explicit Feature Interaction-aware Uplift Network for Online Marketing",
        url="https://arxiv.org/abs/2306.00315",
        source_type="paper",
        license="paper reference",
        license_risk="No code copied from paper; cite paper for module design.",
        adoption="reference user-treatment interaction architecture",
    ),
    "efin_repo": ExternalSource(
        name="dgliu/KDD23_EFIN",
        url="https://github.com/dgliu/KDD23_EFIN",
        source_type="official research code",
        license="GPL-3.0",
        license_risk="Do not copy code into default repo unless the whole derivative license strategy is approved.",
        adoption="read for ideas only; local implementation should remain clean-room or separately isolated",
    ),
    "descn_paper": ExternalSource(
        name="DESCN: Deep Entire Space Cross Networks for Individual Treatment Effect Estimation",
        url="https://arxiv.org/abs/2207.09920",
        source_type="paper",
        license="paper reference",
        license_risk="No code copied; local ESX/DESCN is maintained under current repo contract.",
        adoption="reference entire-space + cross-network objective",
    ),
    "descn_repo": ExternalSource(
        name="kailiang-zhong/DESCN",
        url="https://github.com/kailiang-zhong/DESCN",
        source_type="research code",
        license="unknown/not verified in this run",
        license_risk="Treat as read-only research reference until a license file is verified.",
        adoption="do not copy; compare architecture and training losses conceptually",
    ),
    "routellm": ExternalSource(
        name="RouteLLM",
        url="https://github.com/lm-sys/RouteLLM",
        source_type="open-source framework",
        license="Apache-2.0",
        license_risk="Permissive; current repo references routing pattern, not source code.",
        adoption="LLM routing benchmark/router abstraction reference",
    ),
}


MODEL_DECONSTRUCTIONS: dict[str, ModelDeconstruction] = {
    "TLearnerGBM": ModelDeconstruction(
        model_id="TLearnerGBM",
        display_name="T-Learner GBM",
        family="meta learner",
        role="first-principles baseline for binary treatment uplift",
        current_status="ready via sklearn HistGradientBoosting backend",
        code_paths=(
            "deepuplift/core/registry.py:_build_sklearn_uplift('t')",
            "deepuplift/core/sklearn_models.py:SklearnUpliftModel._fit_t_learner",
            "deepuplift/core/sklearn_models.py:SklearnUpliftModel.predict",
            "deepuplift/core/trainer.py:train_uplift_model",
        ),
        formulas=(
            "mu0(x)=E[Y|X=x,T=0], mu1(x)=E[Y|X=x,T=1]",
            "tau_hat(x)=mu1_hat(x)-mu0_hat(x)",
            "score(x)=tau_hat(x); policy treats top-K or rows with tau_hat(x)*value-cost>0",
        ),
        call_chain=(
            "UpliftConfig.model_name='TLearnerGBM'",
            "registry.build_model -> SklearnUpliftModel(learner_type='t')",
            "trainer.train_uplift_model -> model.fit -> _fit_t_learner",
            "model.predict -> _predict_t -> evaluator.evaluate_uplift_predictions",
            "artifacts -> metrics.json, predictions.csv, train_history.csv, run_note.md, evidence_manifest.json",
        ),
        data_flow=(
            "features are encoded by TabularPreprocessor",
            "binary treatment is normalized to 0/1 by trainer._normalize_binary_column",
            "control rows train m0 and treated rows train m1",
            "prediction exports y0_pred, y1_pred and uplift_score",
        ),
        train_flow=(
            "split holdout with optional stratify by treatment/outcome",
            "fit outcome_estimator on X[T=0],Y[T=0]",
            "fit outcome_estimator on X[T=1],Y[T=1]",
            "record observed factual loss for train/valid history",
        ),
        predict_flow=(
            "predict m0(x) and m1(x) for every holdout row",
            "compute uplift_score=y1_pred-y0_pred",
            "send predictions to QINI/AUUC/Top-K/policy evaluator",
        ),
        eval_metrics=("QINI", "AUUC", "uplift@K", "policy value", "calibration MAE", "bootstrap/sensitivity when enabled"),
        smoke_evidence=(
            "scripts/smoke_test_agent.sh",
            "scripts/smoke_compare_manifest.py --models TLearnerGBM DRLearnerGBM",
            "reports/agent_regression_*.json",
            "reports/no_ui_compare_manifest_*.json",
        ),
        source_notes=(
            "T-Learner is simple, transparent and easy to debug but does not share information between arms.",
            "It is the default interview baseline for explaining uplift before robust or deep learners.",
        ),
        external_sources=(EXTERNAL_SOURCES["meta_learners_pnas"], EXTERNAL_SOURCES["scikit_uplift"]),
        license_policy="Safe to implement locally; external framework code is optional and not copied.",
        when_to_use=(
            "randomized or near-randomized binary treatment logs",
            "strong first baseline for coupon, push, ad exposure and LLM cheap-vs-strong routing",
            "small demo where explainability matters more than maximal robustness",
        ),
        failure_modes=(
            "small treatment arm overfits because m1 has limited data",
            "selection bias is not corrected",
            "poor overlap can make one arm extrapolate outside support",
        ),
        implementation_gaps=(
            "no arm-sharing representation",
            "no propensity or orthogonal correction",
            "no native confidence interval inside model; evaluator adds bootstrap when requested",
        ),
        next_tasks=(
            "add deconstruction screenshot/table in UI model card",
            "add per-arm feature importances when backend exposes them",
            "compare against DR/R on biased synthetic datasets",
        ),
        interview_line="T-Learner is my clean baseline: two factual outcome models, tau=y1-y0, then every claim goes through the shared evaluator and evidence manifest.",
    ),
    "DRLearnerGBM": ModelDeconstruction(
        model_id="DRLearnerGBM",
        display_name="Doubly Robust Learner GBM",
        family="DR / orthogonal learner",
        role="robust tabular baseline for observational uplift with nuisance diagnostics",
        current_status="ready via sklearn backend with optional cross-fitting",
        code_paths=(
            "deepuplift/core/registry.py:_build_sklearn_uplift('dr')",
            "deepuplift/core/sklearn_models.py:SklearnUpliftModel._fit_dr_learner",
            "deepuplift/core/sklearn_models.py:_cross_fit_dr_nuisance",
            "deepuplift/core/sklearn_models.py:_build_nuisance_diagnostics",
            "deepuplift/core/trainer.py:nuisance_diagnostics artifact write",
        ),
        formulas=(
            "e(x)=P(T=1|X=x), mu_t(x)=E[Y|X=x,T=t]",
            "pseudo_DR=(mu1-mu0)+T*(Y-mu1)/e-(1-T)*(Y-mu0)/(1-e)",
            "tau_hat=argmin_f sum_i (pseudo_DR_i-f(X_i))^2",
        ),
        call_chain=(
            "registry.build_model -> SklearnUpliftModel(learner_type='dr')",
            "_fit_dr_learner fits m0,m1,propensity and pseudo outcome",
            "if cross_fit=True or nuisance_folds>=2, folds create out-of-fold nuisance predictions",
            "fit effect_estimator on pseudo_DR",
            "predict uses y0_base and tau model to recover y1=y0+tau",
        ),
        data_flow=(
            "same normalized binary uplift schema as TLearner",
            "nuisance predictions produce m0, m1, raw propensity, clipped propensity",
            "pseudo outcomes and weights are summarized in nuisance_diagnostics.json",
        ),
        train_flow=(
            "fit control and treated outcome models",
            "fit propensity model",
            "construct DR pseudo effect with clipped propensity",
            "fit final effect model on pseudo effect",
            "write diagnostics: propensity ESS, weak-overlap rate, pseudo summary, fold rows",
        ),
        predict_flow=(
            "predict base y0/y1 via T-Learner style heads",
            "predict tau model",
            "return y0 and y0+tau under the common trainer contract",
        ),
        eval_metrics=("QINI", "AUUC", "Top-K observed uplift", "policy value", "nuisance ESS", "weak overlap", "sensitivity"),
        smoke_evidence=(
            "scripts/smoke_nuisance_diagnostics.py",
            "reports/nuisance_diagnostics_smoke_latest.json",
            "reports/ope_engine_smoke_latest.json",
            "scripts/smoke_compare_manifest.py",
        ),
        source_notes=(
            "DR learner is model-agnostic and useful when either outcome or propensity nuisance is reasonably specified.",
            "The same answer should connect to DML: nuisance models are fit separately and the final effect score is orthogonalized against first-stage errors.",
            "In this repo it is a practical bridge between CATE theory and benchmark artifacts.",
        ),
        external_sources=(
            EXTERNAL_SOURCES["dml"],
            EXTERNAL_SOURCES["r_learner"],
            EXTERNAL_SOURCES["econml"],
            EXTERNAL_SOURCES["causalml"],
        ),
        license_policy="Local code is clean-room sklearn implementation; EconML/CausalML remain guarded external adapters.",
        when_to_use=(
            "observational coupon/ads/CRM logs with selection bias risk",
            "offline benchmark before trying neural representation learners",
            "interview questions about why the project is not just a model zoo",
        ),
        failure_modes=(
            "propensity near 0/1 creates high-variance pseudo outcomes",
            "bad nuisance models can still mis-rank uplift",
            "cross-fitting raises runtime and small-sample instability",
        ),
        implementation_gaps=(
            "no analytic standard errors for tau model yet",
            "effect model still uses generic regression objective",
            "DML/DR distinction could be split into clearer registry families",
        ),
        next_tasks=(
            "add DR vs R vs T benchmark page with nuisance artifact links",
            "add clipping sensitivity table to run note",
            "write model-card appendix for nuisance model choices",
        ),
        interview_line="DRLearnerGBM is the credibility upgrade: it turns outcome and propensity models into an auditable pseudo-outcome, then stores ESS and overlap evidence.",
    ),
    "CFRNet": ModelDeconstruction(
        model_id="CFRNet",
        display_name="CFRNet",
        family="representation balancing neural CATE",
        role="deep learner for covariate-shift and representation-balance discussion",
        current_status="ready PyTorch model with differentiable MMD/energy/mean-L2 balance",
        code_paths=(
            "deepuplift/models/TarNet.py:TarNet.forward",
            "deepuplift/models/CFRNet.py:cfrnet_loss",
            "deepuplift/models/BaseModel.py:representation_balance_loss",
            "scripts/smoke_cfrnet_differentiable_balance.py",
            "scripts/smoke_cfrnet_training_balance.py",
        ),
        formulas=(
            "phi=f_shared(X), y0_hat=h0(phi), y1_hat=h1(phi)",
            "L_y=(1-T)*ell(y0_hat,Y)+T*ell(y1_hat,Y)",
            "L_CFR=L_y+alpha*IPM(phi(X_T=1),phi(X_T=0))",
            "MMD=E[k(phi_t,phi_t)]+E[k(phi_c,phi_c)]-2E[k(phi_t,phi_c)]",
        ),
        call_chain=(
            "registry._build_cfrnet -> CFRNet(TarNet subclass)",
            "BaseModel.fit -> model(X,T) returns t_pred,y_preds,phi_x,eps",
            "cfrnet_loss -> BaseLoss(IPM=True)",
            "representation_balance_loss computes differentiable torch MMD/energy/mean_l2",
            "trainer writes model.pt, train_history.csv and evaluation artifacts",
        ),
        data_flow=(
            "encoded numeric feature matrix enters shared TowerUnit",
            "shared representation phi_x feeds two potential-outcome heads",
            "phi_x also feeds IPM balance penalty grouped by treatment",
        ),
        train_flow=(
            "train factual outcome BCE/MSE only on observed arm",
            "add IPM penalty between treated and control representations",
            "return treatment_loss column as the balance term for history/evidence",
        ),
        predict_flow=(
            "forward returns y0/y1 for every row",
            "uplift_score=y1-y0",
            "evaluator handles QINI/AUUC/Top-K/policy value like every other model",
        ),
        eval_metrics=("QINI", "AUUC", "Top-K", "policy value", "balance term", "overlap trimming", "oracle recall when true_uplift exists"),
        smoke_evidence=(
            "reports/cfrnet_differentiable_balance_smoke_latest.json",
            "reports/cfrnet_training_balance_smoke_latest.json",
            "reports/deep_neural_ablation_latest.json",
        ),
        source_notes=(
            "The important local engineering fix is keeping IPM in torch; converting to numpy would remove gradient flow.",
            "CFRNet is a strong interview model because it connects causal assumptions, representation learning and code-level loss design.",
        ),
        external_sources=(EXTERNAL_SOURCES["shalit"], EXTERNAL_SOURCES["cfrnet"]),
        license_policy="MIT reference repo is safe to study; local PyTorch implementation avoids copying legacy TensorFlow code.",
        when_to_use=(
            "treatment/control feature distributions differ but overlap is not hopeless",
            "need a neural baseline beyond meta-learners",
            "want to discuss balancing assumptions in architecture review",
        ),
        failure_modes=(
            "over-balancing can erase true effect modifiers",
            "IPM can look good while policy value is poor",
            "small batches with one treatment arm produce zero balance signal",
        ),
        implementation_gaps=(
            "no Wasserstein Sinkhorn loss in the default ready path",
            "no representation diagnostic plot in UI yet",
            "no alpha sweep benchmark in default full regression",
        ),
        next_tasks=(
            "add representation balance curve to run artifacts",
            "add alpha sweep smoke with leaderboard",
            "surface phi balance diagnostics in Model Deconstruction UI",
        ),
        interview_line="CFRNet is TARNet plus a real gradient-bearing balance penalty; I can point to the exact loss and the smoke that proves the IPM term backpropagates.",
    ),
    "DragonNet": ModelDeconstruction(
        model_id="DragonNet",
        display_name="DragonNet",
        family="propensity-aware neural CATE",
        role="deep learner for joint outcome, propensity and targeted-regularization discussion",
        current_status="ready local PyTorch model with shared encoder, y0/y1 heads, propensity head and targeted regularization",
        code_paths=(
            "deepuplift/models/TarNet.py:TarNet.forward(model_type='dragonnet')",
            "deepuplift/models/DragonNet.py:DragonNet",
            "deepuplift/models/DragonNet.py:dragonnet_loss",
            "deepuplift/models/BaseModel.py:BaseLoss(loss_type='dragonnet')",
            "scripts/smoke_deep_model_training_evidence.py",
        ),
        formulas=(
            "phi=f_shared(X), y0_hat=h0(phi), y1_hat=h1(phi), e_hat=h_t(phi)",
            "L_y=(1-T)*ell(y0_hat,Y)+T*ell(y1_hat,Y)",
            "h=T/e_hat-(1-T)/(1-e_hat), y_tilde=y_hat+epsilon*h",
            "L_Dragon=L_y+alpha*BCE(e_hat,T)+beta*(Y-y_tilde)^2",
        ),
        call_chain=(
            "registry._build_dragonnet -> DragonNet(TarNet model_type='dragonnet')",
            "TarNet.forward returns t_pred,y_preds,phi_x,eps",
            "dragonnet_loss -> BaseLoss(loss_type='dragonnet', tarreg=True)",
            "BaseLoss adds propensity BCE and targeted regularization",
            "trainer writes train_history.csv, metrics.json and evidence_manifest.json",
        ),
        data_flow=(
            "encoded feature matrix enters shared TowerUnit",
            "shared representation feeds y0/y1 factual outcome heads",
            "the same representation feeds h_t propensity head",
            "epsilon layer perturbs factual prediction through inverse-propensity residual h",
        ),
        train_flow=(
            "train factual outcome loss on observed arm",
            "train treatment assignment head as an auxiliary propensity signal",
            "optionally add targeted regularization through epsilon and h",
            "record train_treatment_loss as propensity-head BCE in train history",
        ),
        predict_flow=(
            "forward returns y0/y1 potential outcome predictions for every holdout row",
            "uplift_score=y1-y0",
            "evaluator checks QINI/AUUC/Top-K/policy value and run artifacts",
        ),
        eval_metrics=("QINI", "AUUC", "Top-K", "policy value", "propensity auxiliary loss", "calibration MAE", "oracle recall when true_uplift exists"),
        smoke_evidence=(
            "reports/deep_model_training_evidence_latest.json",
            "reports/deep_model_training_leaderboard_latest.csv",
            "docs/UPLIFT_DEEP_MODEL_TRAINING_EVIDENCE.md",
        ),
        source_notes=(
            "DragonNet is useful in interviews because it makes treatment assignment part of the network instead of a separate diagnostic only.",
            "Targeted regularization can stabilize effect estimation, but it still does not fix missing support or unmeasured confounding.",
        ),
        external_sources=(EXTERNAL_SOURCES["dragonnet"],),
        license_policy="Paper reference only; local implementation extends the existing TarNet code path and does not copy external source.",
        when_to_use=(
            "binary treatment with meaningful non-random assignment signal",
            "want to show propensity-aware neural uplift beyond TARNet/CFRNet",
            "need a bridge between representation learning and causal robustness discussion",
        ),
        failure_modes=(
            "propensity head can overfit selection patterns",
            "targeted regularization becomes unstable when e_hat is near 0 or 1",
            "good propensity loss does not prove causal identification",
        ),
        implementation_gaps=(
            "no explicit propensity calibration chart in the default UI yet",
            "targeted regularization term is not logged separately from total loss",
            "no sensitivity sweep over alpha/beta in full regression yet",
        ),
        next_tasks=(
            "log targeted regularization as a separate train-history column",
            "add propensity calibration and overlap plot for DragonNet",
            "compare DragonNet against DR/R learners on biased synthetic datasets",
        ),
        interview_line="DragonNet is the propensity-aware neural upgrade: one shared encoder predicts y0/y1 and e(x), then targeted regularization ties effect estimation to treatment assignment evidence.",
    ),
    "EFIN": ModelDeconstruction(
        model_id="EFIN",
        display_name="EFIN",
        family="explicit feature interaction uplift network",
        role="industrial deep uplift model for treatment-aware interactions",
        current_status="ready local PyTorch model; external official code is GPL-3.0 and treated as reference only",
        code_paths=(
            "deepuplift/models/EFIN.py:EFIN.forward",
            "deepuplift/models/EFIN.py:interaction_attn",
            "deepuplift/models/BaseModel.py:BaseLoss(loss_type='efin')",
            "deepuplift/core/registry.py:_build_efin",
            "docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md",
        ),
        formulas=(
            "x_rep_j=x_j * embedding_j, t_rep=W_t*1",
            "a_j=softmax(v^T relu(sigmoid(W_t t)+sigmoid(W_x x_j)))",
            "u(x,t)=g_u(sum_j a_j*x_rep_j), y1=base(x)+u(x,t), y0=base(x)",
            "L_EFIN=L_factual+L_treatment_aux; local implementation uses reversed treatment aux label as intervention constraint proxy",
        ),
        call_chain=(
            "registry._build_efin -> EFIN(input_dim,hc_dim,hu_dim,is_self,use_interaction_attention,uplift_tau_scale)",
            "forward builds feature embeddings and optional self-attention",
            "control_net estimates natural response base",
            "interaction_attn learns treatment-aware feature weights",
            "uplift_net outputs u_tau and treatment auxiliary t_pred",
        ),
        data_flow=(
            "tabular features become feature-wise embeddings",
            "control path models natural response without treatment-aware uplift interaction",
            "uplift path uses attention over user/context features conditioned on treatment representation",
        ),
        train_flow=(
            "BaseModel.fit trains factual outcome head and auxiliary treatment objective",
            "classification predicts y0=sigmoid(c_logit), y1=sigmoid(detach(c_logit)+u_tau)",
            "regression predicts y0=c_logit, y1=detach(c_logit)+u_tau",
        ),
        predict_flow=(
            "predict returns treatment auxiliary probability and y0/y1 heads",
            "uplift_score comes from y1-y0 under shared evaluator",
            "attention/path telemetry is exported through guarded smoke/collector artifacts rather than the default predict API",
        ),
        eval_metrics=("QINI", "AUUC", "Top-K", "policy value", "calibration", "attention stability future metric"),
        smoke_evidence=(
            "scripts/smoke_deep_neural_ablation.py covers ready neural baseline family",
            "scripts/smoke_deep_loss_architecture_catalog.py",
            "reports/deep_loss_architecture_catalog_latest.json",
            "reports/deep_model_tuned_benchmark_latest.json covers EFIN_no_attention and EFIN_path_regularized ablations",
            "reports/deep_model_telemetry_ablation_gate_latest.json",
        ),
        source_notes=(
            "EFIN is valuable because marketing treatment can include coupon/creative/offer features, not only a binary flag.",
            "Official repository license is GPL-3.0, so DeepUplift should keep clean-room code and cite rather than copy.",
            "Local ablation variants can disable learned interaction attention or scale the uplift path without changing the default model API.",
        ),
        external_sources=(EXTERNAL_SOURCES["efin_paper"], EXTERNAL_SOURCES["efin_repo"]),
        license_policy="GPL-3.0 upstream means no blind source copy; keep local implementation clean-room or isolate in optional GPL-compatible plugin.",
        when_to_use=(
            "coupon/creative/offer treatment has rich treatment-side feature interactions",
            "online marketing scenarios where user-treatment matching matters",
            "interview discussion about industrial deep uplift beyond TARNet/CFRNet",
        ),
        failure_modes=(
            "can overfit campaign IDs or sparse treatment features",
            "auxiliary treatment loss can learn selection bias if logging policy is biased",
            "current local code does not export attention evidence",
        ),
        implementation_gaps=(
            "intervention constraint is simplified relative to paper",
            "attention is observable but still not a causal explanation until EFIN_no_attention beats the ablation gate",
            "treatment-feature schema support is still simplified to binary treatment in the default trainer",
        ),
        next_tasks=(
            "run tuned-v1 multi-seed EFIN_no_attention and EFIN_path_regularized ablations before upgrading attention/path claims",
            "add treatment-feature schema support instead of using only binary T",
            "compare against DRLearnerGBM on synthetic coupon/creative datasets",
        ),
        interview_line="EFIN is the model I use to show industrial depth: uplift is user-treatment interaction, but the GPL upstream means I cite ideas and keep local code clean-room.",
    ),
    "DESCN": ModelDeconstruction(
        model_id="DESCN",
        display_name="DESCN / ESX",
        family="entire-space cross network",
        role="deep multi-task model for propensity, response and hidden treatment-effect learning",
        current_status="ready local ESX classification model behind DESCN registry alias",
        code_paths=(
            "deepuplift/models/DESCN.py:ESX.forward",
            "deepuplift/models/DESCN.py:esx_loss",
            "deepuplift/core/registry.py:_build_descn",
            "scripts/run_deep_model_tuned_benchmark.py:DESCN_no_constraint",
            "docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md",
        ),
        formulas=(
            "p=propensity(phi), mu0=sigmoid(mu0_head(phi)), mu1=sigmoid(mu1_head(phi))",
            "estr=p*mu1, escr=(1-p)*mu0",
            "cross_tr=sigmoid(mu0_logit+tau_logit), cross_cr=sigmoid(mu1_logit-tau_logit)",
            "L_DESCN=L_propensity+L_estr+L_escr+L_tr+L_cr+L_cross_tr+L_cross_cr+lambda*L_balance",
        ),
        call_chain=(
            "registry._build_descn -> ESX classification model with loss weights for cross-head constraint ablations",
            "forward -> share_network -> propensity/mu1/mu0/tau subnetworks",
            "esx_loss combines propensity, entire-space response and cross-head losses",
            "trainer normalizes DESCN outputs into y0/y1 under common evaluator",
        ),
        data_flow=(
            "shared representation feeds four heads",
            "propensity head estimates exposure probability",
            "entire-space heads multiply response by treatment/control exposure probability",
            "tau head connects counterfactual response via cross constraints",
        ),
        train_flow=(
            "classification only",
            "BCE for propensity and observed/cross response heads",
            "DESCN_no_constraint sets cross-head loss weights to zero to isolate the constraint contribution",
            "optional Wasserstein/MMD imbalance term exists in loss signature",
        ),
        predict_flow=(
            "forward returns p_prpsy and y_preds=[p_h0,p_h1]",
            "evaluator consumes y1-y0 after trainer normalization",
            "additional ESX internal heads are not exported as separate artifact yet",
        ),
        eval_metrics=("QINI", "AUUC", "Top-K", "policy value", "propensity diagnostics future", "head consistency future"),
        smoke_evidence=(
            "scripts/smoke_deep_loss_architecture_catalog.py",
            "reports/deep_loss_architecture_catalog_latest.json",
            "reports/deep_model_tuned_benchmark_latest.json covers DESCN_no_constraint",
            "reports/deep_model_telemetry_ablation_gate_latest.json",
            "docs/UPLIFT_DEEP_MODEL_INTERVIEW_DEEP_DIVE.md",
        ),
        source_notes=(
            "DESCN matters when treatment exposure is biased and response space is imbalanced.",
            "Local implementation maps the paper idea into the same DeepUplift trainer/evaluator contract.",
        ),
        external_sources=(EXTERNAL_SOURCES["descn_paper"], EXTERNAL_SOURCES["descn_repo"]),
        license_policy="Upstream license was not verified in this run; treat repo as read-only and do not copy code.",
        when_to_use=(
            "biased exposure/recommendation logs",
            "need to discuss entire-space propensity and response heads",
            "classification uplift ranking where sample imbalance is central",
        ),
        failure_modes=(
            "head semantics can be confusing without adapter normalization",
            "loss has many terms and can be unstable on tiny demo data",
            "propensity head still cannot solve missing overlap",
        ),
        implementation_gaps=(
            "head-level metrics are not persisted",
            "imbalance term is off by default",
            "cross-head constraint benefit still needs tuned-v1 multi-seed ablation before a stronger performance claim",
        ),
        next_tasks=(
            "add ESX head consistency artifact",
            "run tuned-v1 DESCN_no_constraint and compare PEHE/QINI/policy movement against DESCN_default",
            "audit upstream license before any deeper source borrowing",
        ),
        interview_line="DESCN is my example for entire-space causal modeling: propensity, response and pseudo-effect heads are trained together, then normalized into the common uplift evaluator.",
    ),
    "MultiDRLearnerGBM": ModelDeconstruction(
        model_id="MultiDRLearnerGBM",
        display_name="Multi-Treatment DR Learner / LLM Routing Policy",
        family="multi-action policy learner",
        role="extend binary uplift into action recommendation and LLM routing",
        current_status="ready multi-treatment benchmark plus LLM routing synthetic/policy evidence",
        code_paths=(
            "deepuplift/core/multitreatment.py:train_multi_treatment_model",
            "deepuplift/core/multitreatment.py:_ipw_policy_value",
            "deepuplift/core/doseresponse.py:continuous_treatment_policy_optimizer",
            "deepuplift/models/deep_losses.py:llm_multi_action_routing_loss",
            "scripts/smoke_multi_treatment_benchmark.py",
            "scripts/smoke_llm_routing_policy.py",
        ),
        formulas=(
            "for action a: mu_a(x)=E[Y|X=x,A=a]",
            "tau_a(x)=mu_a(x)-mu_control(x)",
            "DR pseudo for action a: mu_a-mu_c+1{A=a}(Y-mu_a)/p_a-1{A=c}(Y-mu_c)/p_c",
            "pi(x)=argmax_a tau_a(x)*value-cost_a-latency_a-risk_a subject to budget/guardrails",
        ),
        call_chain=(
            "train_multi_treatment_model detects more than two treatment values",
            "fit one outcome model per action and a multinomial propensity model",
            "optional dr_gbm fits per-action effect models on DR pseudo effects",
            "score rows with y_pred_action columns and recommended_treatment",
            "compute IPW policy value, action propensity diagnostics and bootstrap CI",
        ),
        data_flow=(
            "treatment values become action labels",
            "propensity matrix aligns action probabilities to action labels",
            "predictions export per-action outcomes, uplift_vs_control and recommended action",
        ),
        train_flow=(
            "T-learner style: one model per action",
            "DR style: construct action-vs-control pseudo effects",
            "policy optimizer ranks actions by incremental net value, not only raw uplift",
        ),
        predict_flow=(
            "predict all potential outcomes y_a(x)",
            "compute recommended_treatment and recommended_uplift_vs_control",
            "evaluate action mix, IPW policy value and propensity support",
        ),
        eval_metrics=("incremental_policy_value_ipw", "action_count", "propensity_by_action", "bootstrap CI", "Top action rows", "LLM routing net value"),
        smoke_evidence=(
            "reports/multi_treatment_benchmark_latest.json",
            "reports/multi_treatment_benchmark_latest.csv",
            "reports/llm_routing_policy_smoke_latest.json",
            "examples/datasets/synthetic_llm_multi_action_routing_9k.csv",
        ),
        source_notes=(
            "Multi-treatment is the natural upgrade from who-to-treat into which-action-to-take.",
            "LLM routing is represented as cheap model control vs strong/RAG/tool/human treatment arms.",
        ),
        external_sources=(EXTERNAL_SOURCES["causalml"], EXTERNAL_SOURCES["routellm"]),
        license_policy="CausalML/RouteLLM are permissive references; current implementation uses local clean-room policy logic and guarded adapters.",
        when_to_use=(
            "coupon amount/type, channel, offer and journey action selection",
            "LLM routing between cheap/strong/RAG/tool/human paths",
            "budget-constrained industrial policy discussion",
        ),
        failure_modes=(
            "some actions may have weak overlap or no exploration",
            "old policy bias can be copied by the recommender",
            "ROI can be wrong if action costs/latency/risk are not logged",
        ),
        implementation_gaps=(
            "no neural multi-action attention adapter yet",
            "offline IPW value is not online proof",
            "LLM judge quality and drift need a stronger OPE monitor",
        ),
        next_tasks=(
            "add action-level DR/R nuisance diagnostics",
            "add multi-action model card and UI deconstruction panel",
            "add LLM routing OPE + budget pacing benchmark row",
        ),
        interview_line="Multi-treatment turns DeepUplift from ranking users into choosing actions; LLM routing is the same causal policy problem with cost, latency and quality guardrails.",
    ),
}


MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS: tuple[dict[str, Any], ...] = (
    {"prompt": "从源码拆解 TLearnerGBM 的 fit 和 predict 调用链", "model": "TLearnerGBM", "expect": ("_fit_t_learner", "tau_hat", "evidence")},
    {"prompt": "T-Learner 为什么是 uplift 的 baseline，不是普通转化模型？", "model": "TLearnerGBM", "expect": ("mu0", "mu1", "uplift_score")},
    {"prompt": "TLearnerGBM 的失败模式和上线风险怎么讲？", "model": "TLearnerGBM", "expect": ("selection bias", "overlap", "policy value")},
    {"prompt": "DRLearnerGBM 的 pseudo outcome 公式和源码在哪里？", "model": "DRLearnerGBM", "expect": ("pseudo_DR", "_fit_dr_learner", "nuisance")},
    {"prompt": "DRLearnerGBM 为什么需要 cross-fitting 和 nuisance artifacts？", "model": "DRLearnerGBM", "expect": ("cross_fit", "propensity", "ESS")},
    {"prompt": "R/DML 和 DR learner 的 orthogonal 思想怎么讲给面试官？", "model": "DRLearnerGBM", "expect": ("orthogonal", "DML", "nuisance")},
    {"prompt": "CFRNet 的 loss 怎么从 TARNet 推出来？", "model": "CFRNet", "expect": ("IPM", "MMD", "representation_balance_loss")},
    {"prompt": "CFRNet 源码里 forward 返回的 phi_x 为什么重要？", "model": "CFRNet", "expect": ("phi", "forward", "balance")},
    {"prompt": "CFRNet 的 differentiable balance smoke 证明了什么？", "model": "CFRNet", "expect": ("torch", "gradient", "smoke")},
    {"prompt": "CFRNet 的 objective diagnostics 怎么把 loss term 映射到 PEHE/QINI 和 safe claim？", "model": "CFRNet", "expect": ("objective diagnostics", "loss", "safe claim")},
    {"prompt": "DragonNet 的 propensity head 和 targeted regularization 源码怎么讲？", "model": "DragonNet", "expect": ("propensity", "targeted", "epsilon")},
    {"prompt": "DragonNet 相比 TARNet/CFRNet 的核心改动是什么？", "model": "DragonNet", "expect": ("h_t", "propensity", "dragonnet_loss")},
    {"prompt": "DragonNet 的训练证据和 failure mode 怎么讲给面试官？", "model": "DragonNet", "expect": ("deep_model_training_evidence", "overlap", "propensity")},
    {"prompt": "EFIN 的 treatment-aware interaction attention 源码怎么讲？", "model": "EFIN", "expect": ("interaction_attn", "attention", "GPL")},
    {"prompt": "EFIN 和普通双塔/TARNet 的区别是什么？", "model": "EFIN", "expect": ("treatment", "interaction", "user")},
    {"prompt": "EFIN 官方代码 license 风险和 DeepUplift 借鉴策略是什么？", "model": "EFIN", "expect": ("GPL-3.0", "clean-room", "reference")},
    {"prompt": "DESCN/ESX 的 propensity、mu0、mu1、tau head 怎么协同？", "model": "DESCN", "expect": ("propensity", "tau", "esx_loss")},
    {"prompt": "DESCN 为什么叫 entire-space cross network？", "model": "DESCN", "expect": ("entire-space", "cross", "propensity")},
    {"prompt": "DESCN 当前实现有哪些 gap？", "model": "DESCN", "expect": ("smoke", "head", "license")},
    {"prompt": "MultiDRLearnerGBM 如何把二元 uplift 扩展成多动作推荐？", "model": "MultiDRLearnerGBM", "expect": ("action", "recommended_treatment", "IPW")},
    {"prompt": "LLM routing 为什么可以放进 multi-treatment uplift？", "model": "MultiDRLearnerGBM", "expect": ("cheap", "strong", "cost")},
    {"prompt": "multi-treatment 的 OPE 和 propensity diagnostics 怎么验证？", "model": "MultiDRLearnerGBM", "expect": ("propensity_by_action", "incremental_policy_value_ipw", "bootstrap")},
    {"prompt": "Model Deconstruction Agent 3.0 的验收证据有哪些？", "model": "CFRNet", "expect": ("docs/model_deconstruction", "reports/model_deconstruction", "golden")},
    {"prompt": "给我一个 5 分钟讲解 CFRNet 源码的面试话术", "model": "CFRNet", "expect": ("5分钟", "code", "metric")},
    {"prompt": "怎么证明模型拆解不是只写文档，而是能训练评估？", "model": "DRLearnerGBM", "expect": ("smoke", "metrics.json", "QINI")},
    {"prompt": "外部开源代码能不能直接复制进 DeepUplift？", "model": "EFIN", "expect": ("license", "GPL", "guarded")},
    {"prompt": "CFRNet 在 deep tuned benchmark 里为什么输给或赢过 T/DR baseline？", "model": "CFRNet", "expect": ("deep_model_tuned_benchmark", "baseline", "failure attribution")},
    {"prompt": "DragonNet 的 tuned preset 和 no_tarreg ablation 怎么解释？", "model": "DragonNet", "expect": ("battle card", "ablation", "targeted")},
    {"prompt": "EFIN 在 tuned benchmark 中如果没有赢，面试怎么安全表达？", "model": "EFIN", "expect": ("baseline", "ranking", "license")},
    {"prompt": "DESCN 在回归任务被 skip 是失败还是 guarded task boundary？", "model": "DESCN", "expect": ("guarded", "classification", "DESCN")},
    {"prompt": "为什么 T-Learner / DRLearner 在小型 known-CATE benchmark 上经常赢深度模型？", "model": "TLearnerGBM", "expect": ("T/DR", "small tabular", "over-claim")},
    {"prompt": "如何用 algorithm claim ledger 证明一个模型 claim 没有过度包装？", "model": "DRLearnerGBM", "expect": ("algorithm claim ledger", "safe claim", "evidence")},
    {"prompt": "怎么用 paper reproduction gap ledger 说明论文复现到什么程度？", "model": "CFRNet", "expect": ("paper reproduction gap ledger", "license", "benchmark")},
    {"prompt": "怎么用 model evidence promotion matrix 判断一个模型能讲到什么程度？", "model": "DESCN", "expect": ("promotion matrix", "safe claim", "blocked claim")},
    {"prompt": "怎么用 model upgrade recipe card 说明下一步实验怎么做？", "model": "CFRNet", "expect": ("recipe card", "pass gate", "expected artifacts")},
    {"prompt": "怎么用 deep model telemetry gap matrix 解释 CFRNet/DragonNet/EFIN/DESCN 的失败归因？", "model": "DragonNet", "expect": ("telemetry gap", "loss", "pass gate")},
    {"prompt": "deep model telemetry smoke 能证明哪些基础 telemetry 已经可读？", "model": "CFRNet", "expect": ("telemetry smoke", "loss history", "advanced hook")},
    {"prompt": "forward hook contract 怎么指导深度模型源码改造？", "model": "CFRNet", "expect": ("forward hook", "tensor keys", "artifact", "pass gate")},
    {"prompt": "forward hook smoke 实际导出了哪些模型内部 tensor？", "model": "EFIN", "expect": ("forward hook smoke", "sidecar", "attention", "tensor")},
    {"prompt": "hook training bridge 怎么把 forward sidecar 升级成 per-epoch 训练证据？", "model": "CFRNet", "expect": ("hook training bridge", "per-epoch", "pass gate", "trainer")},
    {"prompt": "trainer collector smoke 怎么证明深度模型训练期 telemetry 能跑？", "model": "DragonNet", "expect": ("trainer collector", "per-epoch", "DragonNet", "telemetry")},
    {"prompt": "telemetry-benchmark linkage 怎么把内部指标和 benchmark 胜负连起来？", "model": "CFRNet", "expect": ("telemetry", "benchmark", "next ablation", "safe claim")},
    {"prompt": "telemetry ablation gate 怎么把 linkage 升级成可执行消融实验？", "model": "CFRNet", "expect": ("ablation gate", "variant", "pass gate", "blocked claim")},
    {"prompt": "EFIN_no_attention 和 EFIN_path_regularized 这两个新消融变体怎么讲？", "model": "EFIN", "expect": ("EFIN_no_attention", "EFIN_path_regularized", "ablation")},
    {"prompt": "DESCN_no_constraint 这个新消融变体怎么讲？", "model": "DESCN", "expect": ("DESCN_no_constraint", "cross", "ablation")},
    {"prompt": "deep model ablation interpretation 怎么把消融变体结果讲清楚？", "model": "CFRNet", "expect": ("ablation interpretation", "default", "PEHE", "safe claim")},
    {"prompt": "deep model ablation promotion matrix 怎么判断消融结果能不能升级成 claim？", "model": "CFRNet", "expect": ("ablation promotion", "claim tier", "promotion decision", "blocked claim")},
    {"prompt": "消融 promotion matrix 之后下一步实验计划怎么讲？", "model": "CFRNet", "expect": ("next experiment", "recommended command", "pass gate", "fail action")},
    {"prompt": "next experiment command contract 怎么判断哪些命令当前真的可执行？", "model": "CFRNet", "expect": ("command contract", "runnable fallback", "unsupported", "runner")},
    {"prompt": "fallback command smoke 怎么证明命令真的执行且不覆盖 latest？", "model": "CFRNet", "expect": ("command contract smoke", "no-update-latest", "manifest", "fallback")},
    {"prompt": "command smoke coverage 怎么区分已经 smoke 和 contract-ready backlog？", "model": "CFRNet", "expect": ("smoke coverage", "contract-ready", "coverage gate", "next_action")},
    {"prompt": "command smoke coverage diff 怎么说明这轮从 1/9 提升到 9/9？", "model": "CFRNet", "expect": ("coverage diff", "1/9", "9/9", "review_required -> ok")},
)


def _norm(text: str) -> str:
    return text.lower().replace("-", "").replace("_", "").replace(" ", "")


def get_model_deconstruction(model_id: str) -> ModelDeconstruction:
    if model_id in MODEL_DECONSTRUCTIONS:
        return MODEL_DECONSTRUCTIONS[model_id]
    lookup = {_norm(key): key for key in MODEL_DECONSTRUCTIONS}
    key = lookup.get(_norm(model_id))
    if key:
        return MODEL_DECONSTRUCTIONS[key]
    raise KeyError(f"Unknown model deconstruction: {model_id}")


def detect_deconstruction_model(prompt: str) -> str | None:
    text = _norm(prompt)
    for model_id in sorted(MODEL_DECONSTRUCTIONS, key=len, reverse=True):
        if _norm(model_id) in text:
            return model_id
    if any(key in text for key in ("license", "opensource", "开源代码", "直接复制", "gpl")):
        return "EFIN"
    if any(key in text for key in ("algorithmclaimledger", "claimledger", "safeclaim", "unsafeclaim", "过度包装", "证据层级")):
        return "DRLearnerGBM"
    if any(key in text for key in ("paperreproductiongapledger", "reproductiongap", "论文复现", "复现程度")):
        return "CFRNet"
    if any(key in text for key in ("modelevidencepromotionmatrix", "promotionmatrix", "promotiontier", "模型推广", "推广矩阵", "能讲到什么程度", "blockedclaim")):
        return "DESCN"
    if any(key in text for key in ("modelupgraderecipecard", "upgraderecipe", "recipecard", "升级实验", "下一步实验", "passgate", "failaction")):
        return "CFRNet"
    if any(key in text for key in ("deepmodeltelemetrygapmatrix", "telemetrygap", "losstelemetry", "可观测性", "propensitycalibration", "attentionweights", "headconsistency", "失败归因")):
        return "DragonNet"
    if any(key in text for key in ("deepmodeltelemetrysmoke", "telemetrysmoke", "basictelemetry", "losshistory", "predictionspread", "advancedhook")):
        return "CFRNet"
    if any(key in text for key in ("hooktrainingbridge", "trainertelemetry", "perepoch", "epochcolumns", "optionalcollector", "训练桥接", "训练期可观测")):
        return "CFRNet"
    if any(key in text for key in ("trainercollector", "collectorsmoke", "perepochtelemetry", "advancedtelemetry", "训练期采集", "epochtelemetry")):
        return "DragonNet"
    if any(key in text for key in ("telemetrybenchmarklinkage", "benchmarklinkage", "遥测和benchmark", "nextablation", "failurehypothesis", "为什么赢", "为什么输")):
        return "CFRNet"
    if any(key in text for key in ("telemetryablationgate", "ablationgate", "variantcoverage", "missingvariants", "消融门禁", "实验门禁")):
        return "CFRNet"
    if any(key in text for key in ("ablationpromotion", "ablationpromotionmatrix", "claimtier", "promotiondecision", "variantpromotion", "claim升级", "变体能不能讲")):
        return "CFRNet"
    if any(key in text for key in ("ablationcommandcontract", "commandcontract", "runnablefallback", "unsupportedflags", "runnercli", "命令契约", "可执行命令")):
        return "CFRNet"
    if any(key in text for key in ("commandcontractsmoke", "fallbacksmoke", "noupdatelatest", "真的执行", "命令smoke")):
        return "CFRNet"
    if any(key in text for key in ("commandsmokecoverage", "smokecoverage", "coveragegate", "contractready", "已经smoke", "smoke覆盖率")):
        return "CFRNet"
    if any(key in text for key in ("ablationnextexperiment", "nextexperimentplan", "recommendedcommand", "metricstowatch", "下一步实验", "面试追问")):
        return "CFRNet"
    if any(key in text for key in ("ablationinterpretation", "消融解读", "defaultvsvariant", "metricmovement", "variantscorecard")):
        return "CFRNet"
    if any(key in text for key in ("forwardhooksmoke", "sidecar", "tensorexport", "实际导出")):
        return "EFIN"
    if any(key in text for key in ("forwardhook", "hookcontract", "tensorkeys", "artifactfiles", "headexport", "源码改造")):
        return "CFRNet"
    if any(key in text for key in ("modeldeconstruction", "模型拆解", "源码拆解")):
        return "CFRNet"
    aliases = {
        "TLearnerGBM": ("tlearner", "two model", "twomodel", "meta learner", "metalearner"),
        "MultiDRLearnerGBM": ("multitreatment", "multi treatment", "multidr", "llm routing", "routing", "多动作", "多 treatment"),
        "DRLearnerGBM": ("drlearner", "doubly robust", "doublyrobust", "rlearner", "dml", "orthogonal", "nuisance"),
        "CFRNet": ("cfrnet", "cfr", "representation balance", "mmd", "ipm"),
        "DragonNet": ("dragonnet", "targeted regularization", "targeted", "propensity head", "epsilon"),
        "EFIN": ("efin", "feature interaction", "interaction attention", "treatmentaware"),
        "DESCN": ("descn", "esx", "entire space", "entirespace", "cross network"),
    }
    for model_id, keys in aliases.items():
        if any(_norm(key) in text for key in keys):
            return model_id
    if any(key in prompt for key in ("源码", "拆解", "loss", "forward", "fit", "predict", "从0", "从 0")):
        return "CFRNet"
    return None


def is_model_deconstruction_prompt(prompt: str) -> bool:
    return detect_deconstruction_model(prompt) is not None


def _bullet(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def source_rows(model: ModelDeconstruction) -> str:
    rows = ["| Source | Type | License | Risk | Adoption |", "| --- | --- | --- | --- | --- |"]
    for source in model.external_sources:
        rows.append(
            f"| [{source.name}]({source.url}) | {source.source_type} | {source.license} | "
            f"{source.license_risk} | {source.adoption} |"
        )
    return "\n".join(rows)


def _dict_inline(values: Mapping[str, Any] | None) -> str:
    if not values:
        return "NA"
    return ", ".join(f"{key}={value}" for key, value in sorted(values.items()))


def _benchmark_wins(model_id: str, interpretation: Mapping[str, Any] | None) -> str:
    if not interpretation:
        return "- No paper-level benchmark interpretation artifact is available yet."
    wins = []
    metric_labels = {
        "pehe_mean": "PEHE",
        "ate_error_mean": "ATE error",
        "qini_mean": "QINI",
        "auuc_mean": "AUUC",
        "policy_top10_oracle_value_mean": "Policy Top10",
        "oracle_top10_recall_mean": "Oracle Top10 recall",
    }
    for row in interpretation.get("dataset_winners") or []:
        dataset_id = row.get("dataset_id")
        for metric, label in metric_labels.items():
            if row.get(f"{metric}_winner") == model_id:
                wins.append(f"{dataset_id}: wins {label} with value {row.get(f'{metric}_value')}")
    return _bullet(wins) if wins else "- No latest benchmark-v3 metric win; use this model for its architecture role, assumptions, or guarded task boundary."


def latest_deep_model_tuned_benchmark(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_tuned_benchmark_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_deep_model_ablation_interpretation(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_ablation_interpretation_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_deep_model_ablation_promotion_matrix(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_ablation_promotion_matrix_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_deep_model_ablation_next_experiment_plan(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_ablation_next_experiment_plan_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_deep_model_ablation_command_contract(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_ablation_command_contract_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_deep_model_ablation_command_contract_smoke(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_ablation_command_contract_smoke_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_deep_model_ablation_recommended_command_smoke(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_ablation_recommended_command_smoke_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_deep_model_ablation_command_smoke_coverage(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_ablation_command_smoke_coverage_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def latest_deep_model_ablation_command_smoke_coverage_diff(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "deep_model_ablation_command_smoke_coverage_diff_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _tuned_model_rows(model_id: str, tuned: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not tuned:
        return []
    rows = []
    for card in tuned.get("battle_cards") or []:
        if card.get("model") == model_id or str(card.get("variant_id") or "").startswith(model_id):
            rows.append(dict(card))
    return sorted(rows, key=lambda row: (str(row.get("dataset_id")), str(row.get("variant_id"))))


def _tuned_baseline_rows(model_id: str, tuned: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not tuned:
        return []
    rows = []
    for row in tuned.get("baseline_comparison") or []:
        if row.get("model") == model_id or str(row.get("variant_id") or "").startswith(model_id):
            rows.append(dict(row))
    return sorted(rows, key=lambda row: (str(row.get("dataset_id")), str(row.get("variant_id"))))


def _tuned_model_scorecard(model_id: str, tuned: Mapping[str, Any] | None) -> dict[str, Any]:
    if not tuned:
        return {}
    for row in tuned.get("model_scorecards") or []:
        if row.get("model") == model_id:
            return dict(row)
    return {}


def _variant_metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6g}"


def tuned_benchmark_section(model: ModelDeconstruction, tuned: Mapping[str, Any] | None = None) -> str:
    if not tuned:
        return """## Deep Tuned Benchmark And Failure Attribution

Run `scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1` to attach tuned/ablation battle cards against T/DR baselines.
"""
    model_rows = _tuned_model_rows(model.model_id, tuned)
    baseline_rows = _tuned_baseline_rows(model.model_id, tuned)
    scorecard = _tuned_model_scorecard(model.model_id, tuned)
    if not model_rows and not baseline_rows:
        return f"""## Deep Tuned Benchmark And Failure Attribution

`{model.model_id}` is not part of the latest binary deep tuned benchmark. Use its dedicated smoke/benchmark artifacts instead, and keep the T/DR-vs-deep failure attribution scoped to CFRNet, DragonNet, EFIN and DESCN.

Latest tuned suite: `{tuned.get('preset')}` / `{tuned.get('benchmark_tier')}`, runs={len(tuned.get('runs') or [])}, battle_cards={len(tuned.get('battle_cards') or [])}.
"""

    lines = [
        "## Deep Tuned Benchmark And Failure Attribution",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Tuned suite | `{tuned.get('preset')}` / `{tuned.get('benchmark_tier')}` |",
        f"| Runs | `{len(tuned.get('runs') or [])}` |",
        f"| OK runs | `{sum(1 for row in tuned.get('runs') or [] if row.get('status') == 'ok')}` |",
        f"| Battle cards | `{len(tuned.get('battle_cards') or [])}` |",
    ]
    if scorecard:
        lines.extend(
            [
                "",
                "### Model Scorecard",
                "",
                "| Field | Value |",
                "| --- | --- |",
                f"| Verdict | `{scorecard.get('verdict')}` |",
                f"| OK runs | `{scorecard.get('ok_runs')}` |",
                f"| Best PEHE variant | `{scorecard.get('best_pehe_variant')}` / `{_variant_metric(scorecard.get('best_pehe'))}` |",
                f"| Best QINI variant | `{scorecard.get('best_qini_variant')}` / `{_variant_metric(scorecard.get('best_qini'))}` |",
                f"| Avg delta PEHE vs baseline | `{_variant_metric(scorecard.get('avg_delta_pehe_vs_baseline'))}` |",
                f"| Avg delta QINI vs baseline | `{_variant_metric(scorecard.get('avg_delta_qini_vs_baseline'))}` |",
                f"| Verdict counts | `{_dict_inline(scorecard.get('verdict_counts'))}` |",
                "",
                "Scorecard reading:",
                "",
                f"- Strength: {scorecard.get('strength')}",
                f"- Watchout: {scorecard.get('watchout')}",
                f"- Interview line: {scorecard.get('interview_line')}",
            ]
        )
        if scorecard.get("verdict") in {"needs_more_tuning", "guarded_or_task_limited"}:
            lines.append("- Promotion stance: keep it as architecture/failure-attribution evidence until tuned rows beat T/DR on the deployment metric.")
        elif scorecard.get("verdict") == "tuned_candidate":
            lines.append("- Promotion stance: candidate for paper-level discussion, but still require holdout/OPE before online claims.")
    lines.extend(
        [
            "",
            "### Model Battle Cards",
            "",
            "| Dataset | Variant | Verdict | PEHE | QINI | Policy Top10 | Train Loss Delta |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in model_rows:
        lines.append(
            f"| {row.get('dataset_id')} | {row.get('variant_id')} | {row.get('verdict')} | "
            f"{_variant_metric(row.get('pehe_mean'))} | {_variant_metric(row.get('qini_mean'))} | "
            f"{_variant_metric(row.get('policy_top10_oracle_value_mean'))} | {_variant_metric(row.get('train_loss_delta_mean'))} |"
        )
    if baseline_rows:
        lines.extend(
            [
                "",
                "### Against Best T/DR Baseline",
                "",
                "| Dataset | Variant | Best Baseline | Delta PEHE | Delta QINI | Verdict |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in baseline_rows:
            lines.append(
                f"| {row.get('dataset_id')} | {row.get('variant_id')} | {row.get('best_baseline')} | "
                f"{_variant_metric(row.get('delta_pehe_vs_baseline'))} | {_variant_metric(row.get('delta_qini_vs_baseline'))} | {row.get('verdict')} |"
            )

    failure = tuned.get("failure_attribution") or {}
    verdict_counts = _dict_inline(failure.get("verdict_counts"))
    explanation_rows = [str(row.get("reason") or row.get("interview_line") or "") for row in model_rows[:4]]
    if not explanation_rows:
        explanation_rows = [
            "Baseline rows are reference anchors; the safe claim is that T/DR gives a strong small-tabular floor before deep architecture claims."
        ]
    lines.extend(
        [
            "",
            "### Failure Attribution Reading",
            "",
            f"- Suite verdict distribution: {verdict_counts}.",
            "- If T/DR wins PEHE, explain small tabular sample size, stronger tree nuisance fits and lower neural optimization risk.",
            "- If a deep variant wins QINI but loses PEHE, frame it as ranking/policy candidate rather than clean CATE accuracy win.",
            "- Ablation rows compare each tuned variant with its default architecture, for example DragonNet_no_tarreg versus DragonNet_default or CFRNet alpha sweeps.",
            "- If DESCN is skipped on regression, call it a guarded task boundary because the current registry exposes DESCN as classification-only.",
            "- Do not over-claim: use battle cards to say exactly which dataset, metric and variant improved.",
            "",
            "Concrete rows to cite:",
            "",
            _bullet(explanation_rows),
            "",
            "Evidence: `reports/deep_model_tuned_benchmark_latest.json`, `reports/deep_model_tuned_benchmark_battle_cards_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md`.",
        ]
    )
    return "\n".join(lines)


def ablation_interpretation_section(model: ModelDeconstruction, payload: Mapping[str, Any] | None = None) -> str:
    if not payload:
        return """## Deep Model Ablation Interpretation

Run `scripts/generate_deep_model_ablation_interpretation.py` after the tuned benchmark and ablation gate to attach default-vs-variant metric movement and claim boundaries.
"""
    rows = [dict(row) for row in payload.get("rows") or [] if row.get("model") == model.model_id]
    scorecards = [dict(row) for row in payload.get("variant_scorecards") or [] if row.get("model") == model.model_id]
    summary = payload.get("summary") or {}
    if not rows and not scorecards:
        return f"""## Deep Model Ablation Interpretation

`{model.model_id}` is not part of the latest deep ablation interpretation artifact. Use the model's dedicated smoke, benchmark or multi-treatment evidence instead.
"""
    lines = [
        "## Deep Model Ablation Interpretation",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Report rows | `{summary.get('rows', 'NA')}` |",
        f"| Variants | `{summary.get('variants', 'NA')}` |",
        f"| Support rows | `{summary.get('support_rows', 'NA')}` |",
        f"| Tradeoff rows | `{summary.get('tradeoff_rows', 'NA')}` |",
        f"| Blocked rows | `{summary.get('blocked_rows', 'NA')}` |",
        "",
        "### Variant scorecards",
        "",
    ]
    if scorecards:
        for row in scorecards:
            lines.append(
                f"- `{row.get('variant_id')}`: mechanism={row.get('mechanism')}; PEHE improved={row.get('pehe_improved_datasets')}/{row.get('datasets')}; QINI improved={row.get('qini_improved_datasets')}/{row.get('datasets')}; policy improved={row.get('policy_improved_datasets')}/{row.get('datasets')}; {row.get('interview_line')}"
            )
    else:
        lines.append("- No variant scorecard for this model in the latest artifact.")
    lines.extend(["", "### Rows to cite", ""])
    cite_rows = []
    for row in rows[:8]:
        cite_rows.append(
            f"{row.get('dataset_id')} / {row.get('variant_id')}: verdict={row.get('verdict')}; delta PEHE={_variant_metric(row.get('delta_pehe_mean_vs_default'))}; delta QINI={_variant_metric(row.get('delta_qini_mean_vs_default'))}; safe={row.get('safe_claim')}"
        )
    lines.append(_bullet(cite_rows) if cite_rows else "- No rows to cite.")
    lines.extend(
        [
            "",
            "Evidence: `reports/deep_model_ablation_interpretation_latest.json`, `reports/deep_model_ablation_interpretation_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md`.",
        ]
    )
    return "\n".join(lines)


def ablation_promotion_section(model: ModelDeconstruction, payload: Mapping[str, Any] | None = None) -> str:
    if not payload:
        return """## Deep Model Ablation Promotion Matrix

Run `scripts/generate_deep_model_ablation_promotion_matrix.py` after ablation interpretation to attach claim tiers, promotion decisions and default-or-variant recommendations.
"""
    rows = [dict(row) for row in payload.get("rows") or [] if row.get("model") == model.model_id]
    summary = payload.get("summary") or {}
    if not rows:
        return f"""## Deep Model Ablation Promotion Matrix

`{model.model_id}` is not part of the latest deep ablation promotion matrix. Use its benchmark, training, model-card or multi-treatment evidence instead.
"""
    lines = [
        "## Deep Model Ablation Promotion Matrix",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Variants | `{summary.get('variants', 'NA')}` |",
        f"| Promotion-ready rows | `{summary.get('promotion_ready_rows', 'NA')}` |",
        f"| Tradeoff rows | `{summary.get('tradeoff_rows', 'NA')}` |",
        f"| Blocked/missing rows | `{summary.get('blocked_or_missing_rows', 'NA')}` |",
        "",
        "### Claim tiers to cite",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('variant_id')}`: tier={row.get('claim_tier')}; decision={row.get('promotion_decision')}; score={row.get('promotion_score')}; recommendation={row.get('recommended_claim_boundary')}; {row.get('interview_line')}"
        )
    lines.extend(
        [
            "",
            "### Safe boundary",
            "",
            _bullet(
                f"{row.get('variant_id')}: safe={row.get('safe_claim')} blocked={row.get('blocked_claim')}"
                for row in rows[:6]
            ),
            "",
            "Evidence: `reports/deep_model_ablation_promotion_matrix_latest.json`, `reports/deep_model_ablation_promotion_matrix_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md`.",
        ]
    )
    return "\n".join(lines)


def ablation_next_experiment_section(model: ModelDeconstruction, payload: Mapping[str, Any] | None = None) -> str:
    if not payload:
        return """## Deep Model Ablation Next Experiment Plan

Run `scripts/generate_deep_model_ablation_next_experiment_plan.py` after ablation promotion to attach concrete commands, metrics, pass gates and fail actions.
"""
    rows = [dict(row) for row in payload.get("rows") or [] if row.get("model") == model.model_id]
    summary = payload.get("summary") or {}
    if not rows:
        return f"""## Deep Model Ablation Next Experiment Plan

`{model.model_id}` is not part of the latest ablation next-experiment artifact.
"""
    lines = [
        "## Deep Model Ablation Next Experiment Plan",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| P0 rows | `{summary.get('p0_rows', 'NA')}` |",
        f"| P1 rows | `{summary.get('p1_rows', 'NA')}` |",
        f"| Experiment types | `{_dict_inline(summary.get('experiment_type_counts'))}` |",
        "",
        "### Next experiments to cite",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('variant_id')}`: priority={row.get('priority')}; experiment={row.get('experiment_type')}; command=`{row.get('recommended_command')}`; pass={row.get('pass_gate')}; fail={row.get('fail_action')}"
        )
    lines.extend(
        [
            "",
            "Evidence: `reports/deep_model_ablation_next_experiment_plan_latest.json`, `reports/deep_model_ablation_next_experiment_plan_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md`.",
        ]
    )
    return "\n".join(lines)


def ablation_command_contract_section(model: ModelDeconstruction, payload: Mapping[str, Any] | None = None) -> str:
    if not payload:
        return """## Deep Model Ablation Command Contract

Run `scripts/generate_deep_model_ablation_command_contract.py` after the next-experiment plan to audit planning commands and runnable fallback commands against the tuned benchmark runner CLI.
"""
    rows = [dict(row) for row in payload.get("rows") or [] if row.get("model") == model.model_id]
    summary = payload.get("summary") or {}
    if not rows:
        return f"""## Deep Model Ablation Command Contract

`{model.model_id}` is not part of the latest ablation command-contract artifact.
"""
    lines = [
        "## Deep Model Ablation Command Contract",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Fallback ready rows | `{summary.get('fallback_ready_rows', 'NA')}` |",
        f"| Planner adapter rows | `{summary.get('planner_adapter_rows', 'NA')}` |",
        f"| Status counts | `{_dict_inline(summary.get('status_counts'))}` |",
        "",
        "### Runnable command boundaries to cite",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('variant_id')}`: status={row.get('contract_status')}; fallback_ready={row.get('fallback_ready')}; unsupported_planner_flags={row.get('recommended_unsupported_flags')}; fallback=`{row.get('runnable_fallback_command')}`"
        )
    lines.extend(
        [
            "",
            "Evidence: `reports/deep_model_ablation_command_contract_latest.json`, `reports/deep_model_ablation_command_contract_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md`.",
        ]
    )
    return "\n".join(lines)


def ablation_command_contract_smoke_section(model: ModelDeconstruction, payload: Mapping[str, Any] | None = None) -> str:
    if not payload:
        return """## Deep Model Ablation Command Contract Smoke

Run `scripts/smoke_deep_model_ablation_command_contract.py` after the command contract to execute sampled fallback commands with isolated output.
"""
    rows = [dict(row) for row in payload.get("rows") or [] if row.get("model") == model.model_id]
    summary = payload.get("summary") or {}
    if not rows:
        return f"""## Deep Model Ablation Command Contract Smoke

`{model.model_id}` is not part of the latest fallback-command smoke sample. The smoke may still cover other variants; rerun with a larger limit or selected ordering if this model needs executable proof.
"""
    lines = [
        "## Deep Model Ablation Command Contract Smoke",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| OK rows | `{summary.get('ok_rows', 'NA')}` |",
        f"| Failed rows | `{summary.get('failed_rows', 'NA')}` |",
        f"| Sample limit | `{summary.get('sample_limit', 'NA')}` |",
        "",
        "### Executed fallback rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('variant_id')}`: status={row.get('status')}; runs={row.get('runs')}; ok_runs={row.get('ok_runs')}; manifest=`{row.get('runner_manifest')}`"
        )
    lines.extend(
        [
            "",
            "Evidence: `reports/deep_model_ablation_command_contract_smoke_latest.json`, `reports/deep_model_ablation_command_contract_smoke_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md`.",
        ]
    )
    return "\n".join(lines)


def ablation_recommended_command_smoke_section(model: ModelDeconstruction, payload: Mapping[str, Any] | None = None) -> str:
    if not payload:
        return """## Deep Model Ablation Recommended Command Smoke

Run `scripts/smoke_deep_model_ablation_recommended_commands.py` after the command contract to execute sampled runner-native recommended commands with safe isolated output.
"""
    rows = [dict(row) for row in payload.get("rows") or [] if row.get("model") == model.model_id]
    summary = payload.get("summary") or {}
    if not rows:
        return f"""## Deep Model Ablation Recommended Command Smoke

`{model.model_id}` is not part of the latest recommended-command smoke sample. The smoke may still cover other variants; rerun with a larger limit if this model needs runner-native command proof.
"""
    lines = [
        "## Deep Model Ablation Recommended Command Smoke",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| OK rows | `{summary.get('ok_rows', 'NA')}` |",
        f"| `--models` alias rows | `{summary.get('models_alias_rows', 'NA')}` |",
        f"| `--include-variants` alias rows | `{summary.get('include_variants_alias_rows', 'NA')}` |",
        f"| `--emit-battle-cards` alias rows | `{summary.get('emit_battle_cards_alias_rows', 'NA')}` |",
        "",
        "### Executed recommended rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('variant_id')}`: status={row.get('status')}; aliases=(models={row.get('uses_models_alias')}, include_variants={row.get('uses_include_variants_alias')}, emit_battle_cards={row.get('uses_emit_battle_cards_alias')}); manifest=`{row.get('runner_manifest')}`"
        )
    lines.extend(
        [
            "",
            "Evidence: `reports/deep_model_ablation_recommended_command_smoke_latest.json`, `reports/deep_model_ablation_recommended_command_smoke_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md`.",
        ]
    )
    return "\n".join(lines)


def ablation_command_smoke_coverage_section(model: ModelDeconstruction, payload: Mapping[str, Any] | None = None) -> str:
    if not payload:
        return """## Deep Model Ablation Command Smoke Coverage

Run `scripts/generate_deep_model_ablation_command_smoke_coverage.py` after command smoke artifacts to join contract-ready rows with executed smoke coverage.
"""
    rows = [dict(row) for row in payload.get("rows") or [] if row.get("model") == model.model_id]
    summary = payload.get("summary") or {}
    if not rows:
        return f"""## Deep Model Ablation Command Smoke Coverage

`{model.model_id}` has no command smoke coverage rows in the latest coverage artifact. Check whether it has ablation variants in the command contract.
"""
    lines = [
        "## Deep Model Ablation Command Smoke Coverage",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Any-smoked rows | `{summary.get('any_smoked_rows', 'NA')}` |",
        f"| Recommended-smoked rows | `{summary.get('recommended_smoked_rows', 'NA')}` |",
        f"| Fallback-smoked rows | `{summary.get('fallback_smoked_rows', 'NA')}` |",
        f"| Coverage gate | `{summary.get('coverage_gate', 'NA')}` |",
        "",
        "### Coverage rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('variant_id')}`: coverage_status={row.get('coverage_status')}; fallback_smoked={row.get('fallback_smoked')}; recommended_smoked={row.get('recommended_smoked')}; next_action={row.get('next_action')}"
        )
    lines.extend(
        [
            "",
            "Evidence: `reports/deep_model_ablation_command_smoke_coverage_latest.json`, `reports/deep_model_ablation_command_smoke_coverage_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md`.",
        ]
    )
    return "\n".join(lines)


def ablation_command_smoke_coverage_diff_section(model: ModelDeconstruction, payload: Mapping[str, Any] | None = None) -> str:
    if not payload:
        return """## Deep Model Ablation Command Smoke Coverage Diff

Run `scripts/generate_deep_model_ablation_command_smoke_coverage_diff.py` after command smoke coverage to produce the before/after coverage story.
"""
    rows = [dict(row) for row in payload.get("rows") or [] if row.get("model") == model.model_id]
    summary = payload.get("summary") or {}
    if not rows:
        return f"""## Deep Model Ablation Command Smoke Coverage Diff

`{model.model_id}` has no command smoke coverage diff rows in the latest artifact. Check whether it has ablation variants in the command contract.
"""
    lines = [
        "## Deep Model Ablation Command Smoke Coverage Diff",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Baseline fallback-smoked rows | `{summary.get('baseline_fallback_smoked_rows', 'NA')}/{summary.get('contract_rows', 'NA')}` |",
        f"| Current any-smoked rows | `{summary.get('current_any_smoked_rows', 'NA')}/{summary.get('contract_rows', 'NA')}` |",
        f"| Runner-native recommended-smoked rows | `{summary.get('runner_native_recommended_smoked_rows', 'NA')}/{summary.get('contract_rows', 'NA')}` |",
        f"| Gate delta | `{summary.get('coverage_gate_delta', 'NA')}` |",
        "",
        "### Diff rows",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row.get('variant_id')}`: transition={row.get('transition')}; smoke_gain={row.get('smoke_gain')}; claim={row.get('interview_claim')}; next_action={row.get('next_action')}"
        )
    lines.extend(
        [
            "",
            "Evidence: `reports/deep_model_ablation_command_smoke_coverage_diff_latest.json`, `reports/deep_model_ablation_command_smoke_coverage_diff_latest.csv`, `docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md`.",
        ]
    )
    return "\n".join(lines)


def benchmark_section(model: ModelDeconstruction, interpretation: Mapping[str, Any] | None = None) -> str:
    if not interpretation:
        return """## Benchmark-v3 Evidence

Run `scripts/run_paper_benchmark_suite.py --preset benchmark-v3` and `scripts/generate_paper_benchmark_interpretation.py` to attach model-specific win/loss evidence.
"""
    scorecards = interpretation.get("model_scorecards") or []
    scorecard = next((row for row in scorecards if row.get("model") == model.model_id), {})
    if not scorecard:
        return """## Benchmark-v3 Evidence

This model is not present in the latest paper benchmark interpretation artifact.
"""
    return f"""## Benchmark-v3 Evidence

| Field | Value |
| --- | --- |
| Benchmark tier | `{interpretation.get('benchmark_tier')}` |
| Seeds | `{interpretation.get('seeds')}` |
| Verdict | `{scorecard.get('verdict')}` |
| Mean PEHE | `{scorecard.get('pehe_mean_over_datasets')}` |
| Mean QINI | `{scorecard.get('qini_mean_over_datasets')}` |
| Metric wins | `{_dict_inline(scorecard.get('winner_counts'))}` |
| Stability | `{_dict_inline(scorecard.get('stability_counts'))}` |

Win/loss reading:

{_benchmark_wins(model.model_id, interpretation)}

How to explain it:

- Strength: {scorecard.get('strength')}
- Watchout: {scorecard.get('watchout')}
- Interview line: {scorecard.get('interview_line')}
- Evidence: `reports/paper_benchmark_latest.json`, `reports/paper_benchmark_interpretation_latest.json`, `docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md`
"""


def latest_benchmark_interpretation(root: str | Path = ".") -> dict[str, Any] | None:
    path = Path(root) / "reports" / "paper_benchmark_interpretation_latest.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def model_doc_markdown(
    model: ModelDeconstruction,
    interpretation: Mapping[str, Any] | None = None,
    tuned: Mapping[str, Any] | None = None,
) -> str:
    return f"""# {model.display_name} Model Deconstruction

## Positioning

| Field | Value |
| --- | --- |
| Model ID | `{model.model_id}` |
| Family | {model.family} |
| Current status | {model.current_status} |
| Role | {model.role} |

## Source Code Map

{_bullet(f'`{path}`' for path in model.code_paths)}

## From 0 To 1

{model.interview_line}

The shortest explanation is:

{_bullet(model.source_notes)}

## Math

{_bullet(model.formulas)}

## Call Chain

{_bullet(model.call_chain)}

## Data Flow

{_bullet(model.data_flow)}

## Training Flow

{_bullet(model.train_flow)}

## Prediction Flow

{_bullet(model.predict_flow)}

## Evaluation And Evidence

Metrics:

{_bullet(model.eval_metrics)}

Smoke / report evidence:

{_bullet(f'`{item}`' for item in model.smoke_evidence)}

{benchmark_section(model, interpretation)}

{tuned_benchmark_section(model, tuned)}

{ablation_interpretation_section(model, latest_deep_model_ablation_interpretation())}

{ablation_promotion_section(model, latest_deep_model_ablation_promotion_matrix())}

{ablation_next_experiment_section(model, latest_deep_model_ablation_next_experiment_plan())}

{ablation_command_contract_section(model, latest_deep_model_ablation_command_contract())}

{ablation_command_contract_smoke_section(model, latest_deep_model_ablation_command_contract_smoke())}

{ablation_recommended_command_smoke_section(model, latest_deep_model_ablation_recommended_command_smoke())}

{ablation_command_smoke_coverage_section(model, latest_deep_model_ablation_command_smoke_coverage())}

{ablation_command_smoke_coverage_diff_section(model, latest_deep_model_ablation_command_smoke_coverage_diff())}

## When To Use

{_bullet(model.when_to_use)}

## Failure Modes

{_bullet(model.failure_modes)}

## Implementation Gaps

{_bullet(model.implementation_gaps)}

## Next Engineering Tasks

{_bullet(model.next_tasks)}

## External Sources And License Gate

{source_rows(model)}

License policy: {model.license_policy}

## Interview Talk Track

5 minutes: explain the causal quantity, code path, loss, prediction contract and one evidence artifact.

15 minutes: add failure modes, benchmark/OPE metrics, readiness gate and why this model is selected over simpler baselines.

30 minutes: walk through source code, derive the objective, inspect artifacts, compare with external frameworks, and propose the next production hardening task.
"""


def index_markdown(catalog: Mapping[str, ModelDeconstruction] | None = None) -> str:
    catalog = catalog or MODEL_DECONSTRUCTIONS
    rows = [
        "| Model | Family | Status | Main code | Evidence | License stance |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model in catalog.values():
        rows.append(
            f"| [{model.display_name}](model_deconstruction/{model.model_id}.md) | {model.family} | "
            f"{model.current_status} | `{model.code_paths[0]}` | `{model.smoke_evidence[0]}` | {model.license_policy} |"
        )
    return f"""# DeepUplift Model Deconstruction Index

This index upgrades DeepUplift from a model registry into a model deconstruction workbench. Each selected model now has a source-code map, math objective, call chain, train/predict flow, evidence artifact and license gate.

## Deconstructed Models

{chr(10).join(rows)}

## Standard Template

Every model deconstruction must answer:

- What causal quantity does the model estimate?
- Which source files implement `fit`, `forward`, `loss`, `predict` and evaluation?
- What is the exact objective or pseudo-outcome?
- What artifacts prove that it can train/evaluate?
- What external paper/repo inspired it?
- What license risk exists if code is copied?
- What failure mode should be named before deployment?
- How would I explain it in 5/15/30 minutes?

## Current Coverage

- Meta learner: `TLearnerGBM`
- DR / orthogonal learner: `DRLearnerGBM`
- Representation balancing neural learner: `CFRNet`
- Propensity-aware neural learner: `DragonNet`
- Industrial interaction neural learner: `EFIN`
- Entire-space deep learner: `DESCN`
- Multi-treatment / LLM routing policy learner: `MultiDRLearnerGBM`

## Regression Gate

The Agent-facing model deconstruction capability is validated by `scripts/smoke_model_deconstruction_agent.py`, which writes `reports/model_deconstruction_agent_latest.json` and checks 20+ source-level golden questions.

## Deep Tuned Battle Gate

The deep-model failure-attribution layer is validated by `scripts/run_deep_model_tuned_benchmark.py`. The latest suite writes `reports/deep_model_tuned_benchmark_latest.json`, `reports/deep_model_tuned_benchmark_battle_cards_latest.csv` and `docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md`. Use it to explain why T/DR baselines often beat neural models on small tabular known-CATE slices and where tuned CFRNet/DragonNet/EFIN/DESCN variants improve ranking or policy value.
"""


def _deep_tuned_summary(tuned: Mapping[str, Any] | None) -> dict[str, Any]:
    if not tuned:
        return {"status": "missing"}
    return {
        "status": tuned.get("status"),
        "preset": tuned.get("preset"),
        "benchmark_tier": tuned.get("benchmark_tier"),
        "runs": len(tuned.get("runs") or []),
        "ok_runs": sum(1 for row in tuned.get("runs") or [] if row.get("status") == "ok"),
        "leaderboard_rows": len(tuned.get("leaderboard") or []),
        "battle_cards": len(tuned.get("battle_cards") or []),
        "verdict_counts": (tuned.get("failure_attribution") or {}).get("verdict_counts") or {},
    }


def deconstruction_payload(root: str | Path = ".") -> dict[str, Any]:
    tuned = latest_deep_model_tuned_benchmark(root)
    ablation = latest_deep_model_ablation_interpretation(root)
    ablation_promotion = latest_deep_model_ablation_promotion_matrix(root)
    ablation_next = latest_deep_model_ablation_next_experiment_plan(root)
    ablation_command_contract = latest_deep_model_ablation_command_contract(root)
    ablation_command_contract_smoke = latest_deep_model_ablation_command_contract_smoke(root)
    ablation_recommended_command_smoke = latest_deep_model_ablation_recommended_command_smoke(root)
    ablation_command_smoke_coverage = latest_deep_model_ablation_command_smoke_coverage(root)
    ablation_command_smoke_coverage_diff = latest_deep_model_ablation_command_smoke_coverage_diff(root)
    return {
        "schema_version": 1,
        "models": [asdict(model) for model in MODEL_DECONSTRUCTIONS.values()],
        "golden_questions": list(MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS),
        "latest_deep_tuned_benchmark": _deep_tuned_summary(tuned),
        "latest_deep_ablation_interpretation": (ablation or {}).get("summary") or {"status": "missing"},
        "latest_deep_ablation_promotion_matrix": (ablation_promotion or {}).get("summary") or {"status": "missing"},
        "latest_deep_ablation_next_experiment_plan": (ablation_next or {}).get("summary") or {"status": "missing"},
        "latest_deep_ablation_command_contract": (ablation_command_contract or {}).get("summary") or {"status": "missing"},
        "latest_deep_ablation_command_contract_smoke": (ablation_command_contract_smoke or {}).get("summary") or {"status": "missing"},
        "latest_deep_ablation_recommended_command_smoke": (ablation_recommended_command_smoke or {}).get("summary") or {"status": "missing"},
        "latest_deep_ablation_command_smoke_coverage": (ablation_command_smoke_coverage or {}).get("summary") or {"status": "missing"},
        "latest_deep_ablation_command_smoke_coverage_diff": (ablation_command_smoke_coverage_diff or {}).get("summary") or {"status": "missing"},
        "coverage": {
            "models": len(MODEL_DECONSTRUCTIONS),
            "golden_questions": len(MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS),
            "families": sorted({model.family for model in MODEL_DECONSTRUCTIONS.values()}),
            "deep_tuned_battle_cards": len((tuned or {}).get("battle_cards") or []),
            "deep_ablation_interpretation_rows": len((ablation or {}).get("rows") or []),
            "deep_ablation_promotion_rows": len((ablation_promotion or {}).get("rows") or []),
            "deep_ablation_next_experiment_rows": len((ablation_next or {}).get("rows") or []),
            "deep_ablation_command_contract_rows": len((ablation_command_contract or {}).get("rows") or []),
            "deep_ablation_command_contract_smoke_rows": len((ablation_command_contract_smoke or {}).get("rows") or []),
            "deep_ablation_recommended_command_smoke_rows": len((ablation_recommended_command_smoke or {}).get("rows") or []),
            "deep_ablation_command_smoke_coverage_rows": len((ablation_command_smoke_coverage or {}).get("rows") or []),
            "deep_ablation_command_smoke_coverage_diff_rows": len((ablation_command_smoke_coverage_diff or {}).get("rows") or []),
        },
    }


def model_deconstruction_reply(prompt: str, context: Mapping[str, Any] | None = None) -> str | None:
    if not is_model_deconstruction_prompt(prompt):
        return None
    model_id = detect_deconstruction_model(prompt) or "CFRNet"
    model = get_model_deconstruction(model_id)
    context = context or {}
    evidence_context = []
    if context.get("latest_run_dir"):
        evidence_context.append(f"- Current run evidence: `{context['latest_run_dir']}`")
    if context.get("model_name"):
        evidence_context.append(f"- Current UI model: `{context['model_name']}`")
    evidence_context_text = "\n".join(evidence_context) if evidence_context else "- Use repo evidence listed below."
    benchmark_text = benchmark_section(model, latest_benchmark_interpretation()).replace("## Benchmark-v3 Evidence", "#### Benchmark-v3 Evidence")
    tuned_text = tuned_benchmark_section(model, latest_deep_model_tuned_benchmark()).replace(
        "## Deep Tuned Benchmark And Failure Attribution",
        "#### Deep Tuned Benchmark And Failure Attribution",
    )
    ablation_text = ablation_interpretation_section(model, latest_deep_model_ablation_interpretation()).replace(
        "## Deep Model Ablation Interpretation",
        "#### Deep Model Ablation Interpretation",
    )
    ablation_promo_text = ablation_promotion_section(model, latest_deep_model_ablation_promotion_matrix()).replace(
        "## Deep Model Ablation Promotion Matrix",
        "#### Deep Model Ablation Promotion Matrix",
    )
    ablation_next_text = ablation_next_experiment_section(model, latest_deep_model_ablation_next_experiment_plan()).replace(
        "## Deep Model Ablation Next Experiment Plan",
        "#### Deep Model Ablation Next Experiment Plan",
    )
    ablation_command_contract_text = ablation_command_contract_section(
        model, latest_deep_model_ablation_command_contract()
    ).replace(
        "## Deep Model Ablation Command Contract",
        "#### Deep Model Ablation Command Contract",
    )
    ablation_command_contract_smoke_text = ablation_command_contract_smoke_section(
        model, latest_deep_model_ablation_command_contract_smoke()
    ).replace(
        "## Deep Model Ablation Command Contract Smoke",
        "#### Deep Model Ablation Command Contract Smoke",
    )
    ablation_recommended_command_smoke_text = ablation_recommended_command_smoke_section(
        model, latest_deep_model_ablation_recommended_command_smoke()
    ).replace(
        "## Deep Model Ablation Recommended Command Smoke",
        "#### Deep Model Ablation Recommended Command Smoke",
    )
    ablation_command_smoke_coverage_text = ablation_command_smoke_coverage_section(
        model, latest_deep_model_ablation_command_smoke_coverage()
    ).replace(
        "## Deep Model Ablation Command Smoke Coverage",
        "#### Deep Model Ablation Command Smoke Coverage",
    )
    ablation_command_smoke_coverage_diff_text = ablation_command_smoke_coverage_diff_section(
        model, latest_deep_model_ablation_command_smoke_coverage_diff()
    ).replace(
        "## Deep Model Ablation Command Smoke Coverage Diff",
        "#### Deep Model Ablation Command Smoke Coverage Diff",
    )
    return f"""### Model Deconstruction Agent 3.0: {model.display_name}

**一句话定位**：{model.interview_line}

#### 1. 0-1 原理
{_bullet(model.source_notes)}

#### 2. 关键公式
{_bullet(model.formulas)}

#### 3. 源码入口
{_bullet(f'`{path}`' for path in model.code_paths)}

#### 4. 源码调用链
{_bullet(f'`{path}`' for path in model.call_chain)}

#### 5. fit / forward / predict 拆解
训练流：
{_bullet(model.train_flow)}

推理流：
{_bullet(model.predict_flow)}

#### 6. 怎么训练和评估
Metric contract / 指标：{", ".join(model.eval_metrics)}。训练评估证据最终会落到 `metrics.json`、`predictions.csv`、`train_history.csv` 或对应 smoke report。

Model Deconstruction evidence: `docs/model_deconstruction/{model.model_id}.md`, `docs/DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md`, `reports/model_deconstruction_catalog_latest.json`, `reports/model_deconstruction_agent_latest.json`, `reports/deep_model_objective_diagnostics_latest.json`, `reports/algorithm_claim_ledger_latest.json`, `reports/paper_reproduction_gap_ledger_latest.json`, `reports/model_evidence_promotion_matrix_latest.json`, `reports/model_upgrade_recipe_cards_latest.json`, `reports/deep_model_telemetry_gap_matrix_latest.json`, `reports/deep_model_telemetry_smoke_latest.json`, `reports/deep_model_forward_hook_contracts_latest.json`, `reports/deep_model_forward_hook_smoke_latest.json`, `reports/deep_model_hook_training_bridge_latest.json`, `reports/deep_model_trainer_collector_smoke_latest.json`, `reports/deep_model_telemetry_benchmark_linkage_latest.json`, `reports/deep_model_telemetry_ablation_gate_latest.json`, `reports/deep_model_ablation_interpretation_latest.json`, `reports/deep_model_ablation_promotion_matrix_latest.json`, `reports/deep_model_ablation_next_experiment_plan_latest.json`, `reports/deep_model_ablation_command_contract_latest.json`, `reports/deep_model_ablation_command_contract_smoke_latest.json`, `reports/deep_model_ablation_recommended_command_smoke_latest.json`, `reports/deep_model_ablation_command_smoke_coverage_latest.json`, `reports/deep_model_ablation_command_smoke_coverage_diff_latest.json`, golden question regression.

Objective diagnostics / loss-term claim gate: `docs/DEEPUplift_DEEP_MODEL_OBJECTIVE_DIAGNOSTICS.md` maps each deep model's loss terms to observable diagnostics, PEHE/QINI/policy evidence, failure signals, safe claim and unsafe claim.

Algorithm claim ledger / 过度包装防线: `docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md` maps every model claim to formula, code refs, benchmark verdict, license boundary, safe claim and unsafe claim.

Paper reproduction gap ledger / 论文复现边界: `docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md` separates paper citation, open-source reference, local source reproduction, benchmark proof level, license risk and remaining implementation gaps.

Model evidence promotion matrix / 模型推广等级: `docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md` combines source, benchmark, objective diagnostics, license gate, safe claim, blocked claim and next evidence step into one review row per model.

Model upgrade recipe cards / 下一步实验卡: `docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md` turns the next evidence step into hypothesis, command, metrics, pass gate, fail action and expected artifacts.

Deep model telemetry gap matrix / 深度模型可观测性缺口: `docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md` maps each loss term to current signal, missing telemetry, implementation hook, pass gate, fail action and failure attribution boundary.

Deep model telemetry smoke / 基础可观测性验证: `docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md` reads loss history, predictions, metrics and manifest from the latest run artifacts, then marks each advanced hook such as phi/e_hat/attention/head export as a guarded gap.

Deep model forward hook contracts / 源码改造契约: `docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md` defines each forward hook before code refactor: tensor keys, artifact files, smoke test, pass gate, fallback behavior and backward compatibility. It covers CFRNet `phi_x`, DragonNet `e_hat`/tarreg, EFIN attention/path split and DESCN head/loss exports.

Deep model forward hook smoke / 可执行 hook 证据: `docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md` runs tiny guarded forward batches and writes sidecar artifacts for tensor exports: CFRNet `phi_x`, DragonNet `e_hat` and tarreg components, EFIN attention/path tensors, and DESCN exposure/head/constraint tensors.

Deep model hook training bridge / 训练期证据桥接: `docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md` joins hook sidecars with training run manifests, then defines optional trainer collectors, per-epoch columns, pass gates and fail actions before source-level refactors.

Deep model trainer collector smoke / 训练期采集验证: `docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md` runs tiny training loops and writes per-epoch advanced telemetry for CFRNet representation balance, DragonNet propensity/tarreg, EFIN attention/path and DESCN heads/constraints.

Deep model telemetry-benchmark linkage / 遥测到胜负归因: `docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md` joins internal telemetry with paper/tuned benchmark verdicts, baseline context, safe claim, blocked claim and the next ablation needed before stronger causal claims.

Deep model telemetry ablation gate / 消融实验门禁: `docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md` turns each linkage row into required variants, command, metrics to watch, pass gate, fail action, safe claim and blocked claim, so the reviewer can see which claims are still blocked.

Deep model ablation interpretation / 消融结果解读: `docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md` reads the latest tuned benchmark and explains each runnable variant's PEHE/QINI/policy/loss movement versus the default model, so variant coverage does not get confused with mechanism proof.

Deep model ablation promotion matrix / 消融 claim 升级矩阵: `docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md` turns those movements into claim tiers, promotion decisions, default-or-variant recommendations and blocked claims, so interview wording stays evidence-safe.

Deep model ablation next experiment plan / 下一步实验计划: `docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md` gives the recommended command, metrics to watch, pass gate and fail action for each variant, so the next optimization answer is concrete.

Deep model ablation command contract / 命令契约: `docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md` audits planning commands against the current tuned benchmark runner CLI and requires a runnable fallback command for every variant.

Deep model ablation command contract smoke / 命令执行 smoke: `docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md` executes sampled fallback commands with isolated output and `--no-update-latest`, proving the fallback path can run without overwriting main evidence.

Deep model ablation command smoke coverage / 命令 smoke 覆盖率: `docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md` separates smoke-executed variants from contract-ready backlog, so command evidence is not overclaimed as full benchmark proof.

Deep model ablation command smoke coverage diff / 覆盖率变化: `docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md` explains the before/after upgrade from fallback-only command smoke to full runner-native smoke coverage, while preserving the boundary between executability and benchmark superiority.

证据：
{_bullet(f'`{item}`' for item in model.smoke_evidence)}

{benchmark_text}

{tuned_text}

{ablation_text}

{ablation_promo_text}

{ablation_next_text}

{ablation_command_contract_text}

{ablation_command_contract_smoke_text}

{ablation_recommended_command_smoke_text}

{ablation_command_smoke_coverage_text}

{ablation_command_smoke_coverage_diff_text}

当前上下文：
{evidence_context_text}

#### 7. 适用场景和风险
适用：
{_bullet(model.when_to_use)}

风险：
{_bullet(model.failure_modes)}

#### 8. 外部来源和 license gate
{source_rows(model)}

License policy: {model.license_policy} Guarded adoption means GPL / unknown-license / heavy optional dependency code stays reference-only or isolated until the evidence and license gate is passed.

#### 9. 面试讲法
- 5分钟：先说 causal quantity，再说核心公式和源码入口，最后落到一个 smoke/report evidence。
- 15分钟：补充 failure mode、benchmark/OPE/readiness，说明为什么不是普通转化模型。
- 30分钟：逐行走 `fit -> loss/forward -> predict -> evaluator -> artifact`，再讲外部论文/开源对标和 license 决策。

#### 10. 下一步实现
{_bullet(model.next_tasks)}
"""


def score_deconstruction_answer(answer: str, expected_terms: Iterable[str] = ()) -> dict[str, Any]:
    lower = answer.lower()
    checks = {
        "source_code": any(term in lower for term in ("deepuplift/", "_fit", "forward", "predict", "loss")),
        "math": any(term in lower for term in ("tau", "mu", "pseudo", "ipm", "mmd", "propensity")),
        "training": "训练" in answer or "fit" in lower,
        "evaluation": any(term in lower for term in ("qini", "auuc", "policy", "evidence", "metrics")),
        "license": "license" in lower or "gpl" in lower or "mit" in lower or "apache" in lower,
        "interview": "5分钟" in answer and "15分钟" in answer and "30分钟" in answer,
    }
    missing_terms = [term for term in expected_terms if term.lower() not in lower]
    return {
        "checks": checks,
        "passed": all(checks.values()) and not missing_terms,
        "missing_terms": missing_terms,
        "ratio": sum(1 for value in checks.values() if value) / len(checks),
    }


def deconstruction_regression_rows(context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    for case in MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS:
        answer = model_deconstruction_reply(case["prompt"], context or {}) or ""
        score = score_deconstruction_answer(answer, case.get("expect") or ())
        rows.append(
            {
                "prompt": case["prompt"],
                "model": case["model"],
                "answer_chars": len(answer),
                "checks": score["checks"],
                "missing_terms": score["missing_terms"],
                "rubric_ratio": score["ratio"],
                "passed": score["passed"],
            }
        )
    return rows


def write_model_deconstruction_docs(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    docs_dir = root / "docs" / "model_deconstruction"
    docs_dir.mkdir(parents=True, exist_ok=True)
    interpretation = latest_benchmark_interpretation(root)
    tuned = latest_deep_model_tuned_benchmark(root)
    paths = []
    for model in MODEL_DECONSTRUCTIONS.values():
        path = docs_dir / f"{model.model_id}.md"
        path.write_text(model_doc_markdown(model, interpretation, tuned), encoding="utf-8")
        paths.append(path)
    index_path = root / "docs" / "DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md"
    index_path.write_text(index_markdown(), encoding="utf-8")
    reports_dir = root / "reports"
    reports_dir.mkdir(exist_ok=True)
    catalog_path = reports_dir / "model_deconstruction_catalog_latest.json"

    catalog_path.write_text(json.dumps(deconstruction_payload(root), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "ok",
        "index": str(index_path),
        "catalog": str(catalog_path),
        "docs": [str(path) for path in paths],
        "models": len(paths),
        "golden_questions": len(MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS),
    }
