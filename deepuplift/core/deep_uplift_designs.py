from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import csv
import os


LOSS_CATALOG: list[dict[str, str]] = [
    {
        "loss_id": "factual_outcome_bce_mse",
        "family": "factual outcome",
        "status": "ready",
        "objective": "Fit observed potential outcome with BCE for classification or MSE for regression.",
        "formula": "L_y = (1-t) * ell(y0_hat, y) + t * ell(y1_hat, y)",
        "business_use": "baseline conversion / value fit for every uplift model",
        "metric_link": "required but insufficient for QINI/AUUC and policy value",
        "code_path": "deepuplift/models/BaseModel.py:BaseLoss",
        "source": "TARNet / DragonNet / local DeepUplift",
        "source_url": "https://proceedings.mlr.press/v70/shalit17a.html",
        "risk": "good factual fit can still rank uplift poorly because counterfactual labels are missing",
        "interview_line": "I separate factual prediction quality from treatment-effect ranking quality.",
    },
    {
        "loss_id": "cfr_representation_balance_mmd",
        "family": "representation balance",
        "status": "ready",
        "objective": "Make treated/control representations comparable with differentiable MMD / energy / mean-L2 penalty.",
        "formula": "L = L_y + alpha * IPM(phi(X_treated), phi(X_control))",
        "business_use": "observational campaigns with treatment/control covariate shift",
        "metric_link": "supports overlap and sensitivity gates; does not replace QINI/AUUC",
        "code_path": "deepuplift/models/BaseModel.py:representation_balance_loss; deepuplift/models/CFRNet.py",
        "source": "CFRNet / counterfactual regression",
        "source_url": "https://proceedings.mlr.press/v70/shalit17a.html",
        "risk": "over-balancing can remove predictive heterogeneity; tune alpha and compare against DR/R baselines",
        "interview_line": "I fixed CFR balance so the IPM term is gradient-bearing, not just a logged scalar.",
    },
    {
        "loss_id": "dragonnet_propensity_targeted_regularization",
        "family": "propensity + targeted regularization",
        "status": "ready",
        "objective": "Jointly learn outcome heads and treatment propensity, then add targeted regularization.",
        "formula": "L = L_y + alpha * L_t + beta * L_targeted",
        "business_use": "binary treatment when treatment assignment signal itself is informative",
        "metric_link": "should improve robustness, then be checked by calibration, bootstrap CI and policy value",
        "code_path": "deepuplift/models/DragonNet.py; deepuplift/models/BaseModel.py:BaseLoss",
        "source": "DragonNet / targeted regularization",
        "source_url": "https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects",
        "risk": "propensity head can overfit selection; weak overlap still caps causal confidence",
        "interview_line": "DragonNet is useful because it makes propensity part of the neural objective, not a post-hoc diagnostic only.",
    },
    {
        "loss_id": "dr_r_pseudo_outcome_loss",
        "family": "orthogonal pseudo outcome",
        "status": "ready",
        "objective": "Train final effect model on DR/R pseudo outcomes instead of raw labels.",
        "formula": "tau_hat = argmin sum (pseudo_tau - f(X))^2",
        "business_use": "observational coupon, ads and CRM logs with confounding risk",
        "metric_link": "main P0 robust baseline before neural plugins",
        "code_path": "deepuplift/core/sklearn_models.py; deepuplift/core/external_models.py",
        "source": "EconML / CausalML meta-learners",
        "source_url": "https://econml.azurewebsites.net/",
        "risk": "pseudo outcomes are unstable with poor nuisance models or weak overlap",
        "interview_line": "DR/R losses are the bridge from causal identification to industrial tabular training.",
    },
    {
        "loss_id": "ipw_weighted_loss",
        "family": "propensity weighted",
        "status": "ready",
        "objective": "Reweight factual labels by inverse treatment propensity.",
        "formula": "L = t/e(X) * ell(y1_hat,y) + (1-t)/(1-e(X)) * ell(y0_hat,y)",
        "business_use": "bias correction sanity baseline for non-random targeting",
        "metric_link": "read with propensity overlap and weak-support diagnostics",
        "code_path": "deepuplift/core/sklearn_models.py; deepuplift/core/multitreatment.py",
        "source": "IPW / causal inference baseline",
        "source_url": "https://causalml.readthedocs.io/",
        "risk": "propensity near 0/1 creates high-variance training targets",
        "interview_line": "IPW is a useful stress test, but I never trust it without overlap diagnostics.",
    },
    {
        "loss_id": "descn_entire_space_cross_loss",
        "family": "entire-space ITE",
        "status": "ready",
        "objective": "Use propensity, treatment/control heads, entire-space conversion heads and cross-head constraints.",
        "formula": "L = L_propensity + L_es + L_h0/h1 + L_cross + L_balance",
        "business_use": "biased exposure / recommendation marketing logs",
        "metric_link": "compare under QINI/AUUC, overlap trimming and calibration",
        "code_path": "deepuplift/models/DESCN.py:esx_loss",
        "source": "DESCN / ESX-style uplift network",
        "source_url": "https://arxiv.org/search/?query=DESCN+uplift+modeling&searchtype=all",
        "risk": "head semantics and y0/y1 order must be normalized before evaluation",
        "interview_line": "This is a good example of why adapters must normalize model outputs before metrics.",
    },
    {
        "loss_id": "efin_interaction_constraint",
        "family": "treatment interaction",
        "status": "ready",
        "objective": "Learn user-treatment interactions and intervention-aware outcome heads.",
        "formula": "L = L_factual + L_treatment_aux + interaction regularization",
        "business_use": "coupon, creative, offer and recommendation treatment features",
        "metric_link": "evaluate with Top-K uplift and policy value because treatment interaction should improve targeting",
        "code_path": "deepuplift/models/EFIN.py; deepuplift/models/BaseModel.py:BaseLoss",
        "source": "EFIN KDD 2023",
        "source_url": "https://arxiv.org/abs/2306.00315",
        "risk": "interaction networks can overfit campaign IDs; compare with LightGBM DR/R baselines",
        "interview_line": "EFIN is where uplift becomes user-treatment interaction modeling, not just two outcome heads.",
    },
    {
        "loss_id": "uplift_ranking_surrogate",
        "family": "ranking / Top-K",
        "status": "loss_library_ready",
        "objective": "Optimize pairwise/listwise ranking so persuadable users move above sure things and sleeping dogs.",
        "formula": "L_rank = sum log(1 + exp(-(score_i - score_j) * sign(delta_i - delta_j)))",
        "business_use": "Top-K coupon, CRM and ads audience selection",
        "metric_link": "surrogate for uplift@K, QINI and AUUC",
        "code_path": "deepuplift/models/deep_losses.py:uplift_pairwise_ranking_loss",
        "source": "uplift ranking / QINI surrogate design",
        "source_url": "https://www.uplift-modeling.com/",
        "risk": "true pair labels are not observed; needs pseudo effects or randomized experiments",
        "interview_line": "I treat ranking losses as P1/P2 because their labels are counterfactual and need careful pseudo-outcome design.",
    },
    {
        "loss_id": "policy_value_incremental_profit_loss",
        "family": "business objective",
        "status": "loss_library_ready",
        "objective": "Optimize net value: incremental outcome value minus treatment cost, fatigue or model-routing cost.",
        "formula": "gain = uplift * value - treatment_cost - risk_penalty",
        "business_use": "coupon ROI, ads iROAS, push fatigue, LLM routing cost-quality",
        "metric_link": "directly aligned with Policy page and incremental profit",
        "code_path": "deepuplift/core/policy.py; deepuplift/models/deep_losses.py:cost_aware_policy_loss",
        "source": "decision-focused policy learning",
        "source_url": "https://arxiv.org/abs/2305.05176",
        "risk": "bad margin/cost assumptions can optimize the wrong business objective",
        "interview_line": "The launch objective is not AUUC; it is constrained incremental profit.",
    },
    {
        "loss_id": "calibration_ece_brier_uplift_loss",
        "family": "calibration",
        "status": "loss_library_ready",
        "objective": "Penalize mismatch between predicted uplift buckets and observed uplift buckets.",
        "formula": "L_cal = sum_b | mean(score_b) - observed_uplift_b |",
        "business_use": "threshold reliability, budget pacing and segment explanation",
        "metric_link": "maps to Evaluation calibration MAE and readiness gate",
        "code_path": "deepuplift/core/evaluator.py; deepuplift/models/deep_losses.py:uplift_calibration_loss",
        "source": "causal isotonic calibration",
        "source_url": "https://arxiv.org/abs/2302.14011",
        "risk": "bucket uplift is noisy; needs holdout and bootstrap",
        "interview_line": "A model can rank well but be badly calibrated for budget thresholds.",
    },
    {
        "loss_id": "full_funnel_ecup_multitask_loss",
        "family": "full-funnel multi-task",
        "status": "loss_library_ready",
        "objective": "Jointly model impression, click and conversion treatment effects.",
        "formula": "L = L_imp + L_click + L_conv + consistency/funnel constraints",
        "business_use": "ads, recommendation exposure and ecommerce funnel uplift",
        "metric_link": "maps to full_funnel_top10_* metrics",
        "code_path": "deepuplift/core/evaluator.py:full_funnel_uplift_metrics; deepuplift/models/deep_losses.py:full_funnel_multitask_loss",
        "source": "ECUP / full-funnel conversion uplift",
        "source_url": "https://arxiv.org/abs/2402.03379",
        "risk": "stage selection bias and delayed conversion can distort funnel attribution",
        "interview_line": "ECUP changes the label schema and metric system, not only the network body.",
    },
    {
        "loss_id": "delayed_feedback_window_loss",
        "family": "temporal / censoring",
        "status": "evaluator_ready_plugin_backlog",
        "objective": "Model D1/D7/D14/D30 outcomes and censoring instead of one immature label.",
        "formula": "L = sum_w L_y(window=w) + censoring / maturity regularizer",
        "business_use": "ads attribution, CRM retention and delayed purchase",
        "metric_link": "maps to delayed_d7/d14/d30_top10_uplift and censored rate",
        "code_path": "deepuplift/core/evaluator.py:delayed_feedback_uplift_metrics; planned temporal head",
        "source": "delayed feedback uplift",
        "source_url": "https://ojs.aaai.org/index.php/AAAI/article/view/38686",
        "risk": "future leakage and label-window mismatch can make offline uplift invalid",
        "interview_line": "Delayed feedback is a loss and data maturity problem, not just a reporting column.",
    },
    {
        "loss_id": "continuous_dose_response_loss",
        "family": "continuous treatment",
        "status": "baseline_ready_deep_backlog",
        "objective": "Learn outcome over treatment dose with interpolation and support constraints.",
        "formula": "L = ell(y_hat(x,d), y) + smoothness/support penalties",
        "business_use": "discount depth, subsidy amount, bid multiplier and contact frequency",
        "metric_link": "maps to dose-response gain and recommended dose policy",
        "code_path": "deepuplift/core/doseresponse.py; planned DRNet/VCNet plugin",
        "source": "VCNet / continuous treatment",
        "source_url": "https://openreview.net/forum?id=RmB-88r9dL",
        "risk": "unsafe extrapolation outside observed dose support",
        "interview_line": "Continuous treatment needs dose support diagnostics; binary uplift is not enough for discount depth.",
    },
    {
        "loss_id": "contrastive_uplift_representation_loss",
        "family": "contrastive representation",
        "status": "loss_library_ready",
        "objective": "Pull together comparable treated/control users with similar pseudo-effect signs and push apart opposite-effect groups.",
        "formula": "L = L_y + lambda * SupCon(phi_i, phi_j, pseudo_tau_sign)",
        "business_use": "hard-negative mining for sure thing / lost cause / sleeping dog separation",
        "metric_link": "targets Top-K ordering, calibration and segment stability",
        "code_path": "deepuplift/models/deep_losses.py:contrastive_uplift_representation_loss",
        "source": "contrastive causal representation learning",
        "source_url": "https://arxiv.org/search/?query=contrastive+learning+treatment+effect+estimation&searchtype=all",
        "risk": "pseudo labels can encode selection bias; needs randomized holdout or DR pseudo outcomes",
        "interview_line": "Contrastive uplift is promising, but the hard part is defining positives without observing counterfactuals.",
    },
    {
        "loss_id": "adversarial_treatment_confusion_loss",
        "family": "adversarial balance",
        "status": "research_backlog",
        "objective": "Use a treatment discriminator with gradient reversal so phi(X) is predictive for outcome but not treatment assignment.",
        "formula": "min_encoder L_y - lambda * L_treat_disc; max_disc L_treat_disc",
        "business_use": "biased targeting logs where representation balance matters",
        "metric_link": "read with overlap diagnostics and sensitivity checks",
        "code_path": "planned: adversarial balanced representation plugin",
        "source": "adversarial counterfactual regression",
        "source_url": "https://ojs.aaai.org/index.php/AAAI/article/view/29207",
        "risk": "too much confusion can erase true effect modifiers",
        "interview_line": "Adversarial balance is a sharper version of CFR: balance treatment signal without killing heterogeneity.",
    },
    {
        "loss_id": "llm_routing_cost_quality_loss",
        "family": "LLM routing policy",
        "status": "loss_library_ready",
        "objective": "Learn whether strong-model escalation creates enough incremental quality to justify cost and latency.",
        "formula": "gain = quality_uplift * value - extra_model_cost - latency_penalty",
        "business_use": "small vs strong model routing, RAG/tool/human escalation",
        "metric_link": "maps to Policy LLM routing ROI simulator and synthetic LLM routing dataset",
        "code_path": "deepuplift/models/deep_losses.py:llm_routing_cost_quality_loss; scripts/smoke_llm_routing_policy.py",
        "source": "FrugalGPT / RouteLLM / cost-aware routing",
        "source_url": "https://arxiv.org/abs/2305.05176",
        "risk": "judge labels, drift and budget constraints must be logged as treatment-design assumptions",
        "interview_line": "LLM routing is uplift: treatment is strong-model escalation, outcome is incremental quality net of cost.",
    },
    {
        "loss_id": "llm_multi_action_routing_loss",
        "family": "LLM multi-action policy",
        "status": "loss_library_ready",
        "objective": "Choose cheap model, strong model, RAG, tool or human-review action by expected incremental quality net of cost, latency and risk.",
        "formula": "pi(a|x)=softmax(g_a); gain_a=(q_a-q_cheap)*value-cost_a-latency_a-risk_a",
        "business_use": "RAG/tool/human escalation, safety review, customer-support routing and budget-paced model serving",
        "metric_link": "maps to multi-action LLM routing synthetic dataset, Top-K routing value and risk guardrails",
        "code_path": "deepuplift/models/deep_losses.py:llm_multi_action_routing_loss; examples/datasets/synthetic_llm_multi_action_routing_9k.csv",
        "source": "FrugalGPT cascade + RouteLLM router + action-aware LLM serving",
        "source_url": "https://arxiv.org/abs/2305.05176",
        "risk": "requires randomized exploration across actions; historical routing logs can copy old policy bias",
        "interview_line": "LLM routing is not only cheap vs strong; RAG, tool and human review are different treatments with different costs and risks.",
    },
]


NETWORK_ARCHITECTURE_CATALOG: list[dict[str, str]] = [
    {
        "architecture": "TARNet",
        "status": "ready",
        "core_blocks": "shared representation phi(X) + y0/y1 outcome heads",
        "loss_stack": "factual outcome BCE/MSE",
        "best_for": "first neural baseline after GBM meta-learners",
        "code_path": "deepuplift/models/TarNet.py",
        "source_url": "https://proceedings.mlr.press/v70/shalit17a.html",
        "interview_line": "TARNet is the simplest deep CATE contract: shared encoder, two potential-outcome heads.",
    },
    {
        "architecture": "CFRNet",
        "status": "ready",
        "core_blocks": "TARNet + differentiable representation balance",
        "loss_stack": "factual outcome + MMD/energy/mean-L2 IPM",
        "best_for": "treatment/control covariate shift and overlap pressure tests",
        "code_path": "deepuplift/models/CFRNet.py; deepuplift/models/BaseModel.py:representation_balance_loss",
        "source_url": "https://proceedings.mlr.press/v70/shalit17a.html",
        "interview_line": "I made the balance penalty differentiable so CFRNet really trains balanced representations.",
    },
    {
        "architecture": "DragonNet",
        "status": "ready",
        "core_blocks": "shared encoder + y0/y1 heads + propensity head + epsilon targeted regularization",
        "loss_stack": "factual outcome + treatment BCE + targeted regularization",
        "best_for": "binary treatment with non-random assignment signal",
        "code_path": "deepuplift/models/DragonNet.py",
        "source_url": "https://papers.nips.cc/paper/8520-adapting-neural-networks-for-the-estimation-oftreatment-effects",
        "interview_line": "DragonNet turns propensity into a trainable neural head and uses targeted regularization for effect estimation.",
    },
    {
        "architecture": "EFIN interaction network",
        "status": "ready",
        "core_blocks": "user encoder + treatment-aware interaction + uplift heads",
        "loss_stack": "factual outcome + treatment auxiliary / interaction constraint",
        "best_for": "coupon, offer, ad creative and recommendation intervention features",
        "code_path": "deepuplift/models/EFIN.py",
        "source_url": "https://arxiv.org/abs/2306.00315",
        "interview_line": "EFIN is the deep model to discuss when treatment has its own rich features.",
    },
    {
        "architecture": "DESCN / ESX entire-space network",
        "status": "ready",
        "core_blocks": "propensity tower + treatment/control towers + entire-space conversion heads + tau head",
        "loss_stack": "propensity BCE + treatment/control BCE + cross-head constraints + balance",
        "best_for": "biased recommendation or campaign exposure logs",
        "code_path": "deepuplift/models/DESCN.py",
        "source_url": "https://arxiv.org/search/?query=DESCN+uplift+modeling&searchtype=all",
        "interview_line": "DESCN is a good interview example for biased entire-space treatment effect modeling.",
    },
    {
        "architecture": "ECUP full-funnel multi-task network",
        "status": "evaluator_ready_plugin_backlog",
        "core_blocks": "shared encoder + impression/click/conversion heads + funnel consistency",
        "loss_stack": "stage BCE losses + conversion uplift objective",
        "best_for": "ads and recommendation funnels",
        "code_path": "deepuplift/core/evaluator.py:full_funnel_uplift_metrics; deepuplift/models/deep_losses.py:full_funnel_multitask_loss; planned neural adapter",
        "source_url": "https://arxiv.org/abs/2402.03379",
        "interview_line": "ECUP matters because uplift may happen at exposure, click or conversion stage.",
    },
    {
        "architecture": "Delayed feedback temporal head",
        "status": "evaluator_ready_plugin_backlog",
        "core_blocks": "shared encoder + D1/D7/D14/D30 heads + censoring/maturity indicators",
        "loss_stack": "windowed conversion losses + censoring-aware regularizer",
        "best_for": "retention, ads attribution and delayed purchase",
        "code_path": "deepuplift/core/evaluator.py:delayed_feedback_uplift_metrics; planned temporal adapter",
        "source_url": "https://ojs.aaai.org/index.php/AAAI/article/view/38686",
        "interview_line": "Delayed-feedback uplift changes when labels are trustworthy, not just how a curve is drawn.",
    },
    {
        "architecture": "DRNet / VCNet dose-response",
        "status": "baseline_ready_deep_backlog",
        "core_blocks": "dose-conditioned outcome network with continuous treatment encoding",
        "loss_stack": "factual dose outcome + smoothness/support constraints",
        "best_for": "discount amount, subsidy amount, bid multiplier and contact frequency",
        "code_path": "deepuplift/core/doseresponse.py; planned VCNet/DRNet adapter",
        "source_url": "https://openreview.net/forum?id=RmB-88r9dL",
        "interview_line": "Dose-response networks prevent forcing a numeric treatment into a send/no-send framing.",
    },
    {
        "architecture": "Multi-treatment attention / action network",
        "status": "baseline_ready_deep_backlog",
        "core_blocks": "shared encoder + action/treatment embedding + action-specific heads or attention",
        "loss_stack": "multi-action factual loss + IPW/DR policy value",
        "best_for": "coupon type, channel, offer and journey action selection",
        "code_path": "deepuplift/core/multitreatment.py; planned attention adapter",
        "source_url": "https://arxiv.org/abs/2605.03319",
        "interview_line": "Multi-treatment is what turns uplift from binary targeting into action recommendation.",
    },
    {
        "architecture": "Contrastive uplift encoder",
        "status": "ready",
        "core_blocks": "TARNet encoder + supervised contrastive representation penalty + y0/y1 heads",
        "loss_stack": "factual outcome + DR pseudo-effect contrastive loss",
        "best_for": "segment-stable Top-K ranking and hard-negative mining",
        "code_path": "deepuplift/models/ContrastiveUpliftNet.py; deepuplift/models/deep_losses.py:contrastive_uplift_representation_loss",
        "source_url": "https://arxiv.org/search/?query=contrastive+learning+treatment+effect+estimation&searchtype=all",
        "interview_line": "The key is defining contrastive positives with pseudo-effects, not raw conversions.",
    },
    {
        "architecture": "Adversarial balanced representation network",
        "status": "research_backlog",
        "core_blocks": "encoder + outcome heads + treatment discriminator with gradient reversal",
        "loss_stack": "factual outcome - lambda * discriminator treatment loss",
        "best_for": "highly biased campaign logs",
        "code_path": "planned: adversarial balance plugin",
        "source_url": "https://ojs.aaai.org/index.php/AAAI/article/view/29207",
        "interview_line": "Adversarial balance is useful when propensity is too easy to predict from representation.",
    },
    {
        "architecture": "CRM journey transformer",
        "status": "research_backlog",
        "core_blocks": "sequence encoder over repeated touches + treatment/action head",
        "loss_stack": "sequential factual losses + delayed feedback / fatigue penalty",
        "best_for": "overlapping journeys, push frequency and lifecycle CRM",
        "code_path": "planned: sequence uplift adapter",
        "source_url": "https://arxiv.org/abs/2605.14284",
        "interview_line": "Repeated treatment requires sequence modeling and holdout design, not row-wise independence assumptions.",
    },
    {
        "architecture": "LLM embedding uplift head",
        "status": "research_backlog",
        "core_blocks": "text/prompt/creative embedding + structured features + uplift head",
        "loss_stack": "text-treatment factual loss + contrastive / policy value regularizer",
        "best_for": "ad creative, push copy, offer description and prompt variants",
        "code_path": "planned: text treatment uplift plugin",
        "source_url": "https://arxiv.org/abs/2605.07834",
        "interview_line": "LLM embeddings let text become treatment features, but causal logging still matters.",
    },
    {
        "architecture": "LLM routing uplift network",
        "status": "policy_ready_training_backlog",
        "core_blocks": "query/task encoder + cheap/strong model potential-outcome heads + cost head",
        "loss_stack": "quality factual loss + cost-quality policy loss + calibration",
        "best_for": "strong model escalation, RAG/tool/human review routing",
        "code_path": "deepuplift/models/deep_losses.py:llm_routing_cost_quality_loss; examples/datasets/synthetic_llm_routing_uplift_8k.csv; scripts/smoke_llm_routing_policy.py",
        "source_url": "https://arxiv.org/abs/2305.05176",
        "interview_line": "Routing is a treatment-effect problem because we need incremental quality over the cheap baseline.",
    },
    {
        "architecture": "Multi-action LLM routing network",
        "status": "loss_ready_guarded_adapter_backlog",
        "core_blocks": "query/task encoder + action-aware heads for strong/RAG/tool/human + cheap fallback + cost/risk heads",
        "loss_stack": "multi-action cost-quality loss + factual quality loss + calibration + budget-rate penalty",
        "best_for": "RAG/tool/human review escalation under evidence, safety, latency and budget constraints",
        "code_path": "deepuplift/models/deep_losses.py:llm_multi_action_routing_loss; examples/datasets/synthetic_llm_multi_action_routing_9k.csv",
        "source_url": "https://github.com/lm-sys/RouteLLM",
        "interview_line": "I treat each escalation path as a treatment arm and keep a cheap-model fallback, so the router can learn when not to escalate.",
    },
]


DEEP_INTERVIEW_DRILL_ROWS: list[dict[str, str]] = [
    {
        "question": "为什么 BCE/MSE 不是 uplift 深度模型的完整目标？",
        "core_answer": "BCE/MSE 只拟合 factual outcome，不能直接保证 treatment-effect 排序、Top-K policy value 或 counterfactual robustness。",
        "formula_or_objective": "L_y=(1-t)ell(y0,y)+t ell(y1,y); tau(x)=y1_hat-y0_hat",
        "project_evidence": "Models -> Deep Uplift Loss Catalog; Evaluation -> Loss-to-Metric Map; deepuplift/models/deep_losses.py",
        "follow_up": "追问时解释为什么 factual AUC 高仍可能 QINI/ROI 差，并用 policy value 作为上线目标。",
    },
    {
        "question": "CFRNet 的 representation balance 到底解决什么问题？",
        "core_answer": "当 treatment/control 特征分布不同，CFRNet 用 IPM/MMD/Wasserstein 风格 penalty 让表示空间更可比，降低 extrapolation 风险。",
        "formula_or_objective": "L = L_y + alpha * IPM(phi(X_t), phi(X_c))",
        "project_evidence": "deepuplift/models/BaseModel.py:representation_balance_loss; deepuplift/models/CFRNet.py; CFR balance smoke",
        "follow_up": "追问时强调 balance 不能过强，否则会抹掉真实 effect modifiers，需要和 DR/R/LightGBM baseline 对照。",
    },
    {
        "question": "DragonNet 的 targeted regularization 为什么值得讲？",
        "core_answer": "DragonNet 把 propensity head 和 outcome heads 联合训练，再用 targeted regularization 把 treatment assignment 信息用于更稳健的 effect estimation。",
        "formula_or_objective": "L = L_y + alpha L_t + beta L_targeted",
        "project_evidence": "deepuplift/models/DragonNet.py; Models -> Network Architecture Catalog",
        "follow_up": "追问时说明 propensity 极端或 overlap 差时 targeted regularization 也不能替代实验设计。",
    },
    {
        "question": "EFIN / treatment-aware interaction 和普通双塔有什么区别？",
        "core_answer": "EFIN 把 treatment/offer/creative 特征显式建模，学习 user-treatment 交互，适合优惠券、多文案、推荐干预等 treatment 本身有信息的场景。",
        "formula_or_objective": "L = L_factual + L_aux(treatment) + interaction regularization",
        "project_evidence": "deepuplift/models/EFIN.py; Scenario Model Matrix; coupon/creative/treatment-feature scenario cards",
        "follow_up": "追问时说 interaction 模型容易过拟合 campaign ID，所以必须和 T/DR/R learner + LightGBM 对照。",
    },
    {
        "question": "ECUP / full-funnel uplift 为什么不是普通 conversion uplift？",
        "core_answer": "广告和推荐里 treatment 先影响 impression/click，再影响 conversion；只看最后转化会忽略 stage selection bias 和 click-conversion 不一致。",
        "formula_or_objective": "L = L_imp + L_click + L_conv + funnel consistency; report uplift_imp/uplift_click/uplift_conv",
        "project_evidence": "deepuplift/core/evaluator.py:full_funnel_uplift_metrics; synthetic_shopee_ads_full_funnel_8k; Evaluation -> Frontier Metric Source Trace",
        "follow_up": "追问时解释 click uplift 高但 conversion uplift 低时不能上线加预算。",
    },
    {
        "question": "Delayed feedback uplift 为什么会误导短期评估？",
        "core_answer": "D1 标签可能还没成熟，D30 才体现真实转化或留存；短窗口会把慢响应人群误判成无效，甚至反向优化。",
        "formula_or_objective": "L = sum_w L_y(window=w) + censoring/maturity regularizer; metric = uplift_D1/D7/D14/D30",
        "project_evidence": "deepuplift/core/evaluator.py:delayed_feedback_uplift_metrics; delayed feedback synthetic dataset; diagnostic delayed_feedback_trap",
        "follow_up": "追问时讲 label maturity、censoring、future leakage 和线上回收窗口。",
    },
    {
        "question": "Contrastive uplift 的正负样本怎么定义？",
        "core_answer": "不能用 raw conversion 定义 positives；要用 randomized holdout 或 DR/R pseudo-effect sign，把 persuadable/sure thing/lost cause/sleeping dog 分开。",
        "formula_or_objective": "L = L_y + lambda * SupCon(phi_i, phi_j, pseudo_tau_sign)",
        "project_evidence": "deepuplift/models/ContrastiveUpliftNet.py; contrastive_uplift_representation_loss; smoke_deep_loss_functions.py",
        "follow_up": "追问时强调 pseudo label 有 selection bias 风险，必须看 overlap/bootstrap/calibration。",
    },
    {
        "question": "LLM routing 为什么是 uplift 问题，而不是普通分类问题？",
        "core_answer": "routing 的 treatment 是 strong/RAG/tool/human escalation，control 是 cheap model；目标是增量质量是否覆盖增量成本、延迟和风险。",
        "formula_or_objective": "gain_a=(q_a-q_cheap)*value-cost_a-latency_a-risk_a; pi(a|x)=softmax(g_a)",
        "project_evidence": "Synthetic LLM Multi-Action Routing 9k; Policy -> 多动作路由策略样例; llm_multi_action_routing_loss",
        "follow_up": "追问时讲 randomized exploration、judge calibration、预算 pacing 和 policy drift。",
    },
    {
        "question": "连续 treatment 为什么不能简单切成 treatment/control？",
        "core_answer": "补贴金额、折扣率、出价倍率和触达频次是 dose，不同剂量的成本和边际收益非线性；二值化会丢掉最优强度和 support 风险。",
        "formula_or_objective": "mu(x,d)=E[Y|X=x,D=d]; choose d*=argmax_d mu_uplift(x,d)*value-cost(d)",
        "project_evidence": "Dose-Response page; deepuplift/core/doseresponse.py; continuous_dose_response_loss; synthetic_continuous_bid_discount_uplift_5k",
        "follow_up": "追问时讲 DRNet/VCNet、dose support、unsafe extrapolation 和边际 ROI 递减。",
    },
    {
        "question": "Policy learning / causal bandit 和 uplift 的关系是什么？",
        "core_answer": "uplift 给出离线可解释固定 policy；当环境非平稳、预算动态或 action 池变化时，bandit 在有探索/OPE/guardrail 后做在线自适应。",
        "formula_or_objective": "V_DR(pi)=mean[mu_hat(pi(x),x)+1{a=pi(x)}/p(a|x)*(y-mu_hat(a,x))]",
        "project_evidence": "Policy -> 从离线 uplift 到在线 bandit 的上线门禁; exploration_ope_guardrail_rows",
        "follow_up": "追问时讲 randomized logging、IPS/DR OPE、budget pacing、capacity constraints 和 holdout。",
    },
]


def deep_loss_catalog() -> list[dict[str, str]]:
    return deepcopy(LOSS_CATALOG)


def network_architecture_catalog() -> list[dict[str, str]]:
    return deepcopy(NETWORK_ARCHITECTURE_CATALOG)


def deep_interview_drill_rows() -> list[dict[str, str]]:
    return deepcopy(DEEP_INTERVIEW_DRILL_ROWS)


def deep_design_counts() -> dict[str, int]:
    rows = LOSS_CATALOG + NETWORK_ARCHITECTURE_CATALOG
    counts = {"losses": len(LOSS_CATALOG), "architectures": len(NETWORK_ARCHITECTURE_CATALOG)}
    for row in rows:
        status = row.get("status", "")
        counts[status] = counts.get(status, 0) + 1
    return counts


def loss_metric_mapping_rows() -> list[dict[str, str]]:
    return [
        {
            "training_signal": row["loss_id"],
            "loss_family": row["family"],
            "nearest_offline_metric": row["metric_link"],
            "business_decision": row["business_use"],
            "code_path": row["code_path"],
        }
        for row in LOSS_CATALOG
    ]


def deep_source_rows() -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    for row in LOSS_CATALOG + NETWORK_ARCHITECTURE_CATALOG:
        key = (row.get("source", row.get("architecture", "")), row.get("source_url", ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "topic": row.get("loss_id") or row.get("architecture", ""),
                "source": row.get("source", row.get("architecture", "")),
                "source_url": row.get("source_url", ""),
                "status": row.get("status", ""),
                "code_path": row.get("code_path", ""),
            }
        )
    return rows


def local_causal_report_rows(limit: int = 12) -> list[dict[str, str]]:
    configured_path = os.environ.get("DEEPUPlIFT_RESEARCH_ARTICLES", "").strip()
    if not configured_path:
        return []
    csv_path = Path(configured_path).expanduser()
    if not csv_path.exists():
        return []
    keywords = [
        "treatment",
        "counterfactual",
        "causal",
        "foundation",
        "continuous",
        "llm",
        "routing",
        "calibration",
    ]
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            title = row.get("title", "")
            lower = title.lower()
            if any(keyword in lower for keyword in keywords):
                rows.append(
                    {
                        "title": title,
                        "url": row.get("url", ""),
                        "source": row.get("source_name", row.get("source_id", "")),
                        "trust_level": row.get("trust_level", ""),
                        "fetched_at": row.get("fetched_at", ""),
                    }
                )
            if len(rows) >= limit:
                break
    return rows


def deep_design_agent_reply(prompt: str) -> str | None:
    text = prompt.lower()
    if any(key in text for key in ["深度面试", "算法面试", "loss 面试", "network 面试", "深度 uplift 面试", "追问"]):
        rows = DEEP_INTERVIEW_DRILL_ROWS
        bullets = "\n".join(
            f"- **{row['question']}**：{row['core_answer']} 证据：`{row['project_evidence']}`"
            for row in rows
        )
        return (
            "### 深度 uplift / LLM routing 算法面试追问卡\n\n"
            f"{bullets}\n\n"
            "面试主线：先讲 factual outcome 不是终点，再讲 balance/targeted regularization/interaction/full-funnel/"
            "delayed/contrastive/cost-quality 如何分别服务于 QINI、Top-K、policy value、calibration 和上线 guardrail。\n\n"
            "UI 证据：`Models -> 深度算法面试追问 Drill`、`Evaluation -> Loss-to-Metric Map`、"
            "`Policy -> LLM Routing ROI Simulator`、`Evidence -> Deep Loss / Network Source Gate`。"
        )
    if any(key in text for key in ["llm routing loss", "llm loss", "大模型 loss", "cost-quality", "强模型", "小模型"]):
        return (
            "### LLM routing 的 uplift loss\n\n"
            "LLM routing 可以写成 treatment effect：control 是 cheap model，treatment 是 strong model/RAG/tool/human escalation，"
            "outcome 是质量或任务成功，loss/decision objective 是 `quality_uplift * value - extra_model_cost - latency_penalty`。"
            "这不是普通分类，因为我们关心的是强模型相对便宜模型的增量，而不是请求本身是否困难。\n\n"
            "平台证据：`Policy -> LLM Routing ROI Simulator`、`Synthetic LLM Routing Uplift 8k`、"
            "`deepuplift/models/deep_losses.py:llm_routing_cost_quality_loss`、"
            "`scripts/smoke_deep_loss_functions.py`、`scripts/smoke_llm_routing_policy.py`、"
            "`Models -> Network Architecture Catalog`。"
        )
    if any(key in text for key in ["loss", "损失", "qini loss", "auuc loss", "policy loss", "bce", "mse"]):
        return (
            "### Deep uplift loss 怎么设计\n\n"
            "我把 loss 分成四层：1) factual outcome BCE/MSE，保证 y0/y1 头能拟合观测标签；"
            "2) causal robustness loss，例如 CFRNet 的 differentiable MMD/IPM balance、DragonNet 的 propensity + targeted regularization、DR/R pseudo-outcome；"
            "3) decision loss，例如 uplift ranking surrogate、calibration、policy value / incremental profit；"
            "4) frontier loss，例如 delayed-feedback window、ECUP full-funnel multi-task、contrastive uplift、LLM routing cost-quality。\n\n"
            "当前已落地的代码证据：`deepuplift/models/BaseModel.py:representation_balance_loss`、"
            "`deepuplift/models/CFRNet.py`、`deepuplift/models/DragonNet.py`、`deepuplift/models/deep_losses.py`、"
            "`scripts/smoke_deep_loss_functions.py`、`scripts/smoke_cfrnet_training_balance.py`。"
            "UI 证据：`Models -> Deep Uplift Loss Catalog`、`Evaluation -> Loss-to-Metric Map`、`Evidence -> Deep Loss / Network Source Gate`。"
        )
    if any(key in text for key in ["network", "architecture", "网络结构", "结构", "tarnet", "cfrnet", "dragonnet", "efin", "descn"]):
        return (
            "### Deep uplift network 结构怎么讲\n\n"
            "主线是从简单到复杂：TARNet 是 shared representation + y0/y1 heads；"
            "CFRNet 加 representation balance；DragonNet 加 propensity head 和 targeted regularization；"
            "EFIN 强调 user-treatment interaction；DESCN/ESX 处理 biased entire-space ITE；"
            "ECUP、delayed-feedback、dose-response、contrastive encoder 和 LLM routing network 是 P2 场景驱动扩展。\n\n"
            "面试讲法：我不是把深度模型当黑盒，而是把每个 architecture 拆成 encoder、head、loss stack、业务场景和评估指标。"
            "UI 证据：`Models -> Network Architecture Catalog`、`Story -> Deep Uplift Architecture Interview Card`。"
        )
    if any(key in text for key in ["contrastive", "对比学习", "hard negative", "sure thing", "sleeping dog"]):
        return (
            "### 对比学习怎么用于 uplift\n\n"
            "uplift 的难点是没有每个人的反事实标签，所以 contrastive positives 不能直接用普通转化标签定义。"
            "更靠谱的做法是先用随机实验或 DR/R pseudo-outcome 得到 effect sign / bucket，再做 supervised contrastive："
            "把相似且 pseudo-effect 同号的人拉近，把 sure thing、lost cause、sleeping dog 和 persuadable 拉开。"
            "这服务于 Top-K 排序、segment stability 和 calibration，而不是替代因果识别。\n\n"
            "当前状态是 ready neural baseline：`deepuplift/models/deep_losses.py` 已有 "
            "`contrastive_uplift_representation_loss`，`deepuplift/models/ContrastiveUpliftNet.py` 已接入 registry，"
            "并由 `scripts/smoke_deep_loss_functions.py` 验证梯度；"
            "UI 证据在 `Models -> Deep Uplift Loss Catalog` 和 `docs/UPLIFT_CONTRASTIVE_AND_LLM_UPLIFT.md`。"
        )
    return None
