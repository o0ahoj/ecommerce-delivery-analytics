# E-Commerce Delivery Analytics

Predicting late deliveries and segmenting shipment profiles for an e-commerce logistics operation, using cost-sensitive machine learning.

**Stack:** Python · scikit-learn · pandas · SHAP · LIME · matplotlib / seaborn

---

## The business problem

Late deliveries are one of the leading causes of customer churn in e-commerce. The damage, however, is less about the delay itself than about the customer being *surprised* by it. If a delay can be predicted before dispatch, the company can notify the customer proactively, offer compensation, or reprioritise the shipment.

This project answers two questions:

1. **Can we predict, at order time, whether a shipment will arrive late?** (supervised classification)
2. **Do shipments fall into distinct operational profiles that warrant different handling?** (unsupervised clustering)

## Data

[Customer Analytics — E-Commerce Shipping Data](https://www.kaggle.com/datasets/prachi13/customer-analytics/data) (Kaggle, public)

10,999 shipment records × 12 attributes: warehouse block, shipping mode, product cost, discount, weight, prior purchases, customer care calls, customer rating, product importance, and the on-time / late outcome. The target is moderately imbalanced (roughly 60 / 40 in favour of late deliveries).

---

## Part 1 — Delivery delay classification

### Cost-sensitive framing

Standard accuracy is misleading here. With a 60/40 imbalance, a model that predicts "late" for every shipment scores 60% accuracy while providing no decision value at all. F1 is also inappropriate, because the two error types are not equally expensive.

A business cost matrix was defined instead:

| | Predicted: On Time | Predicted: Late |
|---|---|---|
| **Actual: On Time** | 0 | 2 — unnecessary intervention |
| **Actual: Late** | 10 — missed late delivery | 0 |

Missing a late delivery costs **five times** more than a false alarm. All model selection was therefore driven by **total business cost** rather than accuracy, with Class 0 recall as the secondary business-aligned metric.

### Models compared

| Model | Business cost | AUC | Notes |
|---|---|---|---|
| Decision Tree (cost-optimised, 9 selected features) | **5,134** | 0.667 | Lowest raw cost; `gini`, `max_depth=15`, `min_samples_split=5` |
| **Random Forest (default, reduced features)** | **5,150** | **0.725** | **Recommended** — near-identical cost, substantially better discrimination |
| Dummy baseline (always predicts Late) | 1,774 | — | Degenerate: zero predictive utility, see below |

**Random Forest is the recommended model.** It matched the extensively tuned Decision Tree on cost (a 16-unit difference) while offering clearly superior discriminative ability (AUC 0.725 vs 0.667), which matters for threshold tuning and for ranking shipments by risk.

![ROC curves](images/roc-curves.png)

### Findings worth noting

**The baseline exposes a flaw in naive cost optimisation.** A dummy classifier that always predicts "Late" achieves the *lowest* total cost (1,774) simply because it never produces a false positive. It is nonetheless useless — it identifies no on-time deliveries and makes proactive intervention impossible. This is a case where the optimisation metric alone would select an unusable model, and it is why cost was weighed alongside AUC rather than in isolation.

**Hyperparameter tuning added almost nothing to Random Forest.** Grid searches optimised for F1, Class 0 recall, and business cost converged on parameters producing metrics identical to the sklearn defaults. The dataset's signal appears to be well captured by default ensemble parameters — a useful negative result, and one that argues against spending compute on tuning before establishing that tuning helps.

**Feature engineering did not pay off.** Derived features degraded Random Forest performance through multicollinearity with their source columns, and increased Decision Tree cost under feature selection (5,188 → 5,334). The original feature set was retained.

**Threshold tuning was worth more than model choice.** Both models identified 0.55 as the practical operating point, cutting cost by roughly 660 units while retaining ~92% of on-time delivery predictions.

### Explainability

![Random Forest feature importance](images/rf-feature-importance.png)

`Discount_offered` and `Weight_in_gms` dominate in both models, confirmed independently by EDA correlation analysis, feature importance, and SHAP values. The two models weight them differently: the Decision Tree leans heavily on `Discount_offered` (importance 0.48) as its root split, while Random Forest spreads importance more evenly with `Weight_in_gms` first (0.274), then `Discount_offered` (0.221) and `Cost_of_the_Product` (0.166) — a direct consequence of greedy splitting versus ensemble averaging.

`Warehouse_block`, `Mode_of_Shipment`, and `Gender` carried negligible importance throughout.

![SHAP summary](images/shap-random-forest.png)

Local explanation was performed on a single misclassified instance (ID 4143), which the Decision Tree predicted as On Time with 75% confidence despite an actual late outcome. Both dominant features pointed toward on-time delivery — low discount, heavy package — illustrating a systematic blind spot where unobserved delay causes are invisible to the model.

---

## Part 2 — Shipment segmentation

Scope was narrowed to high-importance electronics, the segment where delivery failures carry the greatest reputational cost.

![Elbow method](images/elbow-method.png)

K-Means and agglomerative hierarchical clustering were both applied; the elbow method identified **k = 3** as the most stable and interpretable configuration. Min-max scaling proved essential so that physical attributes (weight) and customer metrics (rating) carried equal mathematical weight.

![Cluster profiles](images/cluster-profiles.png)

| Cluster | Profile | Operational implication |
|---|---|---|
| 0 — Bulk Electronics | High weight, standard cost | Optimise physical storage and heavy-vehicle routing |
| 1 — Promotional Drivers | High discount, high prior purchases | Loyal customers being rewarded; delays here carry the highest brand-loyalty risk |
| 2 — Premium Tech | High product cost, frequent support calls | Prioritise for immediate dispatch to reduce customer anxiety |

A hold-out instance was withheld from training to demonstrate that new incoming shipments can be assigned to an existing operational cluster at order time. A what-if analysis on that instance showed its cluster assignment to be highly sensitive to `Discount_offered` — meaning a marketing decision to change a discount directly reshapes the logistics handling profile of the item.

The practical conclusion is that a uniform shipping policy is leaving value on the table: the three profiles warrant different handling.

---

## Repository structure

```
├── data/
│   └── Ecommerce.csv
├── notebooks/
│   ├── 01-delivery-delay-classification.ipynb
│   └── 02-shipment-segmentation-clustering.ipynb
├── images/
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Notebooks run top to bottom; both read `data/Ecommerce.csv` relative to the repository root.

---

## Attribution

Coursework project completed by a team of four students at Prague University of Economics and Business (VŠE), 2026.

**My contribution:** _[fill in — e.g. exploratory data analysis, Random Forest tuning and cost-sensitive evaluation, SHAP explainability, clustering interpretation]_
