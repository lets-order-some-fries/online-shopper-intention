# Purchase intent from browsing sessions

Predicts whether a browsing session on a retail site ends in a purchase, using the UCI Online Shoppers Purchasing Intention dataset: 12,330 sessions, 15.5% of them converting.

Only 15.5% of sessions buy, so answering "no purchase" every time is already 84.5% accurate. The notebook therefore scores models on the purchases they find.

| | test set (2,466 sessions) |
|---|---|
| PR-AUC, random forest | 0.73 (random ranking: 0.155) |
| purchases found at the chosen cut-off | 70% (266 of 382) |
| flagged sessions that buy | 63% (266 of 419) |
| accuracy | 89.1% (always "no purchase": 84.5%) |

Most of the signal comes from one feature. PageValues is a Google Analytics metric computed from completed transactions, and it is highest on pages near checkout. Without it, cross-validated PR-AUC falls from 0.742 to 0.386. A model used early in a session should expect the lower figure.

## Contents

- `purchase_intent.ipynb`: the analysis, written in 2026. It uses a stratified split made before any preprocessing, with encoding and scaling inside the cross-validated pipelines. It compares logistic regression, a random forest and gradient boosting on five-fold PR-AUC, picks a threshold on out-of-fold predictions, evaluates once on the held-out test set, and ends with the PageValues ablation.
- `Machine Learning Final Report.pdf`: the original 2023 course report (UC Irvine MSBA '24, Team 13B), which tuned a logistic regression to 87.9% test accuracy.
- `online_shoppers_intention.csv`: the dataset.

## Running it

```
pip install -r requirements.txt
jupyter notebook purchase_intent.ipynb
```

Results are deterministic (seed 42). Tested with scikit-learn 1.9 and pandas 3.1.

## Credits

- Dataset: Sakar, C. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset.* UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q (CC BY 4.0).
- The 2023 report is the work of Ambuj Upadhyay, Akhil Tangutur, Ping-Yen Chung, Yu-Fang Chang and Ziyue Wang. Its preprocessing followed Ben Roshan's public Kaggle notebook, [Online Shopper's Intention | Marketing Analytics](https://www.kaggle.com/code/benroshan/online-shopper-s-intention-marketing-analytics).
