# EV Purchase Prediction (Kaggle Playground S6E9)

Binary classification: predict `Will_Buy_EV` for synthetic consumer data.
Metric: ROC-AUC. My first Kaggle competition, built end to end.

**Final private leaderboard: 0.94325 (rank 1629)**

## Approach
1. **Baseline:** default LightGBM, single 80/20 split, 0.94090 on the leaderboard.
2. **Trustworthy validation:** stratified 5-fold CV with out-of-fold (OOF) scoring. OOF AUC tracked the public leaderboard closely throughout.
3. **Tuning:** Optuna, 60 trials, over learning rate, tree size and regularisation.
4. **Feature engineering driven by evidence:**
   - Charging-station features had near-zero correlation with the target, so I stopped adding features around them.
   - The synthetic generator over-produces certain income/commute values, so **count (frequency) features** for income, commute, age and concern level gave the biggest jump.
   - Income digit features (mod 100, mod 1000, round-number flag) added a smaller gain.

## Results
| Version | OOF AUC | Public LB |
|---|---|---|
| Baseline LightGBM | - | 0.94090 |
| 5-fold CV | 0.94123 | 0.94093 |
| + Optuna (30 trials) + interactions | 0.94203 | 0.94189 |
| + Optuna (60 trials) | 0.94226 | 0.94222 |
| + count features | 0.94335 | 0.94338 |
| + more counts + income digits (final) | 0.94400 | 0.94403 |

## What did not help
- CatBoost blend: weaker alone (0.94037), blending lowered OOF AUC.
- Target encoding on income: 0.93851 unsmoothed, 0.94315 smoothed, both below plain counts.
- Categorical combination counts and income-bucket counts: no change.

## Lessons
- Validate with OOF first; it predicted the leaderboard well.
- Check correlations and feature importance before engineering features.
- Know when a technique has plateaued and move to a different one.

## Run it
Download the data from the [competition page](https://www.kaggle.com/competitions/playground-series-s6e9) into this folder, then:
```
pip install -r requirements.txt
python train.py
```