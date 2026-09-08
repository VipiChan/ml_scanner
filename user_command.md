# ml_scan — milestone acceptance commands

Prereqs (once): existing Timescale `scan_trade` on localhost:5433; copy `.env.example` → `.env` and set `TIMESCALE_PASSWORD`.

```powershell
python -m venv .ml_env
.\.ml_env\Scripts\Activate.ps1
pip install -e ".[dev]"
copy .env.example .env
```

## M01
python -c "import ml_scan, ml_scan.config; print(ml_scan.config.load_settings().timescale.db)"
# expect: scan_trade

## M02
python -c "from ml_scan.storage.db import Database; from ml_scan.config import load_settings; db=Database.from_settings(load_settings());
import pandas as pd
with db.connection() as c:
    n=c.execute('select count(*) as n from ohlcv_60m').fetchone()['n']
print('hourly_rows', n)"
# expect: n > 0

## M03
python -c "import pandas as pd; from ml_scan.data.calendar import NSECalendar; c=NSECalendar(); print(len(c.hourly_index(pd.Timestamp('2024-01-02', tz='Asia/Kolkata'))))"
# expect: 7

## M04
python -c "from ml_scan.data.timescale_adapter import TimescaleAdapter; from ml_scan.config import load_settings; a=TimescaleAdapter(load_settings()); df=a.read(['RELIANCE'], '60minute', '2025-09-01', '2026-09-03'); print(df.columns.tolist(), len(df), df['interval'].unique())"
# expect: standard OHLCV columns, interval == 60minute, no 15:15 partial hours; use warehouse coverage if 2024 is empty

## Training universe
# Symbol list is owned by scan_trade (`model_training_symbol.csv`). Set this path once;
# M09+ feature build and M17 scan use it. M05 (Nifty snapshot), M06 (ADTV liquid/smoke),
# and M25 (liquid e2e) are skipped — not part of the training path.
$UNI = "C:\Users\mail2\OneDrive\projects2\scan_trade\data\universe\model_training_symbol.csv"

## M07
python -m ml_scan.cli data align --symbols RELIANCE,TCS --start 2025-09-01 --end 2026-09-03 --out data/artifacts/align_smoke.parquet
python -m pytest tests/test_mtf_aligner.py -q
# expect: tests prove no same-session daily leak

## M08
python -m pytest tests/test_ta_engine_parity.py -q
# expect: pass vs copied helper on a fixture hourly CSV

## M09
python -m ml_scan.cli features hourly --universe $UNI --start 2025-09-01 --end 2026-09-03 --out data/artifacts/feat_hourly_smoke.parquet
# expect: MultiIndex (symbol, ts); full pandas_ta matrix (200+ indicator columns); no cross-symbol NaN bleed

## M10
python -m ml_scan.cli features mtf --in data/artifacts/feat_hourly_smoke.parquet --out data/artifacts/feat_mtf_smoke.parquet
# expect: columns starting with d_ and m15_

## M11
python -m ml_scan.cli features label --in data/artifacts/feat_mtf_smoke.parquet --out data/artifacts/labeled_smoke.parquet
# expect: y in {0,1}; y_reason counts printed; class is imbalanced — inverse-frequency weights (cwts) are applied from M11b / training onward

## M12
python -m ml_scan.cli features qc --in data/artifacts/labeled_smoke.parquet --out data/artifacts/feature_qc_report.json
python -m pytest tests/test_leakage.py -q

## M13
python -m ml_scan.cli ml boruta --in data/artifacts/labeled_smoke.parquet --out data/artifacts/boruta_features.json --max-iter 50
# expect: JSON list length << raw feature count

## M14
python -m ml_scan.cli ml vif --in data/artifacts/labeled_smoke.parquet --features data/artifacts/boruta_features.json --out data/artifacts/selected_features.json --max-vif 10

## M15
python -m pytest tests/test_purged_split.py -q

## M16
python -m ml_scan.cli ml train --in data/artifacts/labeled_smoke.parquet --features data/artifacts/selected_features.json --out data/artifacts/model.joblib
# expect: fold metrics JSON beside model

## M17
python -m ml_scan.cli scan --asof latest --universe $UNI --out data/artifacts/scan_latest.csv
# expect: columns symbol,asof_ts,score,entry_px,sl_px,tp_px

## M18
python -m pytest tests/test_costs.py -q
# expect: CNC round-trip includes STT both sides, stamp on buy, 0.03% slip/side

## M19
python -m pytest tests/test_fill_lag.py -q

## M20
python -m ml_scan.cli backtest run --signals data/artifacts/scan_history.parquet --start 2025-09-01 --end 2026-09-03 --out data/artifacts/bt_smoke
# expect: trades.parquet + equity.parquet

## M21
python -m ml_scan.cli backtest metrics --run data/artifacts/bt_smoke
# expect: CAGR, Sharpe, MaxDD, profit factor printed

## M22
python -m ml_scan.cli report --run data/artifacts/bt_smoke --out data/artifacts/report_smoke.html

## M23
python -m ml_scan.cli --help
# expect: coverage, universe, features, train, scan, backtest, report

## M24
python -m ml_scan.cli e2e smoke
# expect: exit 0; artifacts under data/artifacts/e2e_smoke/

## Diagnostics — accuracy investigation (see user_command.ipynb M09b/M12b/M13b/M14b/M15b/M16a/M16b)

These commands back the extra notebook cells added to answer "why is accuracy poor": full metrics
(accuracy / balanced accuracy / precision / recall / ROC-AUC), which stocks/folds feed training,
raw pandas_ta feature counts, a same-folds model bake-off, Boruta on the winning model, and a
purged-fold hyperparameter search.

python -m ml_scan.cli ml compare --in data/artifacts/labeled_smoke.parquet --out data/artifacts/model_comparison_initial.json
# expect: accuracy/balanced_accuracy/precision/recall/roc_auc for rf, xgboost, lightgbm on every QC'd feature

python -m ml_scan.cli ml boruta --in data/artifacts/labeled_smoke.parquet --out data/artifacts/boruta_features_best_model.json --estimator-name lightgbm
# expect: same as M13 but the shadow-feature estimator matches the ml-compare winner, not the fixed default

python -m ml_scan.cli ml optimize --in data/artifacts/labeled_smoke.parquet --features data/artifacts/selected_features.json --model lightgbm --out data/artifacts/best_params.json --n-iter 20
# expect: best_params.json with a RandomizedSearchCV winner scored on the purged walk-forward folds (no leakage)

python -m ml_scan.cli ml train --in data/artifacts/labeled_smoke.parquet --features data/artifacts/selected_features.json --out data/artifacts/model.joblib --model lightgbm --params data/artifacts/best_params.json
# expect: same as M16 but retrained with the tuned hyperparameters from ml optimize
