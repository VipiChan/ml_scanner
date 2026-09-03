Act as a Principal Quantitative Software Engineer and Systems Architect. I am building a modular, production-grade Multi-Timeframe (MTF) Machine Learning-Driven Stock Scanner, Backtesting Engine, and Execution Pipeline in Python for the Nifty 500 universe.

**CRITICAL CONTEXT - EXISTING ASSETS:**
To accelerate development, I will be integrating logic from two of my existing, fully functional local projects:

1. **Data Ingestion Pipeline** (`C:\Users\mail2\OneDrive\projects2\scan_trade`): A completed module that successfully fetches, cleans, and stores MTF OHLCV data into my TimescaleDB instance.
2. **Feature Engineering Module** (`C:\Users\mail2\OneDrive\projects1\Final Project`): A completed module originally built to predict Nifty 50 index uptrends, which already handles technical indicator generation using `pandas_ta`.

*Note: both of these projects are in my local drive and you should be able to access these. please reuse these as much as you can. You need to copy the content from there and make it part of this project. You should not simply refer them all the times, this project has to be standalone.*

Your goal is to output a comprehensive, highly granular **PHASED MILESTONE PLAN** and **ARCHITECTURAL BLUEPRINT** detailing how to build this system step-by-step. Keep maximum milestones (not exceeding 25). I will instruct when to implement next milestone.

### System Pipeline & Adaptation Specifications

**1. Data Integration & Scaling Layer (**`data/`**)**

- **Adaptation:** Create adapter interfaces to wrap my existing TimescaleDB ingestion logic.
- **MTF Orchestration:** Efficiently query and align MTF OHLCV data for 500 tickers. The primary trading timeframe is **Hourly**. The **Daily** timeframe will be used strictly for higher-timeframe trend/regime confirmation, and the **15-Minute** timeframe for supplementary support/execution features.
- **Universe Selection:** Implement a dynamic liquidity filter (e.g., 20-day ADTV > ₹5 Crore) to reduce the Nifty 500 universe before processing.

**2. Feature Engineering & ML Architecture (**`features/`**,** `ml_engine/`**,** `execution/`**)**

- **STAGE 1: Scaling Feature Engineering (pandas_ta + Boruta):**
- **Refactoring:** Adapt my existing Nifty 50 (single-asset) feature engineering module to a multi-asset (cross-sectional/panel data) pipeline for the Nifty 500 stock universe.
- **Feature Selection:** Implement the **Boruta** feature selection algorithm, followed by strict multicollinearity tests (e.g., Variance Inflation Factor / VIF), to reduce dimensionality and retain only highly predictive features across the broader universe.
- **Target Variable:** Redefine the target from "Index Uptrend" to individual stock swing-trade targets (lasting up to ~10 days). Otherwise ignore time-based exits; focus entirely on price-action or prediction-based exits.
- **STAGE 2: Machine Learning Predictive Engine:**
- **Model Pipeline:** Train a gradient boosting model (XGBoost, LightGBM) or Random Forest using Purged Walk-Forward Cross-Validation (TimeSeriesSplit) suitable for panel data.
- **Inference Output:** The model and execution logic must collectively output actionable trading data: **Scanned Candidate Tickers, Precise Entry Prices, and Precise Exit Prices (Stop-Loss and Take-Profit).**

**3. ML-Aware Backtesting Engine (**`backtest/`**)**

- **Vectorized & Event-Driven Backtester:** Simulate historical ML-driven signals on the Hourly timeframe without look-ahead bias.
- **Friction Modeling:** Deduct accurate Indian market transaction costs (STT, Exchange Charges, Brokerage, GST) and fixed slippage (~0.03%) per round-trip trade.

**4. Interface & Reporting Layer (**`reporting/`**,** `notebooks/`**)**

- **Core API Separation:** Design the pipeline as pure Python modules (`src/`).
- **Jupyter Notebook Orchestration:** Expose orchestrator functions to generate interactive Plotly charts, ML feature importance plots (post-Boruta), and backtest equity curves.



### Expected Deliverables in Architectural Blueprint

1. **Granular Phased Milestone Plan:** A highly structured, step-by-step roadmap focusing on integration and safe scaling. Maximize the number of milestones (make them micro-milestones). I want to move slowly, step-by-step, as understanding the project deeply and maintaining strict control is my primary focus.
2. **Folder & File Structure:** Comprehensive layout showing where my *existing/legacy* modules will sit alongside the *new* modules (e.g., `config`, `data_ingestion_legacy`, `features_scaled`, `ml_engine`, `backtest`, `notebooks`).
3. **Data Flow & Interface Contracts:** Schema definitions (Pandas DataFrames) detailing how data flows from the legacy ingestion DB ➔ Scaled Feature Engineering ➔ ML Inference ➔ Backtester. Focus explicitly on how the index-based (single-asset) features transition to cross-sectional (multi-asset) DataFrames.
4. **Class Diagrams & Function Signatures:** High-level object-oriented specification of key classes, emphasizing the adapters needed for the existing code (e.g., `TimescaleAdapter`, `PanelFeatureEngineer`, `BorutaSelector`, `MLEstimator`, `Backtester`).
5. `user_command.md` **Specification:** Provide the exact contents of a `user_command.md` file. For every micro-milestone defined in Deliverable 1, this file must list the exact CLI command or Python script execution required to test and validate the accurate completion of that specific milestone.
6. `user_command.ipynb` **Specification:** Provide the exact contents of a `user_command.ipynb` file. For every micro-milestone defined in Deliverable 1, this file must list the sequetial guideline and code and equivalent exact CLI command or Python script execution required to test and validate the accurate completion of that specific milestone.

