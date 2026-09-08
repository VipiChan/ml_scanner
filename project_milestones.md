# ml_scan — Project Milestones

## What this project is

**ml_scan** is a stock research system for India’s Nifty 500 universe. It reads price history that already lives in your existing market database, studies many stocks at once across hourly, daily, and 15-minute charts, trains a machine-learning model to spot promising swing trades, and then:

1. **Scans** for candidate stocks with clear entry, stop-loss, and take-profit prices  
2. **Backtests** those ideas on past data, after realistic Indian trading costs  
3. **Reports** the results in a form a person can review  

The work is split into **25 small milestones**. Each one unlocks a single, checkable capability. You can follow the project in order without needing programming knowledge.

> **Note on the training universe (current setup):** milestones 5, 6, and 25 describe optional local-universe tooling (`universe snapshot`, `universe liquid`, `e2e liquid`) that this repo still supports, but the actual training symbol list is now owned by a separate project (`scan_trade`) and frozen into `model_training_symbol.csv`, refreshed there whenever the model is retrained. Day-to-day feature building, training, and scanning (milestones 9 onward) point at that file instead of the local Nifty 500 / liquidity CSVs. See `user_command.md` for the exact commands in use.

---

## How to read each milestone

For every milestone you will see:

- **What we are building** — the concrete piece of work  
- **Why it matters** — what it enables for the overall product  
- **How we know it is done** — the plain-English success check  

Later milestones depend on earlier ones. Skipping ahead usually breaks the chain.

---

## Phase A — Foundations (can we talk to the data?)

### Milestone 1 — Project setup and settings

**What we are building**  
The basic project shell and a single place where connection settings live (which database to use, passwords from a local secrets file, and related options).

**Why it matters**  
Nothing else can run reliably until the project installs cleanly and knows which warehouse to read. We deliberately reuse your existing Timescale database named `scan_trade` rather than inventing a second copy of market data.

**How we know it is done**  
The project loads its settings successfully and confirms that the configured database name is `scan_trade`.

---

### Milestone 2 — Confirm the price warehouse is reachable

**What we are building**  
A safe connection path into the existing hourly price table so this project can count and read bars that were already collected by your earlier data pipeline.

**Why it matters**  
This project is a *reader* of market history, not a new downloader. If hourly rows are missing or the connection fails, every later step is guessing in the dark.

**How we know it is done**  
A live check against the hourly price table returns a row count greater than zero.

---

### Milestone 3 — Indian market session clock

**What we are building**  
A calendar that understands NSE cash-market trading hours in India time, including how many hourly bars belong to a normal session day.

**Why it matters**  
Stock models must not treat nights, weekends, or holidays as trading time. The session clock keeps features, labels, and backtests aligned with real market hours.

**How we know it is done**  
For a normal trading day, the calendar produces exactly **seven** hourly bars.

---

### Milestone 4 — Clean price reader for any stock and timeframe

**What we are building**  
A standard “ask for prices” interface: give it symbols, a timeframe (hourly, daily, or 15-minute), and a date range, and it returns tidy open–high–low–close–volume data.

**Why it matters**  
Later modules should never invent their own database queries. One shared reader keeps columns consistent and avoids half-formed bars (for example incomplete end-of-day hours) that would distort indicators and trades.

**How we know it is done**  
Reading a month of RELIANCE hourly data returns the expected price columns, marks the interval correctly as hourly, and excludes incomplete late-session bars.

---

## Phase B — Which stocks are we allowed to trade?

### Milestone 5 — Bring the Nifty 500 list into this project *(optional; not part of the current training path — see note above)*

**What we are building**  
A local snapshot of the Nifty 500 symbol map copied into this project’s own data folder.

**Why it matters**  
Scans and filters need a stable universe file inside *this* project. Depending forever on another folder or a live refresh makes results hard to reproduce.

**How we know it is done**  
A Nifty 500 mapping file exists locally with roughly **495** mapped symbols.

---

### Milestone 6 — Keep only liquid names *(optional; not part of the current training path — see note above)*

**What we are building**  
A liquidity screen that keeps stocks whose recent average daily traded value is high enough (target: above ₹5 crore), plus a tiny “smoke” list of a few symbols for fast trial runs.

**Why it matters**  
Illiquid names look attractive on paper and fail in practice (wide spreads, poor fills). Filtering early saves compute and keeps the strategy closer to what you could actually trade.

**How we know it is done**  
A liquid-universe file of passing stocks is written, and a small smoke-symbol file (about eight names) is written for quick testing.

---

## Phase C — Align charts and build features

### Milestone 7 — Align hourly, daily, and 15-minute history honestly

**What we are building**  
A joining step that attaches daily and 15-minute information onto each hourly bar **without using information that was not yet known at that hour**.

**Why it matters**  
If today’s unfinished daily close leaks into morning hours, the model “cheats” with future knowledge. Honest alignment is the difference between a research toy and a trustworthy system.

**How we know it is done**  
A sample aligned dataset is produced, and automated checks prove that same-session daily closes do not leak into earlier hourly bars.

---

### Milestone 8 — Match proven technical-indicator maths

**What we are building**  
Technical indicator calculations that match the helper already proven in your earlier feature-engineering project, instead of a loose rewrite.

**Why it matters**  
Indicators are the raw ingredients of the model. If RSI, ATR, and similar measures silently disagree with the version you already trust, every later metric is suspect.

**How we know it is done**  
Comparison tests pass against the frozen helper on a fixed sample of hourly prices.

---

### Milestone 9 — Features for many stocks at once

**What we are building**  
A panel feature engine that computes indicators **stock by stock** across the universe, producing one organised table keyed by symbol and timestamp.

**Why it matters**  
Your older module was built for a single index. Nifty 500 work needs the same idea scaled to hundreds of names, without one stock’s missing data contaminating another.

**How we know it is done**  
An hourly feature file exists with technical columns present, organised by symbol and time, and without cross-symbol contamination from missing values.

---

### Milestone 10 — Add daily trend and 15-minute context

**What we are building**  
Extra columns on the hourly panel: daily-derived signals (higher-timeframe trend or regime) and 15-minute-derived signals (finer support for timing and execution).

**Why it matters**  
The trading decision is hourly, but context comes from slower and faster charts. Daily columns confirm the broader picture; 15-minute columns refine short-term structure.

**How we know it is done**  
The multi-timeframe feature file includes clearly named daily columns and 15-minute columns.

---

### Milestone 11 — Define what a “successful trade” means for training

**What we are building**  
Labels for each candidate setup: within roughly ten trading sessions, did price reach the planned take-profit **before** the stop-loss? Time simply running out is not counted as a win.

**Why it matters**  
The model learns from examples of success and failure. Vague labels (“the index went up”) do not match individual stock swing trades with explicit exits.

**How we know it is done**  
A labeled dataset exists with a clear yes/no outcome for each row, and a summary of why outcomes were marked (for example take-profit hit versus stop-loss hit).

---

### Milestone 12 — Quality check and anti-cheating review

**What we are building**  
A quality report on the feature table, plus tests that catch columns which accidentally look into the future.

**Why it matters**  
Garbage or leaky features can produce beautiful backtests that fail in live markets. This milestone is a gate before expensive model training.

**How we know it is done**  
A quality report file is written, and leakage tests pass: honest features are allowed, and planted future prices are flagged.

---

## Phase D — Choose predictors and train the model

### Milestone 13 — Automatic shortlist of useful features (Boruta)

**What we are building**  
An automated filter that keeps only features that appear more informative than random noise, cutting a large technical table down to a shorter shortlist.

**Why it matters**  
Hundreds of overlapping indicators confuse models and slow everything down. A shorter list is easier to understand, faster to train, and often more stable.

**How we know it is done**  
A feature-list file is produced whose length is much smaller than the original raw feature count.

---

### Milestone 14 — Remove redundant features (VIF)

**What we are building**  
A second filter that drops predictors that are nearly duplicates of each other (high multicollinearity), using a variance-inflation style rule.

**Why it matters**  
If ten columns all say “trend up” in slightly different words, the model looks sophisticated while learning one idea many times. Cleaning redundancy improves clarity and stability.

**How we know it is done**  
A final selected-features file is produced from the Boruta shortlist under a sensible redundancy limit.

---

### Milestone 15 — Honest time-based practice exams for the model

**What we are building**  
A walk-forward splitting method that trains on earlier periods and tests on later periods, with a buffer so overlapping label windows do not contaminate the exam.

**Why it matters**  
Randomly mixing past and future rows invents fake skill. Purged, time-ordered folds answer: “Would this have worked on periods the model had not trained on?”

**How we know it is done**  
Automated tests confirm that train and test periods do not overlap under the embargo rules.

---

### Milestone 16 — Train the predictive model

**What we are building**  
The actual machine-learning model (a modern tree-based learner) trained on the selected features, with fold-by-fold quality scores saved beside the model file.

**Why it matters**  
This is the brain of the scanner. Without a saved model and transparent fold metrics, you cannot trust or reject the system with evidence.

**How we know it is done**  
A model file exists, and a companion metrics summary for the training folds is written next to it.

---

## Phase E — Turn predictions into trade ideas

### Milestone 17 — Live-style scan output

**What we are building**  
A scan that scores symbols as of a chosen time and writes an actionable table: symbol, timestamp, model score, entry price, stop-loss price, and take-profit price.

**Why it matters**  
A model score alone is not a trade. The product must emit levels you can place or simulate.

**How we know it is done**  
A scan file is produced with columns for symbol, as-of time, score, entry, stop-loss, and take-profit.

---

## Phase F — Realistic backtesting and reporting

### Milestone 18 — Indian transaction-cost model

**What we are building**  
A cost calculator for CNC-style round trips that includes statutory charges (such as STT on both sides and stamp duty on buy), other market fees, taxes where applicable, and a small slippage allowance (about 0.03% per side).

**Why it matters**  
Ignoring costs turns mediocre strategies into “winners.” Indian cash-market friction is material for swing trading and must be in the simulation from day one.

**How we know it is done**  
Automated checks confirm a round-trip cost breakdown matches the intended CNC rules and slippage assumption.

---

### Milestone 19 — Next-bar fill realism

**What we are building**  
An execution rule: a signal seen on one hourly bar is assumed filled at the **next** bar’s open, not at the same bar’s close you already used to decide.

**Why it matters**  
Filling on the signal bar is a common form of look-ahead fantasy. Next-open fills are stricter and closer to how a human or algo would act after seeing a completed bar.

**How we know it is done**  
Automated checks confirm the fill time is the following open, not the signal timestamp itself.

---

### Milestone 20 — Historical trade simulation

**What we are building**  
A backtester that replays historical scan signals on hourly bars, applies stops and targets, subtracts costs, and records every trade plus the equity path over time.

**Why it matters**  
This answers the business question: “If we had followed the system’s rules in the past, what would the book of trades and the account curve have looked like?”

**How we know it is done**  
A backtest run folder contains a trades file and an equity file.

---

### Milestone 21 — Performance summary numbers

**What we are building**  
A short metrics pack on each backtest run: growth rate (CAGR), risk-adjusted return (Sharpe), worst peak-to-trough loss (maximum drawdown), and profit factor.

**Why it matters**  
Raw trade lists are hard to compare. A few standard numbers let you accept, reject, or compare runs quickly.

**How we know it is done**  
Those summary metrics print successfully for a completed backtest run.

---

### Milestone 22 — Human-readable HTML report

**What we are building**  
An HTML report generated from a backtest run, suitable for opening in a browser, including equity-curve style views (and related charts as the reporting layer grows).

**Why it matters**  
Stakeholders who will never open code still need a clean artefact to review. Reports turn research into a shareable decision package.

**How we know it is done**  
An HTML report file is written from a chosen backtest run directory.

---

## Phase G — Product surface and full-path proof

### Milestone 23 — One command-line product surface

**What we are building**  
A single command-line entry point that exposes the main jobs of the system: universe management, feature building, training, scanning, backtesting, reporting, and end-to-end runs.

**Why it matters**  
A pile of scripts is hard to operate. One coherent product surface is how you run the system day to day without hunting through notebooks.

**How we know it is done**  
Asking the tool for help lists the expected command groups: coverage, universe, data, features, ml (compare/boruta/vif/optimize/train), scan, backtest, report, and e2e.

---

### Milestone 24 — End-to-end smoke run (small universe)

**What we are building**  
A single “run everything” path on the tiny smoke symbol list: features → labels → feature selection → train → scan → backtest → report.

**Why it matters**  
Individual pieces can work while the full chain is broken. The smoke run is the first proof that the product path holds together from data to report.

**How we know it is done**  
The end-to-end smoke command finishes successfully and writes artefacts under a dedicated smoke output folder.

---

### Milestone 25 — Larger liquid-universe stress run *(optional scale-up rehearsal; the production model is trained on the full external universe, not this capped subset)*

**What we are building**  
The same full pipeline on a larger liquid subset (capped, for example around eighty symbols), closer to real operating breadth than the tiny smoke list.

**Why it matters**  
Systems often work on eight names and fail on eighty. This milestone is the scale-up rehearsal before treating the scanner as production-ready research infrastructure.

**How we know it is done**  
Scan and backtest artefacts are produced for the liquid end-to-end path, and the official acceptance command list still matches how the product is meant to be run.

---

## Suggested reading order for a non-technical review

1. Read **Phase A–B** to understand data and universe.  
2. Read **Phase C–D** to understand how the model is taught honestly.  
3. Read **Phase E–F** to understand trade ideas, costs, and historical proof.  
4. Read **Phase G** to see how the whole product is operated and verified.

For the exact technical acceptance commands that prove each milestone, see `user_command.md`. For the same checks as notebook cells, see `user_command.ipynb`.
