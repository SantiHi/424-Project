# Do unique players pay off?

COS 424 course project. We look back at NFL draft classes since 2000 and ask whether prospects whose
combine measurements are unusual for their position return more than their draft slot normally does.

## Repository layout

| path | contents |
|---|---|
| `initial_analysis.ipynb` | Milestone 2 notebook: data, preprocessing, descriptive statistics, exploration, first model |
| `src/prepare.py` | loads the raw files and builds the cleaned table of prospects |
| `src/summaries.py` | descriptive tables used by the notebook |
| `src/plots.py` | charts used by the notebook |
| `src/detect.py` | first version of the detection model: Mahalanobis distance on size and speed, with Isolation Forest and Local Outlier Factor kept for comparison |
| `data/raw/` | the three nflverse files, as downloaded |
| `data/processed/prospects.csv` | cleaned table written by `src/prepare.py` |
| `NFL_Unique_Players_Intro__1_.pdf` | project introduction slides |
| `unicorn_replication_package/`, `all_unicorns.csv` | earlier end-to-end prototype and its output; not used by the notebook |

## Reproducing the analysis

Developed with Python 3.12.

```
pip install -r requirements.txt
python -m src.prepare     # optional: rebuilds data/processed/prospects.csv and prints the cleaning log
```

Then open `initial_analysis.ipynb` and run every cell from the top.

## Data

All three files come from the open [nflverse](https://github.com/nflverse/nflverse-data) project,
downloaded in October 2026:

- `combine.csv`: one row per NFL Combine participant, 2000-2026 (size, six drills, draft slot)
- `draft_picks.csv`: one row per draft pick since 1980, with career Approximate Value
- `players.csv`: NFL roster records, used for the position a player held in the NFL
