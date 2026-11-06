# Pricing the Unicorn: replication package

Rebuild everything with `./run_all.sh` (Python 3; needs pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, adjustText).

- `data/`: the three nflverse CSVs used (combine, draft_picks, players), as downloaded October 2026
- `code/01_prepare.py` to `06_tables.py`: data prep, detection, valuation, statistics, figures, LaTeX tables
- `out/all_unicorns.csv`: all 390 flagged players with measurements, scores, type, draft slot, value and grade
- `out/master.csv`: every prospect in the modelling sample with all scores and grades
- `out/teams.csv`, `out/results.json`: team table and every statistic quoted in the paper
- `paper/`: the LaTeX project (also shipped separately as the Overleaf zip)

Overleaf: New Project > Upload Project > choose `unicorn_paper_overleaf.zip`. Compiler: pdfLaTeX. Add your name on the marked line near the top of `main.tex`.
