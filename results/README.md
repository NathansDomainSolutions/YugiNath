Written by `python -m yuginath run`:

- `history.csv`: one row per round (best candidate, change, acceptance win rate, 95% CI, accepted or not).
- `rounds/<run id>-rNN.json`: the full record of a round: every candidate's card change,
  each successive-halving stage, the acceptance match split by going first and second,
  and the champion's list afterwards.

Raw per-game logs are in `runs/duels.sqlite`, which stays out of git.
