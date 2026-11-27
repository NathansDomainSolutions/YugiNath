Minimal stand-ins for ygo-agent's `ygoenv` and `ygoai` packages, used only by
`tests/test_ygoenv_runner.py` to exercise the runner's game loop and
winner bookkeeping without the compiled engine. They do not play Yu-Gi-Oh:
in every stub duel the deck whose name ends in "_w" wins.
