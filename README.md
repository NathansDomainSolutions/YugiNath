# YugiNath

A Yu-Gi-Oh! bot that learns by playing itself and rebuilds its own deck.

This first milestone is the whole loop working end to end, not a strong bot. One
command runs as many rounds as you ask for. Each round does three things:

1. **Train:** the player trains by self-play (PPO with an LSTM, using [ygo-agent]'s
   trainer) on the current champion deck, continuing from its last checkpoint.
2. **Mutate:** it makes 8 variants of the champion by swapping 1 to 3 cards for
   others from a card pool. Every variant has to pass the deck rules and the banlist.
3. **Score:** the variants duel the champion in the real rules engine. The better
   half survives and plays more games until one is left (successive halving). That
   one then plays 400 fresh games against the champion and replaces it only if it
   wins 55% or more.

The starting deck is ygo-agent's tested Blue-Eyes list.

## How the bot plays

The bot doesn't drive a game client. It runs its own copy of the rules engine,
ygopro-core (the engine behind YGOPro), through ygo-agent's `ygoenv`, with no
screen. That's how it can play thousands of duels quickly. Playing people in
EDOPro or YGOPro is a later step.

## Setup

You need Ubuntu 22.04 or newer (on Windows, use WSL2), git, curl, make and
Python 3.10, 3.11 or 3.12. A consumer NVIDIA GPU is enough for training.
ygo-agent's author trained on a GTX 1650 laptop.

```bash
git clone https://github.com/nathansdomainsolutions/yuginath.git
cd yuginath
CUDA=1 PYTHON=python3.12 scripts/setup_ygo_agent.sh   # leave out CUDA=1 without a GPU
```

The setup script:

- makes a virtual environment in `external/venv`
- installs JAX 0.4.28 or older and the libraries ygo-agent's trainer uses
- clones ygo-agent into `external/ygo-agent` at a pinned commit
- downloads its prebuilt engine for your Python version
- runs its `make`, which downloads the card database and card scripts
- plays 32 random duels to check the engine works

If the prebuilt engine fails with a `GLIBCXX` error, see the troubleshooting
section of ygo-agent's README.

## Running it

From the repo root:

```bash
# 1. Plumbing test with no engine (coin flips instead of duels). Runs anywhere.
python3 -m yuginath --config config/smoke-fake.toml run

# 2. Tiny real run: real duels on CPU and one PPO update. Takes minutes.
external/venv/bin/python -m yuginath --config config/smoke.toml run

# 3. The real thing: 3 rounds, about 2,000 duels and 50 PPO updates each.
external/venv/bin/python -m yuginath run
external/venv/bin/python -m yuginath run --rounds 10   # more rounds
```

Running it again carries on: the next run starts from the saved champion and the
last checkpoint, with fresh random seeds.

The pieces also work on their own:

```bash
python3 -m yuginath check decks/champion.ydk               # what rules a deck breaks
python3 -m yuginath repair my_deck.ydk my_deck_legal.ydk   # make a deck legal
python3 -m yuginath mutate decks/champion.ydk --n 4        # 4 legal variants in runs/variants/
external/venv/bin/python -m yuginath evaluate decks/champion.ydk runs/variants/*.ydk --games 200
```

`evaluate` plays the first deck against each of the others, half the games going
first, and prints a win-rate table with 95% confidence intervals. Without
`--checkpoint`, both sides play random legal moves.

### Settings

`config/default.toml` holds everything you're likely to change:

- **Banlist:** `rules.banlist_name`. The default is the 2026.10 TCG list; `"2026.10 OCG"` is also included.
- **Card pool:** `pool.file`. See `pools/README.md`.
- **Search size:** `mutation.candidates`, `scoring.halving_start_games`, `scoring.accept_games` and `scoring.accept_threshold`.
- **Training per round:** `train.iterations`, plus the GPU ids.

Defaults for anything not set live in `yuginath/config.py`.

## Where data lives

| What | Where | In git? |
|---|---|---|
| Code | `yuginath/`, `scripts/`, `tests/` | Yes |
| Current champion deck | `decks/champion.ydk` | Yes |
| Every accepted champion | `decks/history/` | Yes |
| Starting deck, card pool, banlists | `decks/start/`, `pools/`, `banlists/` | Yes |
| Round results | `results/history.csv`, `results/rounds/*.json` | Yes |
| Model checkpoints (`.flax_model`) | `runs/checkpoints/` | No |
| Every duel played (SQLite) | `runs/duels.sqlite` | No |
| Duel jobs, temp decks, loop state | `runs/` | No |
| ygo-agent, the engine, card scripts | `external/` | No |

The loop doesn't commit for you. After a run, commit `decks/` and `results/` to
keep the history.

Checkpoints are large binary files and the duel log grows by about 2,000 rows a
round, so `.gitignore` keeps them out of normal git. To keep a good checkpoint:

- **GitHub releases (simplest):** create a release and attach the `.flax_model`
  file. Each file can be up to 2 GB, and releases don't count against the repo's size.
- **Git LFS:** run `git lfs install`, then `git lfs track "*.flax_model"`, and
  commit `.gitattributes`. Then move the files you want out of `runs/` and commit
  them. GitHub's free LFS storage is limited, so keep only the checkpoints worth keeping.

## Licences

| Piece | Licence | How this repo uses it |
|---|---|---|
| [ygo-agent] (ygoenv, ygoai, trainer) | MIT (some envpool parts Apache-2.0) | Downloaded into `external/` at setup, not copied in. The starting Blue-Eyes list comes from its `assets/deck/`. |
| [ygopro-core, Fluorohydride](https://github.com/Fluorohydride/ygopro-core) | MIT | The engine inside ygo-agent's default `YGOPro-v1` environment |
| [ygopro-scripts, mycard](https://github.com/mycard/ygopro-scripts) | GPL-2.0 | Card scripts the engine runs; downloaded by ygo-agent's `make` |
| [ygopro-core, Project Ignis/EDOPro](https://github.com/edo9300/ygopro-core) and [CardScripts](https://github.com/ProjectIgnis/CardScripts) | AGPL-3.0 | Only used if you switch to ygo-agent's `EDOPro` environment. Not used by default. |
| Banlist data | From the [YGOPro client](https://github.com/Fluorohydride/ygopro)'s `lflist.conf` (GPL-2.0 repo) | Card codes and limits only |

What that means for you:

- The default setup uses MIT-licensed code plus GPL-2.0 card scripts. Nothing GPL
  or AGPL is copied into this repo; it's downloaded at setup.
- **If you share the bot with the card scripts bundled in** (for example, a
  download with everything included), the GPL-2.0 terms apply to that bundle:
  include the source and the licence.
- **If you switch to the EDOPro engine and run the bot as a service other people
  can play against over the internet**, AGPL-3.0 requires you to offer them your
  source code, changes included.
- This repo has no licence file yet. Without one, others can see the code but may
  not reuse it. Pick one when you're ready: MIT for permissive, AGPL-3.0 to match
  EDOPro's copyleft.

Yu-Gi-Oh! card names and the game belong to Konami. This is a fan research
project.

## What's tested and what isn't

- **Tested here:** `python3 -m pytest` runs 17 tests. They cover:
  - deck files, banlist parsing and deck rules
  - 500 random mutations, all checked legal
  - successive halving and seat-balanced scoring
  - a full three-round loop on the fake backend
  - the real ygoenv runner, driven by stand-in `ygoenv`/`ygoai` modules (`tests/stubs/`)
- **Not run yet:** real duels and real training through ygo-agent. The runner
  follows ygo-agent's `scripts/battle.py`, and training calls its
  `scripts/cleanba.py`. Neither has run against the real engine in this repo yet.
  `config/smoke.toml` is the first thing to run after setup.

Known gaps for the next steps:

- The duel log has the winner, who went first, game length and the deck in each
  seat. It doesn't have the cards drawn and used yet. ygoenv doesn't report those
  per game, so the per-card stats in the plan need a change on the engine side.
- There's no opening-hand pre-filter yet. Every candidate goes straight to duels.
- The same frozen player pilots both decks in every scoring match, and the deck
  isn't an input to the network yet. A deck can look worse just because the player
  hasn't learned it. That's why each round trains on the champion first.

[ygo-agent]: https://github.com/sbl1996/ygo-agent
