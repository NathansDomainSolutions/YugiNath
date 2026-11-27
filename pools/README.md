# Card pools

A pool lists the cards the deck search may add. Format, one card per line:

```
<code> <main|extra> [max copies]  # card name
```

`blue_eyes.txt` holds every card from ygo-agent's tested Blue-Eyes list (main, extra
and side) plus generic staples and extra-deck monsters that appear in three or more
of ygo-agent's other tested decks. ygo-agent's README says any combination of cards
from its tested decks is supported, so every card here should run in the engine.
The banlist is applied on top, so forbidden cards in the pool are simply never added.

To widen the search, add lines. Cards outside ygo-agent's `scripts/code_list.txt`
won't load at all, and cards outside its tested decks may throw engine errors.
