# Game Theory Lab: Strict and Weak Dominance

A Python program that takes a finite game in normal form (any number of players, any number of strategies per player), displays it, and then simplifies it by **iterated elimination of dominated strategies**, step by step. Only pure strategies are considered.

## 1. Files

| File | Purpose |
|---|---|
| `dominance.py` | the program |
| `power_game_3players.json` | our own 3-player game (described in section 5) |
| `README.md` | this document |

## 2. How to run

Requirements: Python 3 and `numpy` (`pip install numpy`).

```bash
python dominance.py                                              # interactive menu
python dominance.py --test                                       # tests on examples 1 to 5 from TP1 pdf
python dominance.py --file power_game_3players.json --mode strict   # our game, strict dominance
python dominance.py --file power_game_3players.json --mode weak     # weak dominance
python dominance.py --file power_game_3players.json --mode weak --random --seed 5   # weak, random elimination order
```

The interactive menu offers four sources for the game: **keyboard**, **JSON file**, **random generation**, or a **lab sheet example** (1 to 5). The user then chooses the mode (strict or weak) and, in weak mode, the elimination order (**default** or **random**).

## 3. How the code works

### 3.1 Representation of the game

The game is stored in **a single numpy array with n+1 dimensions**:

```
game[s1, s2, ..., sn, i] = payoff of player i when player 1 plays s1, player 2 plays s2, ...
```

- The first n axes are the strategies of each player (their sizes kᵢ can differ).
- The last axis holds the payoff vector.
- So the number of players is `game.ndim - 1` (function `num_players`). No special case is needed for 2, 3 or n players.

Strategy names are stored in `names[i][j]` (strategy j of player i).

**Key idea:** we never delete rows or columns from the array. Instead we keep a list `active[i]` of the strategy indices still present for each player. Removing a strategy means removing its index from `active`. This way indices never change and the trace can always print the original names.

### 3.2 Dominance test (`payoffs_of`, `dominates`)

- `payoffs_of(game, active, i, s)` returns the payoffs of player i when playing s, against **all remaining combinations of strategies of the other players** (using `np.ix_`). With 3 players this gives one payoff per combination of the two other players.
- `dominates(game, active, i, b, a, mode)` compares the payoffs of b and a:
  - **strict**: `payoffs(b) > payoffs(a)` for every combination of the other players;
  - **weak**: `payoffs(b) >= payoffs(a)` everywhere, with `>` **at least once**.
- Two strategies with identical payoffs do not dominate each other (a strict inequality is required at least once).

### 3.3 Iterated elimination (`find_dominated`, `iterated_elimination`)

1. `find_dominated` goes through the players (in the chosen order), then through their active strategies, and returns the first dominated strategy found together with its dominating strategy.
2. `iterated_elimination` removes it (one at a time), prints the step (player, removed strategy, dominating strategy, remaining game), then starts again.
3. The loop stops when nothing more can be removed.
4. The final result says whether **a single profile** remains (game solved) or not.

With strict dominance, the order of elimination does not matter. With weak dominance it can change the result, which is why the program **prints the order followed** and offers a **random** order (`--random`) as an alternative to the default one. The seed is printed, so a run can be repeated with `--seed`.

### 3.4 Display (`print_game`, `print_table`)

- **2 players**: one matrix (player 1 in rows, player 2 in columns), each cell holds the payoff vector.
- **3 players or more**: one matrix for each combination of strategies of players 3 to n, labelled for example `[P3 = Medium]`.
- Only the strategies still active are shown, so the game visibly shrinks at each step.

### 3.5 Game input (R1)

- `input_keyboard`: number of players, number of strategies of each, names (optional), then the payoffs profile by profile.
- `load_json` / `input_file`: reads a JSON file.
- `input_random`: random integer payoffs between a minimum and a maximum.

**JSON file format:** `payoffs[s1][s2]...[sn]` is the payoff vector of the profile (s1, ..., sn), so `payoffs` has shape (k1, ..., kn, n). Example with 2 players:

```json
{
  "names": [["H", "B"], ["G", "M"]],
  "payoffs": [[[2, 1], [2, 1]], [[2, 1], [1, 0]]]
}
```

## 4. Results on the lab sheet test games

Command: `python dominance.py --test`

| Game | Mode | Result obtained | Expected |
|---|---|---|---|
| Example 1 | strict | (H, C), payoffs (2, 3) | OK |
| Example 2 (Rock-Paper-Scissors) | strict and weak | no elimination, 3×3 unchanged | OK |
| Example 3 | strict | D removed, {H, B} × {G, M} remains | OK |
| Example 3 | weak, player 1 first | H × {G, M} | OK |
| Example 3 | weak, player 2 first | {H, B} × G | OK |
| Example 4 (3 players) | strict | (N, N, N), payoffs (0, 0, 0) | OK |
| Example 5 | strict and weak | (A3, B3), payoffs (5, 3) | OK |

## 5. Our own game: three transmitters choosing their power

### 5.1 Description

Three wireless transmitters share the same area. Each one chooses, **simultaneously**, its transmission power: **Low**, **Medium** or **High**. The payoff of each transmitter represents its useful throughput (arbitrary units). The values are invented: they were chosen so that the chain of eliminations involves **all three players**. The game has 3 × 3 × 3 = 27 profiles and is stored in `power_game_3players.json`.

Notation: each cell contains (payoff P1, payoff P2, payoff P3). Player 1 plays in rows, player 2 in columns, and there is one matrix per strategy of player 3.

**Initial game**

P3 = Low

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Low** | (9, 4, 4) | (6, 2, 0) | (3, 6, 1) |
| **Medium** | (2, 5, 1) | (2, 7, 0) | (2, 4, 1) |
| **High** | (6, 2, 0) | (3, 9, 4) | (4, 0, 5) |

P3 = Medium

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Low** | (8, 7, 6) | (1, 3, 3) | (6, 3, 3) |
| **Medium** | (3, 7, 7) | (0, 3, 6) | (9, 3, 4) |
| **High** | (9, 6, 9) | (9, 9, 7) | (7, 1, 8) |

P3 = High

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Low** | (0, 5, 0) | (0, 9, 3) | (3, 3, 0) |
| **Medium** | (6, 4, 5) | (7, 1, 0) | (3, 1, 9) |
| **High** | (7, 3, 3) | (7, 0, 5) | (5, 1, 6) |

**Starting point:** only one strategy is dominated at the start (Low for player 3). The other eliminations become possible only as strategies disappear, which is what makes this game more interesting than a 2-player game.

### 5.2 The 6 steps (strict dominance)

At each step we compare the payoffs of the player concerned for the removed strategy and for the dominating strategy, **against all remaining combinations of the two other players**.

**Step 1: player 3 removes Low (dominated by Medium).**
Against the 9 combinations (P1, P2), the payoffs of P3 are:
- Low: 4, 0, 1, 1, 0, 1, 0, 4, 5
- Medium: 6, 3, 3, 7, 6, 4, 9, 7, 8

Medium is strictly better in all 9 cases. The matrices P3 = Medium and P3 = High (3×3 each) remain.

**Step 2: player 1 removes Low (dominated by High).**
At the start this was impossible: against (P2 = Low, P3 = Low), Low gave 9 against 6 for High. That combination disappeared in step 1. Now, against the 6 remaining combinations (P2, P3):
- Low: 8, 0, 1, 0, 6, 3
- High: 9, 7, 9, 7, 7, 5

High is strictly better everywhere.

P3 = Medium

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Medium** | (3, 7, 7) | (0, 3, 6) | (9, 3, 4) |
| **High** | (9, 6, 9) | (9, 9, 7) | (7, 1, 8) |

P3 = High

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Medium** | (6, 4, 5) | (7, 1, 0) | (3, 1, 9) |
| **High** | (7, 3, 3) | (7, 0, 5) | (5, 1, 6) |

**Step 3: player 2 removes High (dominated by Low).**
Against the 4 remaining combinations (P1, P3), with P1 ∈ {Medium, High} and P3 ∈ {Medium, High}:
- High: 3, 1, 1, 1
- Low: 7, 4, 6, 3

P3 = Medium

| 1 \ 2 | Low | Medium |
|---|---|---|
| **Medium** | (3, 7, 7) | (0, 3, 6) |
| **High** | (9, 6, 9) | (9, 9, 7) |

P3 = High

| 1 \ 2 | Low | Medium |
|---|---|---|
| **Medium** | (6, 4, 5) | (7, 1, 0) |
| **High** | (7, 3, 3) | (7, 0, 5) |

**Step 4: player 3 removes High (dominated by Medium).**
Against the 4 remaining combinations (P1, P2):
- High: 5, 0, 3, 5
- Medium: 7, 6, 9, 7

Only the matrix P3 = Medium remains:

| 1 \ 2 | Low | Medium |
|---|---|---|
| **Medium** | (3, 7, 7) | (0, 3, 6) |
| **High** | (9, 6, 9) | (9, 9, 7) |

**Step 5: player 1 removes Medium (dominated by High).**
Against (P2 = Low) and (P2 = Medium), the payoffs of P1 are 3 and 0 for Medium, against 9 and 9 for High.

| 1 \ 2 | Low | Medium |
|---|---|---|
| **High** | (9, 6, 9) | (9, 9, 7) |

**Step 6: player 2 removes Low (dominated by Medium).**
The two other players are now fixed (P1 = High, P3 = Medium): Low gives 6 to P2, Medium gives 9.

| 1 \ 2 | Medium |
|---|---|
| **High** | (9, 9, 7) |

### 5.3 Result

A **single profile** remains: **(High, Medium, Medium)**, with payoffs **(9, 9, 7)**. The game is therefore solved by iterated elimination of strictly dominated strategies. The order of removals is P3, P1, P2, P3, P1, P2: each player removes two strategies, and each removal depends on the previous ones.

**Weak dominance:** the program finds the same final profile **(High, Medium, Medium)** with the default order and with a random order (for example `--seed 5`), but the sequence of removals is not the same in the two cases. Since all the dominances in this game are strict, they are also weak dominances, so the result is identical.

To reproduce the full trace:

```bash
python dominance.py --file power_game_3players.json --mode strict
```

## 6. What works and limitations

**Works:** any number of players and different numbers of strategies, keyboard input, JSON file and random generation, display for 2 players and for 3+ players, strict and weak dominance, detailed trace of every step, final result, tests on the 5 lab sheet examples, and our own 3-player game.

**Limitations:**
- Only pure strategies are handled (no dominance by a mixed strategy).
- In weak mode the program follows **one order** at a time (default or random) and does not enumerate all possible orders.
- The JSON file is not validated (shape of the array, number of names).
