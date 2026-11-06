# Game Theory Lab 2: Pure-Strategy Nash Equilibria

A Python program that takes a finite two-player game in normal form (any number of strategies for each player), computes **all the best responses** of each player, finds **all the pure-strategy Nash equilibria**, and checks whether each equilibrium is **Pareto-optimal**. Only pure strategies are considered.

## 1. Files

| File | Purpose |
|---|---|
| `nash.py` | the program |
| `power_game_2players.json` | our own 3×3 game (described in section 5) |
| `README.md` | this document |

## 2. How to run

Requirements: Python 3 and `numpy` (`pip install numpy`).

```bash
python nash.py                                      # interactive menu
python nash.py --test                               # automatic tests on the 8 built-in games
python nash.py --list                               # list the built-in games
python nash.py --example 1                          # analyse built-in game 1 (the course example)
python nash.py --file power_game_2players.json      # analyse a game stored in a JSON file
```

The interactive menu offers four sources for the game: **keyboard**, **JSON file**, **random** payoffs, or a **built-in example**.

**Keyboard input:** the program asks for m (strategies of player 1), n (strategies of player 2), optional strategy names, then the two payoffs `u1 u2` of each profile, for example `26 10`.

**JSON file format:** the payoffs are given as an m × n array of pairs `[u1, u2]`:

```json
{
  "names": [["Low", "Medium", "High"], ["Low", "Medium", "High"]],
  "payoffs": [
    [[5, 0], [9, 6], [1, 8]],
    [[1, 3], [9, 6], [2, 2]],
    [[8, 1], [3, 0], [7, 4]]
  ]
}
```

`names` is optional (default names are A1, A2, ... and B1, B2, ...). Instead of `payoffs`, the file may contain two matrices `"A"` and `"B"` of the same size.

## 3. How the code works

### 3.1 Representation of the game (R1, R2)

The game is stored as **two numpy matrices of the same size m × n**, exactly as in the lab sheet:

- `A[i, j]` = payoff of player 1 when player 1 plays row i and player 2 plays column j;
- `B[i, j]` = payoff of player 2 for the same profile;
- `row_names[i]` and `col_names[j]` hold the strategy names.

`show_game` prints three views of the game: the **matrix of payoff pairs** (u1, u2), then the matrices **A** and **B** separately.

### 3.2 Best responses (R3)

- `best_responses_1(A)` works **column by column**: for each strategy Bj of player 2, it returns **all the rows** i that reach the maximum of `A[:, j]`.
- `best_responses_2(B)` works **row by row**: for each strategy Ai of player 1, it returns **all the columns** j that reach the maximum of `B[i, :]`.

Because every row or column that reaches the maximum is kept, a best response can contain several strategies when there is a tie, as the lab sheet requires. A tolerance (`np.isclose`) is used so that ties between decimal payoffs are detected correctly.

`best_response_profiles` turns these lists into two sets of profiles, **BR1** and **BR2**, with the same notation as the lab sheet.

### 3.3 Nash equilibria (R4, R5)

A profile is a Nash equilibrium when each player's strategy is a best response to the other's, so:

```
NE = BR1 ∩ BR2
```

This is exactly `nash_equilibria`: it intersects the two sets. The program prints the best responses `R1(Bj)` and `R2(Ai)`, the sets BR1 and BR2, the equilibria with their payoffs, and a **best-response matrix** where a `*` marks the payoff that is a best response for the player concerned. A cell with two stars, `(x*, y*)`, is a Nash equilibrium. If the intersection is empty, the program prints an explicit message saying that no pure-strategy equilibrium exists.

### 3.4 Pareto analysis (R6)

- `pareto_dominates(u, v)` is true when `u1 >= v1` and `u2 >= v2` with **at least one strict inequality**.
- `pareto_dominators(A, B, i, j)` lists **all the profiles of the whole game** that Pareto-dominate the profile (i, j).

For each Nash equilibrium, if the list is empty it is **Pareto-optimal**. Otherwise the program prints **Pareto-DOMINATED by ...**, lists the dominating profiles, and marks with `[Nash equilibrium]` those that are themselves equilibria. A final summary gives the Pareto-optimal and the Pareto-dominated equilibria.

### 3.5 Tests (R7)

`python nash.py --test` checks, for each of the 8 built-in games, the exact set of equilibria and the exact set of Pareto-optimal equilibria. It also checks that the best-response sets of the course example, BR1 and BR2, match the ones written in the lab sheet, and that ties are detected (game 7).

### 3.6 Where each requirement is handled

| Requirement | Where |
|---|---|
| R1 Game input (keyboard, file) | `input_keyboard`, `load_json` / `input_file`, `input_random` |
| R2 Two matrices and matrix of pairs | `show_game` |
| R3 All best responses, ties included | `best_responses_1`, `best_responses_2` |
| R4 Intersection of best responses | `nash_equilibria`, `best_response_profiles` |
| R5 Display, explicit message if no NE | `analyse` |
| R6 Pareto analysis | `pareto_dominates`, `pareto_dominators`, `analyse` |
| R7 Tests and validation | `run_tests`, built-in games `EXAMPLES` |

## 4. Results on the test games

Command: `python nash.py --test` (all tests pass).

| # | Game | Pure Nash equilibria (payoffs) | Pareto status | Case covered |
|---|---|---|---|---|
| 1 | Course example (4×4) | (A1, B1) (26, 10) and (A2, B4) (22, 8) | (A1, B1) optimal; (A2, B4) dominated | several NE, ties, different Pareto properties |
| 2 | Prisoner's dilemma | (Confess, Confess) (−5, −5) | dominated by (Stay silent, Stay silent) | single NE |
| 3 | Battle of the sexes | (Opera, Opera) (2, 1) and (Football, Football) (1, 2) | both optimal | several NE |
| 4 | Matching pennies | none | not applicable | no NE |
| 5 | Rock-paper-scissors (3×3) | none | not applicable | no NE |
| 6 | Stag hunt | (Stag, Stag) (4, 4) and (Hare, Hare) (2, 2) | (Hare, Hare) dominated by (Stag, Stag) | several NE, Pareto-ranked |
| 7 | Game with ties (2×3) | (H, G), (H, M), (B, G), all (2, 1) | all optimal | several best responses for both players |
| 8 | Chapter 2 dominance example (3×3) | (H, G) (4, 3) | dominated by (M, M) (8, 4) and (B, M) (9, 6) | single NE found by dominance too |
| ours | Two transmitters (3×3, `power_game_2players.json`) | (Medium, Medium) (9, 6) and (High, High) (7, 4) | (High, High) dominated | several NE, tie, dominated by a non-equilibrium profile |

### 4.1 The course example in detail (game 1)

```bash
python nash.py --example 1
```

| 1 \ 2 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| **A1** | (26, 10) | (10, 2) | (30, 8) | (14, 0) |
| **A2** | (2, 2) | (2, 2) | (22, 8) | (22, 8) |
| **A3** | (24, 8) | (8, 0) | (28, 8) | (8, 2) |
| **A4** | (4, 0) | (0, 10) | (20, 8) | (16, 10) |

**Best responses of player 1** (the best row in each column, looking at u1):
R1(B1) = {A1}, R1(B2) = {A1}, R1(B3) = {A1}, R1(B4) = {A2}.

**Best responses of player 2** (the best column in each row, looking at u2). Here there are ties:
R2(A1) = {B1}, R2(A2) = {B3, B4}, R2(A3) = {B1, B3}, R2(A4) = {B2, B4}.

This gives exactly the sets of the lab sheet:

```
BR1 = {(A1, B1), (A1, B2), (A1, B3), (A2, B4)}
BR2 = {(A1, B1), (A2, B3), (A2, B4), (A3, B1), (A3, B3), (A4, B2), (A4, B4)}
```

**Best-response matrix** (the program's output, `*` = best response):

| 1 \ 2 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|
| **A1** | (26\*, 10\*) | (10\*, 2) | (30\*, 8) | (14, 0) |
| **A2** | (2, 2) | (2, 2) | (22, 8\*) | (22\*, 8\*) |
| **A3** | (24, 8\*) | (8, 0) | (28, 8\*) | (8, 2) |
| **A4** | (4, 0) | (0, 10\*) | (20, 8) | (16, 10\*) |

Two cells carry two stars, so **NE = {(A1, B1), (A2, B4)}**. For example, (A1, B3) has a star only for player 1: against B3, player 1 is happy with A1, but player 2 would rather play B1 against A1 (10 > 8), so it is not an equilibrium.

**Pareto analysis:**
- (A1, B1) with (26, 10) is **Pareto-optimal**: no profile gives both players at least as much with one strictly more.
- (A2, B4) with (22, 8) is **Pareto-dominated**. It is dominated by the other equilibrium (A1, B1), and also by (A1, B3), (A3, B1) and (A3, B3), which are not equilibria.

The lab sheet says that (A1, B1) dominates (A2, B4). The program confirms this and finds the other dominating profiles too, because it checks the whole game. So this game has several equilibria, some of which are not Pareto-optimal.

### 4.2 Why the equilibrium is not always the best outcome

- **Prisoner's dilemma (game 2):** the only equilibrium is (Confess, Confess) with (−5, −5). Both would prefer (Stay silent, Stay silent) with (−1, −1), but it is not an equilibrium: if the other stays silent, confessing gives 0, which is better than −1. Each player's best response leads them to an outcome that is worse for both. This is exactly what "a Nash equilibrium is not necessarily Pareto-optimal" means.
- **Stag hunt (game 6):** there are two equilibria. (Stag, Stag) with (4, 4) is optimal, while (Hare, Hare) with (2, 2) is also an equilibrium (nobody gains by switching alone) but both players would be better off at (Stag, Stag).
- **Battle of the sexes (game 3):** two equilibria, and both are Pareto-optimal. Each is better for one player and worse for the other, so neither dominates.
- **Matching pennies and rock-paper-scissors (games 4 and 5):** in every profile, one player can gain by changing strategy. No pure equilibrium exists, and the program prints an explicit message. (Such games have equilibria in mixed strategies, which are outside this lab.)
- **Game with ties (game 7):** best responses have several strategies, for example R1(G) = {H, B} and R2(H) = {G, M}. This gives three equilibria, all with payoffs (2, 1). Equal payoffs do not dominate each other, so all three are Pareto-optimal.
- **Chapter 2 example (game 8):** iterated elimination of strictly dominated strategies leads to (H, G), and the program finds the same profile as the **unique** Nash equilibrium. Yet (H, G) with (4, 3) is Pareto-dominated by (M, M) with (8, 4) and by (B, M) with (9, 6).

## 5. Our own game: two transmitters choosing their power

### 5.1 Description

Two wireless transmitters share the same area and choose **simultaneously** their power: **Low**, **Medium** or **High**. The payoffs (arbitrary units of useful throughput) are invented. They were chosen so that the game has two equilibria with different Pareto properties and a tie in a best response. The game is in `power_game_2players.json`.

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Low** | (5, 0) | (9, 6) | (1, 8) |
| **Medium** | (1, 3) | (9, 6) | (2, 2) |
| **High** | (8, 1) | (3, 0) | (7, 4) |

```bash
python nash.py --file power_game_2players.json
```

### 5.2 Best responses

| Against... | Player 1's best response R1 | Player 2's best response R2 (against player 1's row) |
|---|---|---|
| Low | {High} | {High} (row Low) |
| Medium | **{Low, Medium}** (tie, both give 9) | {Medium} (row Medium) |
| High | {High} | {High} (row High) |

The two columns read as follows: R1 is the best row of player 1 against each column of player 2, and R2 is the best column of player 2 against each row of player 1.

```
BR1 = {(Low, Medium), (Medium, Medium), (High, Low), (High, High)}
BR2 = {(Low, High), (Medium, Medium), (High, High)}
```

Best-response matrix:

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Low** | (5, 0) | (9\*, 6) | (1, 8\*) |
| **Medium** | (1, 3) | (9\*, 6\*) | (2, 2) |
| **High** | (8\*, 1) | (3, 0) | (7\*, 4\*) |

### 5.3 Nash equilibria and Pareto analysis

**NE = BR1 ∩ BR2 = {(Medium, Medium), (High, High)}**, with payoffs (9, 6) and (7, 4).

- **(Medium, Medium) (9, 6): Pareto-optimal.**
- **(High, High) (7, 4): Pareto-dominated.** It is dominated by (Medium, Medium), the other equilibrium, and also by **(Low, Medium)**, which has the same payoffs (9, 6) but is **not** an equilibrium: against Medium, player 1 is indifferent between Low and Medium (the tie), but player 2 would then prefer High against Low (8 > 6).

The tie matters here: (Low, Medium) is a best response for player 1 but not for player 2, so it appears in BR1 only. This is why the program keeps **all** best responses instead of just one.

(High, High) is a stable outcome (nobody gains by changing alone), but both players would be better off at (Medium, Medium). It shows once more that an equilibrium is not always the best outcome for the players.

## 6. What works and limitations

**Works:** any m × n game, keyboard input, JSON file and random generation, display of the matrix of pairs and of the two matrices A and B, all best responses with ties, all pure Nash equilibria, an explicit message when none exists, Pareto analysis against the whole game, and automatic tests on 8 built-in games covering every case of the lab sheet.

**Limitations:**
- Only two-player games and pure strategies are handled (no mixed-strategy equilibria).
- The JSON file is not validated beyond checking that A and B have the same size.
- Pareto-optimality is checked among the pure-strategy profiles of the game only.
