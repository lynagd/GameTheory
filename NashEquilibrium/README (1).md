# Game Theory Lab 3: Pure-Strategy Nash Equilibria for n Players

This is the n-player version of the Nash equilibrium program of TP 2. It takes a finite game with **any number of players n ≥ 2** and **any number of strategies for each player**, computes **all the best responses** of every player, finds **all the pure-strategy Nash equilibria**, and checks whether each equilibrium is **Pareto-optimal**. Only pure strategies are considered.

## 1. Files

| File | Purpose |
|---|---|
| `nash_n.py` | the program |
| `price_game_3players.json` | our own 3-player game (described in section 5) |
| `README.md` | this document |

## 2. How to run

Requirements: Python 3 and `numpy` (`pip install numpy`).

```bash
python nash_n.py                                   # interactive menu
python nash_n.py --test                            # automatic tests on the 8 built-in games
python nash_n.py --list                            # list the built-in games
python nash_n.py --example 6                       # analyse built-in game 6 (4 players)
python nash_n.py --file price_game_3players.json   # analyse a game stored in a JSON file
```

The interactive menu offers four sources for the game: **keyboard**, **JSON file**, **random** payoffs, or a **built-in example**.

**Keyboard input:** the program asks for the number of players n, the number of strategies of each player, optional strategy names, then the n payoffs of each profile, for example `3 3 3` for a 3-player game.

**JSON file format:** `payoffs[s1][s2]...[sn]` is the list of the n payoffs of the profile (s1, ..., sn), so `payoffs` has shape (k1, ..., kn, n). This is the same format as in TP 1. A 2-player example:

```json
{
  "names": [["A1", "A2"], ["B1", "B2"]],
  "payoffs": [[[3, 3], [0, 4]], [[4, 0], [1, 1]]]
}
```

`names` is optional (default names are A1, A2, ..., then B1, B2, ..., one letter per player).

## 3. How the code works

### 3.1 Representation of the game (R1, R2)

The game is stored in **one numpy array with n + 1 dimensions**:

```
game[s1, s2, ..., sn, i] = payoff of player i for the profile (s1, ..., sn)
```

- The first n axes are the strategies of each player; their sizes kᵢ may differ.
- The last axis holds the payoff vector.
- The number of players is `game.ndim - 1`, so there is no special case for 2, 3 or 4 players.

**Display (`show_matrices`):** each cell shows the vector (u1, u2, ..., un). Player 1 plays in rows and player 2 in columns. For n ≥ 3 there is **one matrix for each combination of strategies of players 3 to n**, labelled for example `[P3 = Low, P4 = High]`.

### 3.2 What changes compared with TP 2

For two players, the best response of player 1 depends on the column chosen by player 2, and vice versa. With n players, the best response of player i depends on **the strategies of all the other n − 1 players**:

```
R_i(s_-i) = arg max over s_i of u_i(s_i, s_-i)
```

The Nash condition is the same idea: a profile is an equilibrium when **every** player plays a best response. The intersection of two sets BR1 ∩ BR2 becomes the intersection of n sets:

```
NE = BR1 ∩ BR2 ∩ ... ∩ BRn
```

### 3.3 Best responses (R3)

`best_response_masks(game)` returns, for each player i, a boolean array `masks[i]` with the same shape as the game:

- `masks[i][s1, ..., sn]` is True when `s_i` is a best response to the strategies of the others in that profile.
- It is computed in one step: `u = game[..., i]`, then `u.max(axis=i, keepdims=True)` gives the best payoff of player i for each combination of the others, and `u >= best` marks **every** strategy that reaches it.

Because every maximizer is marked, **ties are kept**: a best response may contain several strategies, as the lab sheet requires. A small tolerance (`1e-9`) is used so that ties between decimal payoffs are detected correctly.

`best_responses(game, masks, i)` lists these sets in the form `R_i(others)`. In the output, `R2(High, Medium)` means: the best response(s) of player 2 when player 1 plays High and player 3 plays Medium (the others are written in player order, skipping player i). The sets BR1, ..., BRn are printed as sets of full profiles.

### 3.4 Nash equilibria (R4, R5)

`nash_equilibria` combines the masks with `np.logical_and.reduce`: a profile is kept when it is a best response for all players. The program prints:

- the best responses `R_i(...)` of every player and the sets BR1, ..., BRn;
- a **best-response matrix**, where a `*` after the payoff of player i means "the strategy of player i is a best response". A cell where **all n payoffs carry a star** is a Nash equilibrium;
- the equilibria with their payoffs, or an **explicit message** if no pure equilibrium exists.

### 3.5 Pareto analysis (R6)

`pareto_dominators(game, profile)` finds **all profiles of the whole game** where every player gets at least as much, and at least one player gets strictly more. For each equilibrium:

- no such profile: the equilibrium is **Pareto-optimal**;
- otherwise: **Pareto-DOMINATED by ...**, listing the dominating profiles, with `[Nash equilibrium]` next to those that are equilibria themselves.

A final summary lists the Pareto-optimal and the Pareto-dominated equilibria.

### 3.6 Tests and validation (R7)

`python nash_n.py --test` checks, for each of the 8 built-in games, the exact set of equilibria and the exact set of Pareto-optimal equilibria. It also checks the best-response sets of the TP 2 course example (so the n-player program gives the same results as the 2-player one), the size of the 4-player game, and that ties are detected.

In addition, the numpy computation was cross-checked against a separate brute-force implementation that follows the definition literally (every profile, every unilateral deviation) on 300 random games with 2 to 4 players and 1 to 3 strategies per player. There were no differences, in equilibria or in Pareto dominators.

### 3.7 Where each requirement is handled

| Requirement | Where |
|---|---|
| R1 Game input (keyboard, file) | `input_keyboard`, `load_json` / `input_file`, `input_random` |
| R2 Display of the game | `show_matrices` |
| R3 All best responses, ties included | `best_response_masks`, `best_responses` |
| R4 Intersection of best responses | `nash_equilibria` |
| R5 Display, explicit message if no NE | `analyse` |
| R6 Pareto analysis | `pareto_dominators`, `analyse` |
| R7 Tests and validation | `run_tests`, built-in games `EXAMPLES` |

## 4. Results on the test games

Command: `python nash_n.py --test` (all tests pass).

| # | Game | Players, strategies | Pure Nash equilibria (payoffs) | Pareto status |
|---|---|---|---|---|
| 1 | Course example of TP 2 | 2, 4×4 | (A1, B1) (26, 10) and (A2, B4) (22, 8) | (A1, B1) optimal; (A2, B4) dominated |
| 2 | Prisoner's dilemma | 2, 2×2 | (Confess, Confess) (−5, −5) | dominated by (Silent, Silent) |
| 3 | Contribution game | 3, 2×2×2 | (N, N, N) (0, 0, 0) | dominated by (C, C, C) (3, 3, 3) and others |
| 4 | Cyclic matching pennies | 3, 2×2×2 | none | not applicable |
| 5 | Coordination | 3, 2×2×2 | (Left, Left, Left) (2, 2, 2) and (Right, Right, Right) (1, 1, 1) | (Right, Right, Right) dominated |
| 6 | Volunteer's dilemma | 4, 2×2×2×2 | 4 equilibria, exactly one player volunteers | all four optimal |
| 7 | TP 1 game | 3, 3×3×3 | (High, Medium, Medium) (9, 9, 7) | optimal |
| 8 | Ties with different strategy counts | 3, 2×3×2 | (X1, Y2, Z2) (3, 2, 2) and (X2, Y3, Z2) (0, 3, 1) | (X2, Y3, Z2) dominated |
| ours | Price war | 3, 3×3×3 | (Low, Low, Low) (5, 5, 4) and (High, High, Medium) (4, 9, 9) | (Low, Low, Low) dominated |

Cases required by the lab sheet: **one equilibrium** (games 2, 3, 7), **several equilibria** (1, 5, 6, 8), **no equilibrium** (4), **several best responses** (1, 8, ours), **different Pareto properties** (1, 5, 8, ours).

### 4.1 A 3-player game in detail: the contribution game (game 3)

Each of 3 players chooses C (contribute) or N (do not contribute). If c players contribute, a player who contributes gets 2c − 3 and a player who does not gets 2c.

```bash
python nash_n.py --example 3
```

**Best responses.** For every combination of the two other players, the best response of each player is {N}:

```
R1(C, C) = {N}   R1(C, N) = {N}   R1(N, C) = {N}   R1(N, N) = {N}
```

and the same for players 2 and 3. For example, if the two others contribute, contributing gives 3 but not contributing gives 4. So the sets are:

```
BR1 = {(N, C, C), (N, C, N), (N, N, C), (N, N, N)}
BR2 = {(C, N, C), (C, N, N), (N, N, C), (N, N, N)}
BR3 = {(C, C, N), (C, N, N), (N, C, N), (N, N, N)}
```

The only profile that belongs to **all three** sets is **(N, N, N)**, so **NE = {(N, N, N)}** with payoffs (0, 0, 0). In the best-response matrix it is the only cell with three stars: `(0*, 0*, 0*)`.

**Pareto analysis.** (N, N, N) is Pareto-**dominated** by (C, C, C) with (3, 3, 3), and also by (C, C, N), (C, N, C) and (N, C, C). Each player's best response leads to an outcome that is worse for everybody. It is the n-player version of the prisoner's dilemma.

### 4.2 Other results explained

- **Cyclic matching pennies (game 4):** player 1 wants to match player 2, player 2 wants to match player 3, and player 3 wants to differ from player 1. Whatever the profile, one player can gain by changing strategy, so the intersection of BR1, BR2 and BR3 is empty and the program prints an explicit message.
- **Coordination (game 5):** everybody gets (2, 2, 2) if all play Left, (1, 1, 1) if all play Right, and 0 otherwise. Both unanimous profiles are equilibria (no single player can improve by leaving the group), but (Right, Right, Right) is Pareto-dominated by (Left, Left, Left): the players can get stuck in a worse equilibrium.
- **Volunteer's dilemma (game 6):** 4 players, a volunteer pays a cost, so volunteering gives 0.6 and waiting gives 1 if somebody else volunteers (0 if nobody does). If nobody volunteers, one player gains by volunteering (0.6 > 0). If two volunteer, one of them gains by waiting (1 > 0.6). Only the profiles with **exactly one volunteer** are stable: 4 equilibria, all Pareto-optimal. The game also shows that the program works with 4 players and 16 profiles.
- **TP 1 game (game 7):** in TP 1, iterated elimination of strictly dominated strategies led to (High, Medium, Medium) with (9, 9, 7). Here the Nash analysis finds the same profile as the **only** equilibrium. This is expected: when iterated strict dominance leaves a single profile, that profile is the unique Nash equilibrium.
- **Different strategy counts (game 8):** players have 2, 3 and 2 strategies, and the best responses contain ties. Two equilibria: (X1, Y2, Z2) is Pareto-optimal, while (X2, Y3, Z2) is dominated, for example by (X1, Y2, Z1) with (0, 3, 2), which is not an equilibrium.

## 5. Our own game: three companies in a price war

### 5.1 Description

Three companies choose **simultaneously** their price level: **Low**, **Medium** or **High**. The payoffs (arbitrary units of profit) are invented. They were chosen so that the game has two equilibria with different Pareto properties and ties in some best responses. The game has 3 × 3 × 3 = 27 profiles and is stored in `price_game_3players.json`.

Each cell contains (profit of company 1, company 2, company 3). Company 1 plays in rows, company 2 in columns, and there is one matrix per price of company 3.

P3 = Low

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Low** | (5, 5, 4) | (7, 4, 8) | (6, 0, 8) |
| **Medium** | (0, 2, 0) | (1, 8, 6) | (2, 9, 1) |
| **High** | (5, 2, 7) | (3, 1, 3) | (9, 6, 3) |

P3 = Medium

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Low** | (3, 0, 0) | (7, 1, 3) | (1, 6, 5) |
| **Medium** | (9, 4, 6) | (9, 0, 4) | (1, 9, 5) |
| **High** | (4, 6, 8) | (0, 9, 2) | (4, 9, 9) |

P3 = High

| 1 \ 2 | Low | Medium | High |
|---|---|---|---|
| **Low** | (0, 6, 1) | (9, 0, 6) | (1, 0, 2) |
| **Medium** | (8, 3, 3) | (9, 5, 1) | (0, 7, 4) |
| **High** | (4, 5, 8) | (6, 6, 6) | (9, 2, 5) |

```bash
python nash_n.py --file price_game_3players.json
```

### 5.2 Best responses

The program computes 27 sets (9 for each player). The label `R2(High, Medium)` reads: best response of company 2 when company 1 plays High and company 3 plays Medium.

| Player 1 | | Player 2 | | Player 3 | |
|---|---|---|---|---|---|
| R1(Low, Low) | {Low, High} | R2(Low, Low) | {Low} | R3(Low, Low) | {Low} |
| R1(Low, Medium) | {Medium} | R2(Low, Medium) | {High} | R3(Low, Medium) | {Low} |
| R1(Low, High) | {Medium} | R2(Low, High) | {Low} | R3(Low, High) | {Low} |
| R1(Medium, Low) | {Low} | R2(Medium, Low) | {High} | R3(Medium, Low) | {Medium} |
| R1(Medium, Medium) | {Medium} | R2(Medium, Medium) | {High} | R3(Medium, Medium) | {Low} |
| R1(Medium, High) | {Low, Medium} | R2(Medium, High) | {High} | R3(Medium, High) | {Medium} |
| R1(High, Low) | {High} | R2(High, Low) | {High} | R3(High, Low) | {Medium, High} |
| R1(High, Medium) | {High} | R2(High, Medium) | {Medium, High} | R3(High, Medium) | {High} |
| R1(High, High) | {High} | R2(High, High) | {Medium} | R3(High, High) | {Medium} |

Four of these sets contain two strategies (ties): R1(Low, Low), R1(Medium, High), R2(High, Medium) and R3(High, Low).

### 5.3 Nash equilibria

**NE = BR1 ∩ BR2 ∩ BR3 = {(Low, Low, Low), (High, High, Medium)}**

- **(Low, Low, Low), payoffs (5, 5, 4).** Company 1's best response to (Low, Low) is {Low, High}, which contains Low (a tie: both give 5). Company 2's best response to (Low, Low) is {Low}, and company 3's is {Low}. So nobody gains by changing alone. In the best-response matrix for P3 = Low, the cell shows `(5*, 5*, 4*)`.
- **(High, High, Medium), payoffs (4, 9, 9).** Company 1's best response to (High, Medium) is {High}. Company 2's best response to (High, Medium) is {Medium, High}, which contains High (a tie: both give 9). Company 3's best response to (High, High) is {Medium}. In the matrix for P3 = Medium, the cell shows `(4*, 9*, 9*)`.

The tie in R2(High, Medium) matters: if the program kept only one best response, it could miss this equilibrium. This is why all maximizers are kept.

### 5.4 Pareto analysis

- **(High, High, Medium) (4, 9, 9): Pareto-optimal.**
- **(Low, Low, Low) (5, 5, 4): Pareto-dominated** by (High, Medium, High) with (6, 6, 6), where all three companies earn more. That profile is **not** an equilibrium: companies 2 and 3 are playing a best response there, but company 1 is not. Against (Medium, High), the best response of company 1 is R1(Medium, High) = {Low, Medium}, which gives it 9, while High gives only 6. So company 1 would deviate.

So the price war (Low, Low, Low) is a stable outcome that all three companies would leave together if they could coordinate on (High, Medium, High). The two equilibria cannot be ranked against each other: (Low, Low, Low) is better for company 1 (5 against 4), while (High, High, Medium) is better for companies 2 and 3 (9 against 5 and 4).

## 6. What works and limitations

**Works:** any number of players and any number of strategies per player, keyboard input, JSON file and random generation, display of the game for 2 players and for 3 or more players, all best responses with ties, all pure Nash equilibria, an explicit message when none exists, Pareto analysis against the whole game, and automatic tests on 8 built-in games plus a brute-force cross-check.

**Limitations:**
- Only pure strategies are handled (no mixed-strategy equilibria).
- The number of profiles is the product of the kᵢ, so it grows exponentially with the number of players. Games with many players are practical only with few strategies each (for example 10 players with 2 strategies is 1024 profiles), and the printed output becomes long.
- The JSON file is validated only for its shape (n payoffs per profile).
- Pareto-optimality is checked among the pure-strategy profiles of the game only.
