# Game Theory Lab 4: Nash Equilibrium in Mixed Strategies

A Python program that reads a finite n-player game from a JSON file, builds its payoff tensor with numpy, computes expected payoffs in mixed strategies, **reduces the game by iterated elimination of strictly dominated strategies (a strategy may be dominated by a mixture)**, and finds the **mixed-strategy Nash equilibrium** of the reduced game. The homework (R7), the **Lemke-Howson algorithm**, is also implemented.

## 1. Files

| File | Purpose |
|---|---|
| `mixed_nash.py` | the program (R1 to R7) |
| `example_theory.json` | the theoretical example of the lab sheet (3×3, checked by hand) |
| `teacher_game_3players.json` | the 3-player JSON game of the lab sheet, rebuilt from the format shown in it |
| `own_game_cyber.json` | our own game: a defender and an attacker (3×3) |
| `bonus_2x3.json` | a 2×3 game for the bonus of R4 |
| `battle_of_sexes.json` | a small game with three equilibria, used to illustrate Lemke-Howson |
| `README.md` | this document |

## 2. How to run

Requirements: Python 3, `numpy` and `scipy` (`pip install numpy scipy`).

```bash
python mixed_nash.py example_theory.json              # R1 to R5 on the theoretical example
python mixed_nash.py teacher_game_3players.json       # the 3-player game of the lab sheet
python mixed_nash.py own_game_cyber.json              # our own game
python mixed_nash.py bonus_2x3.json                   # bonus 2 x 3 game
python mixed_nash.py example_theory.json --lemke      # also run Lemke-Howson (R7), all starting labels
python mixed_nash.py example_theory.json --lemke --label 2    # Lemke-Howson with one starting label
python mixed_nash.py --test                           # R6: automatic tests (28 checks)
```

**JSON format** (exactly the one of the lab sheet): `strategies` gives the number of pure strategies of each player, and each key of `payoffs` is a pure profile `"s1,s2,...,sn"` (0-indexed) whose value is the list of the n payoffs.

```json
{
  "players": 3,
  "strategies": [2, 2, 2],
  "payoffs": {
    "0,0,0": [3, 2, 1],
    "0,0,1": [1, 3, 2],
    "...": "..."
  }
}
```

An **optional** field `"names": [["H", "M", "B"], ["G", "M", "D"]]` gives names to the strategies for the display. Without it, the strategies are called A1, A2, ... for player 1, B1, B2, ... for player 2, and so on. Spaces inside the keys are tolerated. Every profile must be present, otherwise the program stops with an explicit error message.

## 3. How the code works

### 3.1 R1: reading the game (`load_game`)

The function reads the JSON file and builds the **tensor T of shape (s1 × s2 × ... × sn × n)**. `T[s1, ..., sn, i]` is the payoff of player i for the pure profile (s1, ..., sn): the last axis holds the payoff vector of all the players. The function checks that `strategies` has one entry per player, that every key is a valid profile, that every payoff list has n values, and that no profile is missing.

### 3.2 R2: expected payoffs (`payoff_vs_mixed`, `expected_payoff`, `expected_payoffs`)

- `payoff_vs_mixed(T, i, sigmas)` computes u_i(s_i, σ_-i) = Σ over s_-i of (Π over j ≠ i of σ_j(s_j)) · u_i(s_i, s_-i) **for every pure strategy s_i of player i at once**. It uses one `np.einsum` that contracts the payoff tensor of player i with the probability vector of each other player.
- `expected_payoff(T, i, sigmas)` generalizes it to u_i(σ) = Σ over s of σ(s) u_i(s): it is the scalar product of σ_i with the vector above.
- `expected_payoffs` returns the vector (u_1(σ), ..., u_n(σ)).
- `is_nash(T, sigmas)` checks that no player can gain by deviating to a pure strategy. Since a mixed deviation is an average of pure ones, no mixed deviation can help either.

### 3.3 R3: reduction by mixed dominance (`iterated_mixed_elimination`)

For a player i and one of his remaining strategies s_k, a table U is built: `U[a, c]` is the payoff of player i for his a-th remaining strategy against the c-th pure profile of the other players (restricted to the strategies still in the game). The strategy s_k is strictly dominated by a mixture p of the other remaining strategies when

```
Σ_j p_j · U[j, c]  >  U[k, c]     for every c,    with  p ≥ 0,  Σ_j p_j = 1.
```

A strict inequality system cannot be given to a linear program directly, so `mixed_dominator` solves **one LP with `scipy.optimize.linprog`**: maximize t subject to

```
Σ_j p_j · U[j, c] − U[k, c]  ≥  t     for every c,     Σ_j p_j = 1,   p ≥ 0,   t free.
```

If the optimal t is strictly positive (more than 1e-7), a mixture with all inequalities strict exists, so s_k is dominated and is eliminated. The LP also returns the mixture with the **largest smallest gain** t, which is the mixture shown in the trace. For a pure strategy that dominates, the program says "pure strategy" and does not need the LP.

The algorithm follows the lab sheet exactly: for every player and every strategy, test; if a strategy is dominated, eliminate it and **restart from the beginning**; stop when nothing can be eliminated. The function returns the reduced tensor, the strategies that remain, **the list of eliminated strategies per player** (in order), and the trace of each step (player, strategy, dominating mixture, margin).

### 3.4 R4: Nash equilibrium of the reduced game (`solve`)

- **One strategy per player left:** the profile is directly the Nash equilibrium.
- **Two players with a choice:** the reduced game is a 2-player game. If the original game has more than two players but the reduction leaves only two of them with several strategies (as in the 3-player file of the lab sheet), the others are fixed to their single remaining strategy and the game is a 2-player game between the two others.
- **2 × 2 (`indifference_2x2`):** the indifference principle, step by step as in the lab sheet. Player 2 plays σ2 = (y, 1−y) and u1(s1, σ2) = u1(s1', σ2) gives y. Player 1 plays σ1 = (x, 1−x) and u2(s2, σ1) = u2(s2', σ1) gives x. The program then checks that x and y are in [0, 1]. Both steps are printed.
- **2 × 3 and 3 × 2 (bonus), and every other small size (`support_enumeration`):** the indifference principle is applied to every pair of supports (I, J) with |I| = |J|. For a pair, player 2's mix on J makes player 1 indifferent between all the rows of I, and player 1's mix on I makes player 2 indifferent between all the columns of J. Both are solved as small linear systems. A solution is kept when the probabilities are positive and **no strategy outside the supports gives a higher payoff**. This also finds all the pure equilibria (supports of size 1). The 2 × 2 case goes through the same function, which gives the same equilibrium as the formulas of the lab sheet (checked in the tests).
- **More than two players still have a choice:** the program prints an explicit message that the indifference principle of the lab covers two players only. It then lists the pure-strategy equilibria of the reduced game, if any, since they are also equilibria in mixed strategies.

**Check in the original game.** Every equilibrium is embedded back into the **original** game (probability 0 on the eliminated strategies) and verified with `is_nash`. This illustrates the property of the lab sheet: the Nash equilibria of the original game are preserved in the reduced game. The tests check this property on 60 random games.

### 3.5 R5: display

The program prints the original game as a table (for 3 or more players, one matrix per combination of the strategies of players 3 to n), each elimination with the dominating mixture and the remaining strategies, the game after each step, the list of eliminated strategies per player, the final reduced game, and the equilibrium with its probabilities, its expected payoffs and its verification. If no equilibrium can be determined, an explicit message is printed. Probabilities are shown as exact fractions when they are (1/4, 2/3, ...).

### 3.6 R7: Lemke-Howson (`lemke_howson`, `--lemke`)

The algorithm finds one Nash equilibrium of a bimatrix game (A, B), where A and B hold the payoffs of players 1 and 2.

1. A constant is added to all payoffs so that they are positive. This does not change the equilibria.
2. Two systems are considered: `A y + r = 1` (with y, r ≥ 0) for player 2's mix, and `Bᵀ x + s = 1` (with x, s ≥ 0) for player 1's mix. The variables x_i and r_i carry the label i, and y_j and s_j carry the label m + j.
3. The starting point x = 0, y = 0 has every label. One label k is chosen and "dropped": its variable enters the basis of one system (a pivot of the simplex method). The variable that leaves carries some label l. In the other system, the variable with label l then enters the basis. The path continues like this, alternating between the two systems.
4. It stops when the label that was dropped leaves a basis. The two basic solutions then have all the labels, and x/Σx, y/Σy is a Nash equilibrium.

The leaving variable is chosen with the **lexicographic minimum-ratio rule**, so degenerate games (ties in the payoffs) are also handled. Different starting labels may lead to different equilibria. `--lemke` runs all the labels, verifies each result, and counts the distinct equilibria.

## 4. Results

### 4.1 The theoretical example of the lab sheet

```bash
python mixed_nash.py example_theory.json --lemke
```

| 1 \ 2 | G | M | D |
|---|---|---|---|
| **H** | (1, 1) | (0, 2) | (0, 4) |
| **M** | (0, 2) | (5, 0) | (1, 6) |
| **B** | (0, 2) | (1, 1) | (2, 1) |

**Step 1: no pure domination, then mixed domination of M for player 2.** No strategy is strictly dominated by a pure strategy, as stated in the lab sheet. The program eliminates M of player 2, dominated by a mixture of G and D. For a mixture p·G + (1−p)·D, the gains over M against the rows H, M, B are 2 − 3p, 6 − 4p and p. They are all positive exactly when 0 < p < 2/3, the interval found by hand in the lab sheet. The LP chooses the mixture that makes the smallest gain largest, which is **p = 1/2**: the gains are then 1/2, 4 and 1/2.

```
Step 1: player 2 eliminates M, strictly dominated by the mixture 1/2 G + 1/2 D (smallest gain over M: 1/2)
```

**Step 2: M of player 1 is dominated by a mixture of H and B.** In the reduced game, the mixture q·H + (1−q)·B gives q against G and 2(1−q) against D, while M gives 0 and 1. The gains q and 1 − 2q are both positive for 0 < q < 1/2. The LP picks **q = 1/3**: the gains are then 1/3 and 1/3.

```
Step 2: player 1 eliminates M, strictly dominated by the mixture 1/3 H + 2/3 B (smallest gain over M: 1/3)
```

Nothing more can be eliminated. The eliminated strategies are M for player 1 and M for player 2. The reduced game is

| 1 \ 2 | G | D |
|---|---|---|
| **H** | (1, 1) | (0, 4) |
| **B** | (0, 2) | (2, 1) |

**Indifference principle.**
- Player 1 is indifferent between H and B when player 2 plays (y, 1−y) on (G, D): y = 2(1−y), so **y = 2/3**.
- Player 2 is indifferent between G and D when player 1 plays (x, 1−x) on (H, B): x + 2(1−x) = 4x + (1−x), so **x = 1/4**.
- Both are in [0, 1].

**Nash equilibrium of the original game:** σ1 = 1/4 H + 3/4 B (and 0 on M), σ2 = 2/3 G + 1/3 D (and 0 on M), with expected payoffs **(2/3, 7/4)**.

**Check in the original game.** Against σ2, player 1 gets 2/3 with H, 1/3 with M and 2/3 with B, so nobody beats the equilibrium payoff 2/3. Against σ1, player 2 gets 7/4 with G, 5/4 with M and 7/4 with D, so nobody beats 7/4. The program prints this check. **Lemke-Howson** gives exactly the same equilibrium from all 6 starting labels.

### 4.2 The 3-player file of the lab sheet

```bash
python mixed_nash.py teacher_game_3players.json
```

This file is rebuilt from the profile list shown in the lab sheet: if your teacher's file is different, just run the program on it.

- **Reduction:** player 2's strategy B2 is strictly dominated by the pure strategy B1 (the gain is exactly 1 against each of the 4 pure profiles of players 1 and 3). Nothing else can be eliminated.
- **Reduced game:** only players 1 and 3 still have a choice, while player 2 plays B1. This is a 2-player game between players 1 and 3 (rows A1, A2 for player 1, columns C1, C2 for player 3), so the indifference principle applies.
- **Indifference:** player 3 plays (y, 1−y) on (C1, C2) and makes player 1 indifferent between A1 and A2: 3y + (1−y) = y + 2(1−y), so y = 1/3. Player 1 plays (x, 1−x) on (A1, A2) and makes player 3 indifferent between C1 and C2: x + 4(1−x) = 2x + (1−x), so x = 3/4.
- **Equilibrium:** σ1 = 3/4 A1 + 1/4 A2, σ2 = B1, σ3 = 1/3 C1 + 2/3 C2, with expected payoffs **(5/3, 8/3, 7/4)**. The program verifies it in the original 3-player game: in particular, player 2 gets 8/3 with B1 but only 5/3 with B2.

### 4.3 Our own game: a defender and an attacker

```bash
python mixed_nash.py own_game_cyber.json --lemke
```

A network defender (player 1) chooses a defense and an attacker (player 2) chooses an attack. The payoffs are invented. The game was chosen so that **a strategy is dominated only by a mixture** and the equilibrium is genuinely mixed (it has no pure-strategy equilibrium).

| 1 \ 2 | Phishing | DoS | Exploit |
|---|---|---|---|
| **Firewall** | (7, 3) | (9, 2) | (1, 1) |
| **Monitoring** | (9, 3) | (3, 5) | (9, 0) |
| **Honeypot** | (5, 5) | (6, 2) | (4, 1) |

**Step 1: Honeypot is dominated only by a mixture.** No pure strategy dominates Honeypot. But the mixture 4/7 Firewall + 3/7 Monitoring gives 55/7 ≈ 7.86 against Phishing (Honeypot: 5), 45/7 ≈ 6.43 against DoS (Honeypot: 6) and 31/7 ≈ 4.43 against Exploit (Honeypot: 4). It is strictly better in all three cases, and the smallest gain is 3/7. This is a case where the LP is really needed.

**Step 2: Exploit is dominated by the pure strategy Phishing.** Against the two remaining rows (Firewall and Monitoring), the attacker gets 3 and 3 with Phishing, against 1 and 0 with Exploit (smallest gain 2). The reduced game is 2 × 2:

| 1 \ 2 | Phishing | DoS |
|---|---|---|
| **Firewall** | (7, 3) | (9, 2) |
| **Monitoring** | (9, 3) | (3, 5) |

**Indifference:** 7y + 9(1−y) = 9y + 3(1−y) gives **y = 3/4**; 3x + 3(1−x) = 2x + 5(1−x) gives **x = 2/3**.

**Equilibrium:** σ1 = 2/3 Firewall + 1/3 Monitoring, σ2 = 3/4 Phishing + 1/4 DoS, with expected payoffs **(15/2, 3)**. It is verified in the original 3 × 3 game, and Lemke-Howson finds the same equilibrium from all 6 starting labels.

### 4.4 Bonus: a 2 × 3 game

```bash
python mixed_nash.py bonus_2x3.json --lemke
```

| 1 \ 2 | B1 | B2 | B3 |
|---|---|---|---|
| **A1** | (2, 3) | (3, 1) | (4, 3) |
| **A2** | (4, 3) | (0, 9) | (9, 7) |

No strategy is dominated, so the reduced game is the 2 × 3 game itself. The support enumeration tries the pairs of supports of equal size. The only equilibrium uses the supports {A1, A2} and {B2, B3}:

- Player 2's mix on (B2, B3) makes player 1 indifferent: 3·y2 + 4·y3 = 9·y3, so y2 = 5/8 and y3 = 3/8.
- Player 1's mix makes player 2 indifferent between B2 and B3: 9 − 8x = 7 − 4x, so x = 1/2.
- B1 is not used: it gives player 2 only 3, against 5 at the equilibrium.

**Equilibrium:** σ1 = (1/2, 1/2), σ2 = (0, 5/8, 3/8), with expected payoffs **(27/8, 5)**. Lemke-Howson finds the same one.

### 4.5 Lemke-Howson on a game with several equilibria

```bash
python mixed_nash.py battle_of_sexes.json --lemke
```

The battle of the sexes has three equilibria: (Opera, Opera) with payoffs (2, 1), (Football, Football) with (1, 2), and a mixed one σ1 = 2/3 Opera + 1/3 Football, σ2 = 1/3 Opera + 2/3 Football with payoffs (2/3, 2/3). The support enumeration of the main program finds all three. Lemke-Howson returns **one equilibrium per starting label**: from the four labels it returns the two pure equilibria. This shows that the algorithm always finds an equilibrium but not necessarily all of them.

### 4.6 Automatic tests (R6)

Command: `python mixed_nash.py --test` (28 checks, all pass).

| Requirement | What is tested |
|---|---|
| R1 | the tensor built from the 3-player file has shape (2, 2, 2, 3) and the right payoffs for the profiles checked |
| R2 | u_i(σ) and u_i(s_i, σ_-i) equal the definition (sum over all pure profiles) on 50 random games with 2 to 4 players |
| R3 | on the theoretical example: no pure domination, step 1 = M of player 2 dominated by 1/2 G + 1/2 D, step 2 = M of player 1 dominated by 1/3 H + 2/3 B, 2 × 2 reduced game |
| R3 | prisoner's dilemma: reduced to one profile (Confess, Confess) by pure dominance only |
| R3 | **the equilibria of the original game are exactly those of the reduced game** (60 random games) |
| R4 | theoretical example: x = 1/4, y = 2/3, payoffs (2/3, 7/4), verified in the original game; matching pennies: (1/2, 1/2) with no elimination |
| R6 | the 3-player file, our own game (including the mixture 4/7 Firewall + 3/7 Monitoring) and the 2 × 3 bonus game, with exact probabilities and payoffs |
| R7 | Lemke-Howson returns a Nash equilibrium for every starting label on 200 random games from 1×1 to 6×6 (with many ties, so many degenerate cases), and finds the equilibrium of the theoretical example |

## 5. What works and limitations

**Works:** reading the JSON format of the lab sheet into a numpy tensor, expected payoffs for any number of players, iterated elimination with mixed dominance through `linprog`, the Nash equilibrium after reduction (directly, 2 × 2 with the steps of the lab sheet, 2 × 3 bonus and other small 2-player games), verification of every equilibrium in the original game, and Lemke-Howson for any bimatrix game.

**Limitations:**
- Only **strict** dominance is used for the reduction, as the lab sheet requires. A strategy that is only weakly dominated stays in the game.
- Dominance is tested with a numerical tolerance (1e-7 on the LP margin).
- The indifference principle covers **two-player** reduced games. If more than two players still have a choice after the reduction, the program says so and only reports pure equilibria.
- The support enumeration handles reduced games up to 6 × 6, and skips support pairs whose indifference system is singular (degenerate games with infinitely many equilibria may therefore be only partly reported).
- Lemke-Howson returns one equilibrium per starting label, not necessarily all equilibria.
- The 3-player file is rebuilt from the lab sheet, so run the program on the exact file you are given.
