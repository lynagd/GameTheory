"""
Game theory lab 4: Nash equilibrium in mixed strategies.

What the program does :
    R1  reads a game from a JSON file and builds the payoff tensor T of shape (s1 x ... x sn x n)
    R2  computes expected payoffs for mixed strategies: u_i(s_i, sigma_-i) and u_i(sigma)
    R3  reduces the game by iterated elimination of strictly dominated strategies, where a
        strategy can be dominated by a MIXED strategy (checked with a linear program, linprog)
    R4  finds the Nash equilibrium of the reduced game: directly if one strategy per player is
        left, otherwise with the indifference principle (2x2, and 2x3 / 3x2 as a bonus). When the
        reduction leaves only two players with a choice (the others have one strategy), the
        reduced game is a 2-player game between those two players.
    R5  displays the original game, each elimination, the reduced game and the equilibrium
    R6  tests on the theoretical example, the teacher's JSON file and our own games (--test)
    R7  Lemke-Howson algorithm for a bimatrix game (A, B)  (--lemke)

Usage:  python mixed_nash.py game.json                  (R1 to R5)
        python mixed_nash.py game.json --lemke           (also run Lemke-Howson, R7)
        python mixed_nash.py game.json --lemke --label 2 (Lemke-Howson with one initial label)
        python mixed_nash.py --test                      (R6: automatic tests)
"""
import os
import sys
import json
import argparse
import itertools
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog

TOL = 1e-9          # general numerical tolerance
DOM_TOL = 1e-7      # a strategy is dominated if the LP margin is larger than this


# ----------------------------------------------------------------------------
# Small helpers (names, display of numbers and tables)
# ----------------------------------------------------------------------------
def default_names(ks):
    letters = "ABCDEFGH"
    return [[f"{letters[i] if i < len(letters) else 'P' + str(i + 1) + '_'}{j + 1}"
             for j in range(k)] for i, k in enumerate(ks)]


def fr(x):
    """Number as an exact fraction when it is one (0.25 -> 1/4), otherwise 4 decimals."""
    x = float(x)
    f = Fraction(x).limit_denominator(1000)
    return str(f) if abs(float(f) - x) < 1e-9 else f"{x:.4f}"


def mix_str(names_i, p):
    """Mixed strategy as a sum: 1/4 H + 3/4 B (strategies with probability 0 are left out)."""
    terms = [f"{fr(q)} {names_i[k]}" if abs(q - 1) > 1e-9 else f"{names_i[k]}"
             for k, q in enumerate(p) if q > TOL]
    return " + ".join(terms)


def vec_str(names_i, p):
    """Mixed strategy as a vector: (H: 1/4, M: 0, B: 3/4)."""
    return "(" + ", ".join(f"{names_i[k]}: {fr(q)}" for k, q in enumerate(p)) + ")"


def payoff_str(T, profile, fmt=fr):
    return "(" + ", ".join(fmt(v) for v in T[tuple(profile)]) + ")"


def print_game(T, names):
    """Player 1 = rows, player 2 = columns, one matrix per combination of players 3..n."""
    n = T.ndim - 1
    k = T.shape[:-1]
    for fixed in itertools.product(*[range(x) for x in k[2:]]):
        if n > 2:
            print("  [" + ", ".join(f"P{p + 3} = {names[p + 2][s]}" for p, s in enumerate(fixed)) + "]")
        table = [["1\\2"] + list(names[1])]
        for r in range(k[0]):
            table.append([names[0][r]] + [payoff_str(T, (r, c) + fixed) for c in range(k[1])])
        widths = [max(len(row[j]) for row in table) for j in range(len(table[0]))]
        for row in table:
            print("  " + "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if n > 2:
            print()


# ----------------------------------------------------------------------------
# R1. Reading the game from a JSON file
# ----------------------------------------------------------------------------
def load_game(path):
    """Reads the JSON file and builds the payoff tensor T of shape (s1, ..., sn, n).
    T[s1, ..., sn, i] is the payoff of player i for the pure profile (s1, ..., sn).
    File format:  {"players": 3, "strategies": [2, 2, 2],
                   "payoffs": {"0,0,0": [3, 2, 1], "0,0,1": [1, 3, 2], ...}}
    An optional field "names" ([[names of player 1], [names of player 2], ...]) is used for display."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data = {str(k).strip(): v for k, v in data.items()}          # tolerate stray spaces in field names
    for key in ("players", "strategies", "payoffs"):
        if key not in data:
            raise ValueError(f"The JSON file has no field '{key}'.")
    n = int(data["players"])
    ks = [int(k) for k in data["strategies"]]
    if n < 2:
        raise ValueError("The game needs at least 2 players.")
    if len(ks) != n or min(ks) < 1:
        raise ValueError("'strategies' must contain one positive integer per player.")
    T = np.full(tuple(ks) + (n,), np.nan)
    for key, vec in data["payoffs"].items():
        idx = tuple(int(part) for part in key.split(","))          # " 0 ,1 " -> (0, 1)
        if len(idx) != n or any(not 0 <= idx[i] < ks[i] for i in range(n)):
            raise ValueError(f"Invalid profile '{key}' for strategies {ks}.")
        if len(vec) != n:
            raise ValueError(f"Profile '{key}' must have {n} payoffs, one per player.")
        T[idx] = vec
    missing = [p for p in itertools.product(*[range(k) for k in ks]) if np.isnan(T[p]).any()]
    if missing:
        raise ValueError(f"{len(missing)} profile(s) have no payoffs, for example {missing[0]}.")
    names = data.get("names") or default_names(ks)
    return T, names


# ----------------------------------------------------------------------------
# R2. Expected payoffs in mixed strategies
# ----------------------------------------------------------------------------
def payoff_vs_mixed(T, i, sigmas):
    """u_i(s_i, sigma_-i) for EVERY pure strategy s_i of player i, against the mixed strategies
    of the other players:   sum over s_-i of (prod_{j != i} sigma_j(s_j)) * u_i(s_i, s_-i).
    sigmas[j] is the probability vector of player j (sigmas[i] is ignored).
    Returns a vector of length s_i."""
    n = T.ndim - 1
    operands = [T[..., i], list(range(n))]
    for j in range(n):
        if j != i:
            operands += [np.asarray(sigmas[j], dtype=float), [j]]
    return np.einsum(*operands, [i])


def expected_payoff(T, i, sigmas):
    """u_i(sigma) = sum over s of sigma(s) u_i(s), with sigma(s) = prod_j sigma_j(s_j)."""
    return float(np.dot(np.asarray(sigmas[i], dtype=float), payoff_vs_mixed(T, i, sigmas)))


def expected_payoffs(T, sigmas):
    """Vector (u_1(sigma), ..., u_n(sigma))."""
    return np.array([expected_payoff(T, i, sigmas) for i in range(T.ndim - 1)])


def is_nash(T, sigmas, tol=1e-7):
    """True if no player can gain by deviating to a pure strategy (then no mixed deviation helps either)."""
    for i in range(T.ndim - 1):
        vec = payoff_vs_mixed(T, i, sigmas)
        if expected_payoff(T, i, sigmas) < vec.max() - tol:
            return False
    return True


# ----------------------------------------------------------------------------
# R3. Reduction by iterated elimination of strictly dominated strategies (mixed dominance)
# ----------------------------------------------------------------------------
def payoff_table(T, kept, i):
    """U[a, c] = payoff of player i for his a-th remaining strategy against the c-th pure profile
    of the other players (restricted to the strategies still in the game)."""
    sub = T[np.ix_(*kept)][..., i]
    sub = np.moveaxis(sub, i, 0)
    return sub.reshape(sub.shape[0], -1)


def pure_dominator(U, a):
    """Index of a pure strategy that strictly dominates strategy a, or None."""
    for b in range(U.shape[0]):
        if b != a and np.all(U[b] > U[a] + DOM_TOL):
            return b
    return None


def mixed_dominator(U, a):
    """Is strategy a strictly dominated by a mixture p of the other remaining strategies?
    Linear program: maximize t subject to
        sum_j p_j U[j, c] - U[a, c] >= t   for every pure profile c of the others,
        sum_j p_j = 1,  p >= 0,  t free.
    The strict inequalities hold for some p exactly when the optimal t is > 0.
    Returns (p, t) over the rows of U (p[a] = 0), or None."""
    others = [b for b in range(U.shape[0]) if b != a]
    if not others:
        return None
    m = len(others)
    c = np.zeros(m + 1)
    c[-1] = -1.0                                                   # maximize t
    A_ub = np.hstack([-U[others].T, np.ones((U.shape[1], 1))])     # -sum p_j U[j,c] + t <= -U[a,c]
    b_ub = -U[a]
    A_eq = np.hstack([np.ones((1, m)), np.zeros((1, 1))])          # sum p_j = 1
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * m + [(None, None)], method="highs")
    if res.status == 0 and -res.fun > DOM_TOL:
        p = np.zeros(U.shape[0])
        p[others] = np.clip(res.x[:m], 0, None)
        return p / p.sum(), float(-res.fun)
    return None


def iterated_mixed_elimination(T):
    """Algorithm of the lab sheet:
         1. for every player i and every remaining strategy s_k, check with the LP whether s_k is
            strictly dominated by a mixture of the remaining strategies of i;
         2. if yes: eliminate s_k and start again from step 1;
         3. stop when no elimination is possible.
    Returns (T_red, kept, eliminated, steps):
        T_red      reduced payoff tensor
        kept[i]    indices (in the original game) of the strategies of player i that remain
        eliminated[i]  strategies of player i eliminated, in the order of elimination
        steps      one dict per elimination (player, strategy, dominating mixture, margin)"""
    n = T.ndim - 1
    kept = [list(range(k)) for k in T.shape[:-1]]
    eliminated = [[] for _ in range(n)]
    steps = []
    while True:
        found = False
        for i in range(n):
            if len(kept[i]) < 2:
                continue
            U = payoff_table(T, kept, i)
            for a in range(len(kept[i])):
                b = pure_dominator(U, a)
                if b is not None:
                    p = np.zeros(U.shape[0]); p[b] = 1.0
                    margin, kind = float(np.min(U[b] - U[a])), "pure"
                else:
                    res = mixed_dominator(U, a)
                    if res is None:
                        continue
                    p, margin = res
                    kind = "mixed"
                mixture = np.zeros(T.shape[i])                    # in the ORIGINAL strategy indices
                for local, orig in enumerate(kept[i]):
                    mixture[orig] = p[local]
                strategy = kept[i].pop(a)
                eliminated[i].append(strategy)
                steps.append({"step": len(steps) + 1, "player": i, "strategy": strategy,
                              "mixture": mixture, "margin": margin, "kind": kind,
                              "kept_after": [list(k) for k in kept]})
                found = True
                break
            if found:
                break
        if not found:
            break
    T_red = T[np.ix_(*kept)]
    return T_red, kept, eliminated, steps


# ----------------------------------------------------------------------------
# R4. Nash equilibrium of the reduced game
# ----------------------------------------------------------------------------
def indifference_2x2(A, B):
    """Indifference principle for a 2x2 game, as in the lab sheet.
       1. sigma2 = (y, 1-y):  u1(s1, sigma2) = u1(s1', sigma2)  ->  y
       2. sigma1 = (x, 1-x):  u2(s2, sigma1) = u2(s2', sigma1)  ->  x
       Returns (x, y); a value is None when its equation has no unique solution."""
    d1 = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
    d2 = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
    y = None if abs(d1) < TOL else (A[1, 1] - A[0, 1]) / d1
    x = None if abs(d2) < TOL else (B[1, 1] - B[1, 0]) / d2
    return x, y


def support_enumeration(A, B, tol=1e-9):
    """Indifference principle for any m x n bimatrix game (used for the 2x3 bonus, and as a check
    of the 2x2 case). For every pair of supports (I, J) with |I| = |J| = k:
        - player 2's mix y on J makes player 1 indifferent between all the rows of I,
        - player 1's mix x on I makes player 2 indifferent between all the columns of J,
        - x and y must be > 0 on their supports, and no strategy outside a support may do better.
    Returns the list of equilibria (x, y); pure equilibria are the case k = 1."""
    m, n = A.shape
    found = []
    for k in range(1, min(m, n) + 1):
        for I in itertools.combinations(range(m), k):
            for J in itertools.combinations(range(n), k):
                I_, J_ = list(I), list(J)
                # unknowns (y_J, v1):  A[I,J] y - v1 = 0 ,  sum y = 1
                M1 = np.zeros((k + 1, k + 1)); M1[:k, :k] = A[np.ix_(I_, J_)]; M1[:k, k] = -1; M1[k, :k] = 1
                # unknowns (x_I, v2):  B[I,J]^T x - v2 = 0 ,  sum x = 1
                M2 = np.zeros((k + 1, k + 1)); M2[:k, :k] = B[np.ix_(I_, J_)].T; M2[:k, k] = -1; M2[k, :k] = 1
                rhs = np.zeros(k + 1); rhs[k] = 1
                try:
                    z1, z2 = np.linalg.solve(M1, rhs), np.linalg.solve(M2, rhs)
                except np.linalg.LinAlgError:
                    continue                                   # degenerate support pair: skipped
                y_J, v1, x_I, v2 = z1[:k], z1[k], z2[:k], z2[k]
                if np.any(y_J <= tol) or np.any(x_I <= tol):
                    continue
                x = np.zeros(m); x[I_] = x_I
                y = np.zeros(n); y[J_] = y_J
                if np.all(A @ y <= v1 + 1e-7) and np.all(x @ B <= v2 + 1e-7):
                    if not any(np.allclose(x, fx, atol=1e-7) and np.allclose(y, fy, atol=1e-7)
                               for fx, fy in found):
                        found.append((x, y))
    return found


def pure_nash_profiles(T):
    """Pure-strategy Nash equilibria of an n-player game (every player plays a best response)."""
    masks = [T[..., i] >= T[..., i].max(axis=i, keepdims=True) - TOL for i in range(T.ndim - 1)]
    return [tuple(int(v) for v in p) for p in np.argwhere(np.logical_and.reduce(masks))]


def embed(p_red, kept_i, size_i):
    """Probability vector over the reduced strategies -> vector over the original ones (0 elsewhere)."""
    p = np.zeros(size_i)
    for local, orig in enumerate(kept_i):
        p[orig] = p_red[local]
    return p


# ----------------------------------------------------------------------------
# R7. Lemke-Howson algorithm for a bimatrix game (A, B)
# ----------------------------------------------------------------------------
def lemke_howson(A, B, label=0):
    """Finds one Nash equilibrium (x, y) of the bimatrix game (A, B) by complementary pivoting.
    A[i, j], B[i, j] = payoffs of players 1 and 2 when player 1 plays row i and player 2 column j.

    Method (von Stengel). After adding a constant to make all payoffs positive (this does not
    change the equilibria), consider the two systems
        (1)  A y + r = 1   with y >= 0, r >= 0        (variables y_j, slacks r_i)
        (2)  B^T x + s = 1 with x >= 0, s >= 0        (variables x_i, slacks s_j)
    Labels: x_i and r_i have label i (0..m-1); y_j and s_j have label m+j.
    A pair of basic solutions with all m+n labels present (a variable of each label equal to 0
    somewhere) and (x, y) != (0, 0) gives an equilibrium: x/sum(x), y/sum(y).
    Start at (0, 0), which has all labels. Drop `label`: bring the variable of that label into the
    basis of one system. The variable that leaves has some label l: in the other system bring in the
    variable with label l, and so on. The path ends when the label that was dropped leaves a basis.
    The leaving variable is chosen by the lexicographic minimum-ratio rule, so degenerate games work."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    m, n = A.shape
    c = 1.0 - min(A.min(), B.min())
    A2, B2 = A + c, B + c                       # all payoffs >= 1
    L = m + n
    # Both tableaux have one column per label (0..L-1) plus the right-hand side (last column).
    T1 = np.zeros((m, L + 1)); T1[:, :m] = np.eye(m); T1[:, m:L] = A2; T1[:, L] = 1
    T2 = np.zeros((n, L + 1)); T2[:, :m] = B2.T; T2[:, m:L] = np.eye(n); T2[:, L] = 1
    systems = {1: [T1, list(range(m)), list(range(m))],            # [tableau, basis, identity columns]
               2: [T2, list(range(m, L)), list(range(m, L))]}

    def pivot(system, entering):
        T, basis, id_cols = systems[system]
        col = T[:, entering]
        rows = np.flatnonzero(col > 1e-12)
        if len(rows) == 0:
            raise RuntimeError("Lemke-Howson: unbounded ray (should not happen for positive payoffs).")
        r = min(rows, key=lambda q: tuple(np.round(np.concatenate(([T[q, -1]], T[q, id_cols])) / col[q], 10)))
        leaving = basis[r]
        T[r] /= T[r, entering]
        for q in range(T.shape[0]):
            if q != r:
                T[q] -= T[q, entering] * T[r]
        basis[r] = entering
        return leaving

    k = label
    system = 2 if k < m else 1                  # the variable of label k is nonbasic in that system
    entering = k
    for _ in range(100000):
        leaving = pivot(system, entering)
        if leaving == k:
            break
        entering, system = leaving, 3 - system
    else:
        raise RuntimeError("Lemke-Howson did not terminate.")

    x = np.zeros(m)
    for r, lab in enumerate(systems[2][1]):
        if lab < m:
            x[lab] = systems[2][0][r, -1]
    y = np.zeros(n)
    for r, lab in enumerate(systems[1][1]):
        if lab >= m:
            y[lab - m] = systems[1][0][r, -1]
    return x / x.sum(), y / y.sum()


# ----------------------------------------------------------------------------
# R5. Full pipeline with display
# ----------------------------------------------------------------------------
def solve(T, names, verbose=True):
    """Runs R3 and R4 and prints R5. Returns a dict with the results (used by the tests)."""
    n = T.ndim - 1
    sizes = list(T.shape[:-1])
    out = {"equilibria": [], "message": None}

    if verbose:
        print("=== Original game ===")
        print_game(T, names)
        print("\n=== Reduction by iterated elimination of strictly dominated strategies "
              "(mixed strategies allowed) ===")

    T_red, kept, eliminated, steps = iterated_mixed_elimination(T)
    names_red = [[names[i][k] for k in kept[i]] for i in range(n)]
    out.update(T_red=T_red, kept=kept, eliminated=eliminated, steps=steps)

    if verbose:
        if not steps:
            print("No strategy is strictly dominated (neither by a pure nor by a mixed strategy).")
        for s in steps:
            i, nm = s["player"], names[s["player"]]
            if s["kind"] == "pure":
                how = f"by the pure strategy {mix_str(nm, s['mixture'])}"
            else:
                how = f"by the mixture {mix_str(nm, s['mixture'])}"
            print(f"Step {s['step']}: player {i + 1} eliminates {nm[s['strategy']]}, strictly dominated "
                  f"{how} (smallest gain over {nm[s['strategy']]}: {fr(s['margin'])})")
            rest = ", ".join(f"player {q + 1} = {{{', '.join(names[q][k] for k in s['kept_after'][q])}}}"
                             for q in range(n))
            print(f"  Remaining strategies: {rest}")
            sub = T[np.ix_(*s["kept_after"])]
            print_game(sub, [[names[q][k] for k in s["kept_after"][q]] for q in range(n)])
            print()
        if steps:
            print("No further elimination is possible.")
        print("Eliminated strategies per player: " + "; ".join(
            f"player {i + 1}: " + (", ".join(names[i][k] for k in eliminated[i]) or "none") for i in range(n)))
        print("\n=== Reduced game ===")
        print_game(T_red, names_red)
        print("\n=== Nash equilibrium ===")

    reduced_sizes = [len(k) for k in kept]
    equilibria = []                                    # list of sigma profiles (original strategy space)
    active = [i for i in range(n) if reduced_sizes[i] > 1]      # players who still have a choice

    if not active:
        sig = [embed(np.ones(1), kept[i], sizes[i]) for i in range(n)]
        equilibria.append(sig)
        if verbose:
            print("One strategy per player is left: this profile is directly the Nash equilibrium.")

    elif len(active) <= 2:
        # Two players (at most) still have a choice, the others have a single strategy left:
        # the reduced game is a 2-player game between players p and q (the others are fixed).
        others = [j for j in range(n) if j not in active]
        p, q = sorted(active + others[:2 - len(active)])
        idx = tuple(slice(None) if i in (p, q) else 0 for i in range(n)) + (slice(None),)
        sub = T_red[idx]                                        # shape (k_p, k_q, n)
        A, B = sub[..., p], sub[..., q]                         # payoffs of player p and of player q
        names_p, names_q = names_red[p], names_red[q]
        m_, k_ = A.shape
        if verbose:
            if n > 2:
                fixed = ", ".join(f"player {i + 1} plays {names_red[i][0]}" for i in range(n) if i not in (p, q))
                print(f"Only players {p + 1} and {q + 1} still have a choice ({fixed}): "
                      f"the reduced game is a 2-player game between players {p + 1} and {q + 1}.")
            print(f"  Matrices of the 2-player game: rows = player {p + 1}, columns = player {q + 1}.")
        if (m_, k_) == (2, 2):
            x, y = indifference_2x2(A, B)
            if verbose:
                print("Reduced game 2 x 2: indifference principle.")
                print(f"  Step 1: player {q + 1} plays sigma{q + 1} = (y, 1-y) on ({names_q[0]}, {names_q[1]}); "
                      f"u{p + 1}({names_p[0]}, sigma{q + 1}) = u{p + 1}({names_p[1]}, sigma{q + 1})", end="")
                print(f"  ->  y = {fr(y)}" if y is not None else "  ->  no unique solution")
                print(f"  Step 2: player {p + 1} plays sigma{p + 1} = (x, 1-x) on ({names_p[0]}, {names_p[1]}); "
                      f"u{q + 1}({names_q[0]}, sigma{p + 1}) = u{q + 1}({names_q[1]}, sigma{p + 1})", end="")
                print(f"  ->  x = {fr(x)}" if x is not None else "  ->  no unique solution")
                if x is None or y is None:
                    print("  Step 3: the indifference equation is degenerate (no unique solution), "
                          "so there is no fully mixed equilibrium computed by this principle.")
                elif 0 <= x <= 1 and 0 <= y <= 1:
                    print(f"  Step 3: x = {fr(x)} and y = {fr(y)} are both in [0, 1]: valid.")
                else:
                    print(f"  Step 3: x = {fr(x)} and y = {fr(y)} are not both in [0, 1]: "
                          f"no equilibrium in which both players mix.")
        elif verbose:
            tag = " (bonus)" if (m_, k_) in ((2, 3), (3, 2)) else ""
            print(f"Reduced game {m_} x {k_}{tag}: indifference principle on every pair of supports "
                  f"of equal size.")
        if max(m_, k_) <= 6:
            for x, y in support_enumeration(A, B):
                sig = [embed(x, kept[p], sizes[p]) if i == p else
                       embed(y, kept[q], sizes[q]) if i == q else
                       embed(np.ones(1), kept[i], sizes[i]) for i in range(n)]
                equilibria.append(sig)
        elif verbose:
            print(f"The reduced game is {m_} x {k_}: too large for the support enumeration used here.")
        if not equilibria:
            out["message"] = ("No Nash equilibrium could be determined by the indifference principle "
                              "(the game is probably degenerate).")
            if verbose:
                print(out["message"])
    else:
        pure = pure_nash_profiles(T_red)
        for pr in pure:
            equilibria.append([embed(np.ones(1), [kept[i][pr[i]]], sizes[i]) for i in range(n)])
        out["message"] = (f"{len(active)} players still have several strategies after the reduction: the "
                          f"indifference principle of this lab covers two players only, so no mixed "
                          f"equilibrium is computed.")
        if verbose:
            print(out["message"])
            if pure:
                print("The pure-strategy equilibria of the reduced game (which are also equilibria "
                      "in mixed strategies) are shown below.")
            else:
                print("The reduced game has no pure-strategy equilibrium either.")

    for sig in equilibria:
        pay = expected_payoffs(T, sig)
        ok = is_nash(T, sig)
        out["equilibria"].append({"sigmas": sig, "payoffs": pay, "verified": ok})
        if verbose:
            print()
            for i in range(n):
                print(f"  sigma{i + 1} = {mix_str(names[i], sig[i])}   {vec_str(names[i], sig[i])}")
            print(f"  expected payoffs: ({', '.join(fr(v) for v in pay)})")
            print(f"  checked in the original game: no player gains by deviating -> "
                  f"{'yes, it is a Nash equilibrium' if ok else 'NO (problem)'}")
    return out


def lemke_report(T, names, label=None):
    """R7: runs Lemke-Howson on the bimatrix game (A, B) of a 2-player game."""
    if T.ndim - 1 != 2:
        print("Lemke-Howson needs a 2-player game.")
        return []
    A, B = T[..., 0], T[..., 1]
    m, n = A.shape
    labels = [label] if label is not None else list(range(m + n))
    print("=== Lemke-Howson algorithm on the original game ===")
    found = []
    for k in labels:
        x, y = lemke_howson(A, B, k)
        ok = is_nash(T, [x, y])
        who = names[0][k] if k < m else names[1][k - m]
        print(f"  initial label {k} ({who}):  sigma1 = {mix_str(names[0], x)},  "
              f"sigma2 = {mix_str(names[1], y)},  payoffs ({fr(x @ A @ y)}, {fr(x @ B @ y)}),  "
              f"Nash equilibrium: {'yes' if ok else 'NO (problem)'}")
        if not any(np.allclose(x, fx, atol=1e-7) and np.allclose(y, fy, atol=1e-7) for fx, fy in found):
            found.append((x, y))
    print(f"  {len(found)} distinct equilibrium(s) found with {len(labels)} starting label(s).")
    return found


# ----------------------------------------------------------------------------
# R6. Tests
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))


def run_tests():
    rng = np.random.default_rng(2024)

    def check(name, cond):
        print(f"[{'OK' if cond else 'FAIL'}] {name}")
        assert cond, name

    def close(a, b, tol=1e-7):
        return bool(np.allclose(np.asarray(a, float), np.asarray(b, float), atol=tol))

    # ---- R1: reading the JSON file
    T, names = load_game(os.path.join(HERE, "teacher_game_3players.json"))
    check("R1  teacher's 3-player file gives a tensor of shape (2, 2, 2, 3)", T.shape == (2, 2, 2, 3))
    check("R1  profile '0,1,1' -> payoffs (4, 2, 1)", close(T[0, 1, 1], [4, 2, 1]))
    check("R1  profile '1,0,0' -> payoffs (1, 2, 4)", close(T[1, 0, 0], [1, 2, 4]))

    # ---- R2: expected payoffs, compared with the definition (sum over all pure profiles)
    ok = True
    for _ in range(50):
        n = int(rng.integers(2, 5)); ks = [int(rng.integers(1, 4)) for _ in range(n)]
        G = rng.integers(-5, 10, size=tuple(ks) + (n,)).astype(float)
        sig = [rng.dirichlet(np.ones(k)) for k in ks]
        for i in range(n):
            direct = sum(np.prod([sig[j][s[j]] for j in range(n)]) * G[s][i]
                         for s in itertools.product(*[range(k) for k in ks]))
            ok &= close(expected_payoff(G, i, sig), direct)
            for si in range(ks[i]):           # u_i(s_i, sigma_-i): player i plays the pure strategy si
                pure_i = [np.eye(ks[i])[si] if j == i else sig[j] for j in range(n)]
                ok &= close(payoff_vs_mixed(G, i, sig)[si], expected_payoff(G, i, pure_i))
    check("R2  u_i(sigma) and u_i(s_i, sigma_-i) match the definition on 50 random games (2 to 4 players)", ok)

    # ---- R3 + R4 on the theoretical example of the lab sheet
    T, names = load_game(os.path.join(HERE, "example_theory.json"))
    T_red, kept, elim, steps = iterated_mixed_elimination(T)
    check("R3  example: step 1 eliminates M of player 2, dominated by 1/2 G + 1/2 D (as in the lab sheet)",
          steps[0]["player"] == 1 and steps[0]["strategy"] == 1 and steps[0]["kind"] == "mixed"
          and close(steps[0]["mixture"], [0.5, 0, 0.5]))
    check("R3  example: no pure domination exists in the original game",
          all(pure_dominator(payoff_table(T, [list(range(3))] * 2, i), a) is None for i in range(2) for a in range(3)))
    check("R3  example: step 2 eliminates M of player 1, dominated by 1/3 H + 2/3 B",
          steps[1]["player"] == 0 and steps[1]["strategy"] == 1 and close(steps[1]["mixture"], [1 / 3, 0, 2 / 3]))
    check("R3  example: eliminated = {player 1: [M], player 2: [M]}, reduced game is 2 x 2 (H,B) x (G,D)",
          elim == [[1], [1]] and kept == [[0, 2], [0, 2]] and T_red.shape == (2, 2, 2))
    res = solve(T, names, verbose=False)
    sig = res["equilibria"][0]["sigmas"]
    check("R4  example: sigma1 = (1/4, 0, 3/4), sigma2 = (2/3, 0, 1/3)",
          len(res["equilibria"]) == 1 and close(sig[0], [0.25, 0, 0.75]) and close(sig[1], [2 / 3, 0, 1 / 3]))
    check("R4  example: expected payoffs (2/3, 7/4)", close(res["equilibria"][0]["payoffs"], [2 / 3, 7 / 4]))
    check("R4  example: the equilibrium is verified in the ORIGINAL game", res["equilibria"][0]["verified"])
    x, y = indifference_2x2(T_red[..., 0], T_red[..., 1])
    check("R4  example: indifference principle gives x = 1/4 and y = 2/3", close([x, y], [0.25, 2 / 3]))

    # ---- R6: the teacher's 3-player file (the profile format given in the lab sheet)
    T, names = load_game(os.path.join(HERE, "teacher_game_3players.json"))
    res = solve(T, names, verbose=False)
    check("R6  teacher's file: player 2's B2 is eliminated, strictly dominated by the pure strategy B1",
          len(res["steps"]) == 1 and res["steps"][0]["player"] == 1 and res["steps"][0]["strategy"] == 1
          and res["steps"][0]["kind"] == "pure" and res["kept"] == [[0, 1], [0], [0, 1]])
    e = res["equilibria"][0]
    check("R6  teacher's file: the game reduces to players 1 and 3; sigma1 = (3/4, 1/4), sigma2 = B1, "
          "sigma3 = (1/3, 2/3)",
          len(res["equilibria"]) == 1 and close(e["sigmas"][0], [.75, .25]) and close(e["sigmas"][1], [1, 0])
          and close(e["sigmas"][2], [1 / 3, 2 / 3]))
    check("R6  teacher's file: expected payoffs (5/3, 8/3, 7/4), verified in the original game",
          close(e["payoffs"], [5 / 3, 8 / 3, 7 / 4]) and e["verified"])

    # ---- R6: our own game and the 2x3 bonus game
    T, names = load_game(os.path.join(HERE, "own_game_cyber.json"))
    res = solve(T, names, verbose=False)
    st = res["steps"]
    check("R6  own game: no pure-strategy equilibrium, so the equilibrium has to be mixed",
          pure_nash_profiles(T) == [])
    check("R6  own game: step 1 eliminates Honeypot, dominated only by the MIXTURE 4/7 Firewall + 3/7 Monitoring",
          st[0]["player"] == 0 and st[0]["strategy"] == 2 and st[0]["kind"] == "mixed"
          and close(st[0]["mixture"], [4 / 7, 3 / 7, 0]) and close(st[0]["margin"], 3 / 7))
    check("R6  own game: step 2 eliminates Exploit (pure domination by Phishing); reduced game is 2 x 2",
          len(st) == 2 and st[1]["player"] == 1 and st[1]["strategy"] == 2 and st[1]["kind"] == "pure"
          and res["kept"] == [[0, 1], [0, 1]])
    e = res["equilibria"][0]
    check("R6  own game: sigma1 = (2/3, 1/3, 0), sigma2 = (3/4, 1/4, 0), payoffs (15/2, 3), verified",
          len(res["equilibria"]) == 1 and close(e["sigmas"][0], [2 / 3, 1 / 3, 0])
          and close(e["sigmas"][1], [.75, .25, 0]) and close(e["payoffs"], [7.5, 3]) and e["verified"])
    T, names = load_game(os.path.join(HERE, "bonus_2x3.json"))
    res = solve(T, names, verbose=False)
    e = res["equilibria"][0]
    check("R6  bonus 2 x 3 game: nothing eliminated, one equilibrium sigma1 = (1/2, 1/2), sigma2 = (0, 5/8, 3/8)",
          res["steps"] == [] and len(res["equilibria"]) == 1 and close(e["sigmas"][0], [.5, .5])
          and close(e["sigmas"][1], [0, 5 / 8, 3 / 8]))
    check("R6  bonus 2 x 3 game: payoffs (27/8, 5), verified in the original game",
          close(e["payoffs"], [27 / 8, 5]) and e["verified"])

    # ---- R3: a game solved by pure dominance only (prisoner's dilemma) -> direct equilibrium
    PD = np.array([[(-5, -5), (0, -10)], [(-10, 0), (-1, -1)]], dtype=float)
    res = solve(PD, [["Confess", "Silent"]] * 2, verbose=False)
    check("R3/R4  prisoner's dilemma: one strategy per player left, equilibrium (Confess, Confess)",
          res["kept"] == [[0], [0]] and close(res["equilibria"][0]["payoffs"], [-5, -5]))

    # ---- R3/R4: no domination at all (matching pennies) -> (1/2, 1/2) for both
    MP = np.array([[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]], dtype=float)
    res = solve(MP, [["Heads", "Tails"]] * 2, verbose=False)
    check("R3/R4  matching pennies: nothing eliminated, equilibrium (1/2, 1/2) for both players",
          res["steps"] == [] and close(res["equilibria"][0]["sigmas"][0], [.5, .5])
          and close(res["equilibria"][0]["sigmas"][1], [.5, .5]))

    # ---- R3: NE are preserved by the reduction (random 2-player games, real payoffs = no degeneracy)
    ok = True
    for _ in range(60):
        m, n = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        G = rng.normal(size=(m, n, 2))
        full = support_enumeration(G[..., 0], G[..., 1])
        T_red, kept, elim, steps = iterated_mixed_elimination(G)
        red = support_enumeration(T_red[..., 0], T_red[..., 1]) if min(T_red.shape[:-1]) >= 1 else []
        emb = [(embed(x, kept[0], m), embed(y, kept[1], n)) for x, y in red]
        same = len(full) == len(emb) and all(
            any(close(x, ex) and close(y, ey) for ex, ey in emb) for x, y in full)
        ok &= same
        ok &= all(is_nash(G, [x, y]) for x, y in emb)
    check("R3  the Nash equilibria of the original game are exactly those of the reduced game (60 random games)", ok)

    # ---- R7: Lemke-Howson
    A = np.array([[1., -1], [-1, 1]]); B = -A
    x, y = lemke_howson(A, B, 0)
    check("R7  Lemke-Howson on matching pennies: (1/2, 1/2) for both", close(x, [.5, .5]) and close(y, [.5, .5]))
    A = np.array([[2., 0], [0, 1]]); B = np.array([[1., 0], [0, 2]])        # battle of the sexes
    eqs = {tuple(np.round(np.concatenate(lemke_howson(A, B, k)), 6)) for k in range(4)}
    check("R7  Lemke-Howson on the battle of the sexes: only genuine equilibria are returned",
          all(is_nash(np.stack([A, B], axis=-1), [np.array(e[:2]), np.array(e[2:])]) for e in eqs))
    ok = True
    for _ in range(200):
        m, n = int(rng.integers(1, 7)), int(rng.integers(1, 7))
        A = rng.integers(0, 6, size=(m, n)).astype(float)      # small integers: many ties (degenerate games)
        B = rng.integers(0, 6, size=(m, n)).astype(float)
        G = np.stack([A, B], axis=-1)
        for k in range(m + n):
            x, y = lemke_howson(A, B, k)
            ok &= is_nash(G, [x, y]) and abs(x.sum() - 1) < 1e-9 and abs(y.sum() - 1) < 1e-9
    check("R7  Lemke-Howson returns a Nash equilibrium for every starting label on 200 random "
          "games (1x1 to 6x6, with ties)", ok)
    T, names = load_game(os.path.join(HERE, "example_theory.json"))
    x, y = lemke_howson(T[..., 0], T[..., 1], 0)
    check("R7  Lemke-Howson on the theoretical example finds the equilibrium (1/4, 0, 3/4) / (2/3, 0, 1/3)",
          is_nash(T, [x, y]) and close(x, [.25, 0, .75]) and close(y, [2 / 3, 0, 1 / 3]))
    print("\nAll tests passed.")


# ----------------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Nash equilibrium in mixed strategies")
    parser.add_argument("file", nargs="?", help="JSON file describing the game")
    parser.add_argument("--test", action="store_true", help="run the automatic tests")
    parser.add_argument("--lemke", action="store_true", help="also run Lemke-Howson (2-player games)")
    parser.add_argument("--label", type=int, help="initial dropped label for Lemke-Howson (default: all)")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return
    if not args.file:
        parser.error("give the JSON file of the game, or use --test")
    try:
        T, names = load_game(args.file)
    except (OSError, ValueError, KeyError, TypeError) as err:
        sys.exit(f"Error while reading '{args.file}': {err}")
    solve(T, names)
    if args.lemke:
        print()
        lemke_report(T, names, args.label)


if __name__ == "__main__":
    main()