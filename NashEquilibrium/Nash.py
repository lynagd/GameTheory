"""
Game theory lab 2: pure-strategy Nash equilibria of a two-player finite game.

How a game is stored:
    A[i, j]  = payoff of player 1 when player 1 plays row i and player 2 plays column j
    B[i, j]  = payoff of player 2 for the same profile
    row_names[i], col_names[j] = names of the strategies

The program:
    1. displays the game (matrix of payoff pairs, matrix A, matrix B)
    2. computes ALL best responses of each player (ties included)
    3. intersects them to get the pure Nash equilibria
    4. checks whether each equilibrium is Pareto-optimal

Usage:  python nash.py                           (interactive menu)
        python nash.py --test                    (tests on the example games)
        python nash.py --example 1               (one of the built-in games)
        python nash.py --file game.json          (game loaded from a JSON file)
        python nash.py --list                    (list the built-in games)
"""
import sys
import json
import argparse
import numpy as np


# ----------------------------------------------------------------------------
# 1. Basic tools
# ----------------------------------------------------------------------------
def default_names(m, n):
    return [f"A{i + 1}" for i in range(m)], [f"B{j + 1}" for j in range(n)]


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def pair(A, B, i, j):
    return f"({fmt(A[i, j])}, {fmt(B[i, j])})"


def profile_name(rn, cn, i, j):
    return f"({rn[i]}, {cn[j]})"


def profiles_str(profiles, rn, cn):
    """{(A1, B1), (A2, B4)} for a set of (i, j) profiles."""
    return "{" + ", ".join(profile_name(rn, cn, i, j) for i, j in sorted(profiles)) + "}"


# ----------------------------------------------------------------------------
# 2. Display (R2)
# ----------------------------------------------------------------------------
def print_matrix(corner, rn, cn, cells):
    """cells[i][j] = text of the cell. Prints an aligned table."""
    table = [[corner] + list(cn)] + [[rn[i]] + list(cells[i]) for i in range(len(rn))]
    widths = [max(len(row[k]) for row in table) for k in range(len(table[0]))]
    for row in table:
        print("  " + "  ".join(c.ljust(widths[k]) for k, c in enumerate(row)))


def show_game(A, B, rn, cn):
    m, n = A.shape
    print("Payoff pairs (u1, u2): player 1 = rows, player 2 = columns")
    print_matrix("1\\2", rn, cn, [[pair(A, B, i, j) for j in range(n)] for i in range(m)])
    print("\nMatrix A (payoffs of player 1)")
    print_matrix("A", rn, cn, [[fmt(A[i, j]) for j in range(n)] for i in range(m)])
    print("\nMatrix B (payoffs of player 2)")
    print_matrix("B", rn, cn, [[fmt(B[i, j]) for j in range(n)] for i in range(m)])


# ----------------------------------------------------------------------------
# 3. Best responses (R3) and Nash equilibria (R4)
# ----------------------------------------------------------------------------
def best_responses_1(A):
    """R1[j] = list of rows i maximizing A[i, j] (all of them, ties included)."""
    return [list(np.flatnonzero(np.isclose(A[:, j], A[:, j].max()))) for j in range(A.shape[1])]


def best_responses_2(B):
    """R2[i] = list of columns j maximizing B[i, j] (all of them, ties included)."""
    return [list(np.flatnonzero(np.isclose(B[i, :], B[i, :].max()))) for i in range(B.shape[0])]


def best_response_profiles(A, B):
    """BR1 and BR2 as sets of profiles (i, j)."""
    br1 = {(i, j) for j, rows in enumerate(best_responses_1(A)) for i in rows}
    br2 = {(i, j) for i, cols in enumerate(best_responses_2(B)) for j in cols}
    return br1, br2


def nash_equilibria(A, B):
    br1, br2 = best_response_profiles(A, B)
    return sorted(br1 & br2)


# ----------------------------------------------------------------------------
# 4. Pareto analysis (R6)
# ----------------------------------------------------------------------------
def pareto_dominates(u, v):
    """True if payoff vector u Pareto-dominates v: >= for both players, > for at least one."""
    return bool(u[0] >= v[0] and u[1] >= v[1] and (u[0] > v[0] or u[1] > v[1]))


def pareto_dominators(A, B, i, j):
    """All profiles of the whole game that Pareto-dominate the profile (i, j)."""
    m, n = A.shape
    u = (A[i, j], B[i, j])
    return [(k, l) for k in range(m) for l in range(n)
            if (k, l) != (i, j) and pareto_dominates((A[k, l], B[k, l]), u)]


# ----------------------------------------------------------------------------
# 5. Full analysis with display (R5)
# ----------------------------------------------------------------------------
def analyse(A, B, rn, cn, verbose=True):
    """Returns (ne, pareto_optimal_ne): two lists of (i, j) profiles."""
    m, n = A.shape
    r1, r2 = best_responses_1(A), best_responses_2(B)
    br1, br2 = best_response_profiles(A, B)
    ne = sorted(br1 & br2)
    pareto_opt = [p for p in ne if not pareto_dominators(A, B, *p)]

    if verbose:
        print("\n=== Best responses ===")
        print("Player 1 (best row(s) against each column of player 2):")
        for j in range(n):
            print(f"  R1({cn[j]}) = {{{', '.join(rn[i] for i in r1[j])}}}")
        print("Player 2 (best column(s) against each row of player 1):")
        for i in range(m):
            print(f"  R2({rn[i]}) = {{{', '.join(cn[j] for j in r2[i])}}}")
        print(f"\nBR1 = {profiles_str(br1, rn, cn)}")
        print(f"BR2 = {profiles_str(br2, rn, cn)}")

        print("\nBest-response matrix (* marks a best response of the player concerned)")
        cells = [[f"({fmt(A[i, j])}{'*' if (i, j) in br1 else ''}, "
                  f"{fmt(B[i, j])}{'*' if (i, j) in br2 else ''})" for j in range(n)]
                 for i in range(m)]
        print_matrix("1\\2", rn, cn, cells)

        print("\n=== Pure-strategy Nash equilibria (BR1 and BR2 intersected) ===")
        if not ne:
            print("No Nash equilibrium in pure strategies exists in this game "
                  "(no profile is a best response for both players at the same time).")
        else:
            print(f"NE = {profiles_str(ne, rn, cn)}")
            for i, j in ne:
                print(f"  {profile_name(rn, cn, i, j)} -> payoffs {pair(A, B, i, j)}")

        print("\n=== Pareto analysis ===")
        if not ne:
            print("Nothing to analyse: there is no Nash equilibrium.")
        else:
            for i, j in ne:
                doms = pareto_dominators(A, B, i, j)
                head = f"{profile_name(rn, cn, i, j)} {pair(A, B, i, j)}"
                if not doms:
                    print(f"  {head}: Pareto-optimal")
                else:
                    txt = ", ".join(
                        f"{profile_name(rn, cn, k, l)} {pair(A, B, k, l)}"
                        + (" [Nash equilibrium]" if (k, l) in ne else "")
                        for k, l in doms)
                    print(f"  {head}: Pareto-DOMINATED by {txt}")
            dominated = [p for p in ne if p not in pareto_opt]
            print(f"\nPareto-optimal Nash equilibria : "
                  f"{profiles_str(pareto_opt, rn, cn) if pareto_opt else 'none'}")
            print(f"Pareto-dominated Nash equilibria: "
                  f"{profiles_str(dominated, rn, cn) if dominated else 'none'}")
    return ne, pareto_opt


# ----------------------------------------------------------------------------
# 6. Game input (R1)
# ----------------------------------------------------------------------------
def read_int(msg, lo=1):
    while True:
        try:
            v = int(input(msg))
            if v >= lo:
                return v
        except ValueError:
            pass
        print(f"  Enter an integer >= {lo}.")


def input_keyboard():
    m = read_int("Number of strategies of player 1 (m): ")
    n = read_int("Number of strategies of player 2 (n): ")
    rn, cn = default_names(m, n)
    for label, names in (("1", rn), ("2", cn)):
        s = input(f"Strategy names of player {label} "
                  f"(separated by spaces, Enter = {' '.join(names)}): ").split()
        if len(s) == len(names):
            names[:] = s
    A, B = np.zeros((m, n)), np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            while True:
                try:
                    vals = [float(x) for x in
                            input(f"Payoffs (u1 u2) for ({rn[i]}, {cn[j]}): ").split()]
                    if len(vals) == 2:
                        break
                except ValueError:
                    pass
                print("  Enter exactly 2 numbers.")
            A[i, j], B[i, j] = vals
    return A, B, rn, cn


def load_json(path):
    """JSON file: {"names": [[row names], [column names]],
                   "payoffs": [[[u1, u2], ...], ...]}      (m x n x 2)
       or, instead of "payoffs":  "A": [[...]], "B": [[...]]   (two m x n matrices)"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if "payoffs" in data:
        p = np.array(data["payoffs"], dtype=float)
        A, B = p[..., 0], p[..., 1]
    else:
        A, B = np.array(data["A"], dtype=float), np.array(data["B"], dtype=float)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError("A and B must be two matrices of the same size m x n.")
    names = data.get("names")
    rn, cn = (names[0], names[1]) if names else default_names(*A.shape)
    return A, B, rn, cn


def input_file():
    return load_json(input("Path of the JSON file: ").strip())


def input_random():
    m = read_int("Number of strategies of player 1 (m): ")
    n = read_int("Number of strategies of player 2 (n): ")
    lo = read_int("Minimum payoff: ", -10 ** 6)
    hi = read_int("Maximum payoff: ", lo)
    rng = np.random.default_rng()
    A = rng.integers(lo, hi + 1, size=(m, n)).astype(float)
    B = rng.integers(lo, hi + 1, size=(m, n)).astype(float)
    return A, B, *default_names(m, n)


# ----------------------------------------------------------------------------
# 7. Built-in games (R7). Each game is built from a matrix of payoff pairs.
# ----------------------------------------------------------------------------
def from_pairs(pairs, rn=None, cn=None):
    p = np.array(pairs, dtype=float)
    A, B = p[..., 0], p[..., 1]
    d_rn, d_cn = default_names(*A.shape)
    return A, B, rn or d_rn, cn or d_cn


def ex_course():
    return from_pairs([[(26, 10), (10, 2), (30, 8), (14, 0)],
                       [(2, 2), (2, 2), (22, 8), (22, 8)],
                       [(24, 8), (8, 0), (28, 8), (8, 2)],
                       [(4, 0), (0, 10), (20, 8), (16, 10)]])


def ex_prisoner():
    return from_pairs([[(-5, -5), (0, -10)], [(-10, 0), (-1, -1)]],
                      ["Confess", "Stay silent"], ["Confess", "Stay silent"])


def ex_battle():
    return from_pairs([[(2, 1), (0, 0)], [(0, 0), (1, 2)]],
                      ["Opera", "Football"], ["Opera", "Football"])


def ex_pennies():
    return from_pairs([[(1, -1), (-1, 1)], [(-1, 1), (1, -1)]],
                      ["Heads", "Tails"], ["Heads", "Tails"])


def ex_rps():
    return from_pairs([[(0, 0), (-1, 1), (1, -1)],
                       [(1, -1), (0, 0), (-1, 1)],
                       [(-1, 1), (1, -1), (0, 0)]],
                      ["Rock", "Paper", "Scissors"], ["Rock", "Paper", "Scissors"])


def ex_stag():
    return from_pairs([[(4, 4), (0, 3)], [(3, 0), (2, 2)]],
                      ["Stag", "Hare"], ["Stag", "Hare"])


def ex_ties():
    return from_pairs([[(2, 1), (2, 1), (2, 0)], [(2, 1), (1, 0), (2, 0)]],
                      ["H", "B"], ["G", "M", "D"])


def ex_dominance():
    return from_pairs([[(4, 3), (5, 1), (6, 2)],
                       [(2, 1), (8, 4), (3, 6)],
                       [(3, 0), (9, 6), (2, 8)]],
                      ["H", "M", "B"], ["G", "M", "D"])


# key: (title, builder, expected NE as 0-indexed (i, j), expected Pareto-optimal NE)
EXAMPLES = {
    1: ("Course example of the lab sheet (4x4): 2 NE, one Pareto-dominated, ties", ex_course,
        {(0, 0), (1, 3)}, {(0, 0)}),
    2: ("Prisoner's dilemma: a single NE, Pareto-dominated", ex_prisoner,
        {(0, 0)}, set()),
    3: ("Battle of the sexes: 2 NE, both Pareto-optimal", ex_battle,
        {(0, 0), (1, 1)}, {(0, 0), (1, 1)}),
    4: ("Matching pennies: no pure NE", ex_pennies, set(), set()),
    5: ("Rock-paper-scissors (3x3): no pure NE", ex_rps, set(), set()),
    6: ("Stag hunt: 2 NE, one Pareto-dominated by the other", ex_stag,
        {(0, 0), (1, 1)}, {(0, 0)}),
    7: ("Game with ties for both players: 3 NE, all Pareto-optimal", ex_ties,
        {(0, 0), (0, 1), (1, 0)}, {(0, 0), (0, 1), (1, 0)}),
    8: ("Chapter 2 dominance example (3x3): unique NE, Pareto-dominated", ex_dominance,
        {(0, 0)}, set()),
}


def run_tests():
    def check(name, got, expected):
        ok = got == expected
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
        assert ok, f"{name}: got {got}, expected {expected}"

    for k, (title, build, exp_ne, exp_opt) in EXAMPLES.items():
        A, B, rn, cn = build()
        ne, opt = analyse(A, B, rn, cn, verbose=False)
        check(f"Game {k} Nash equilibria   : {title}", set(ne), exp_ne)
        check(f"Game {k} Pareto-optimal NE : {title}", set(opt), exp_opt)

    # best responses with ties (game 7): player 1 is indifferent between H and B against G and D
    A, B, rn, cn = ex_ties()
    check("Game 7 R1(G) has two best responses", best_responses_1(A)[0], [0, 1])
    check("Game 7 R2(H) has two best responses", best_responses_2(B)[0], [0, 1])
    # the course example: BR1 and BR2 exactly as written in the lab sheet
    A, B, rn, cn = ex_course()
    br1, br2 = best_response_profiles(A, B)
    check("Course example BR1", br1, {(0, 0), (0, 1), (0, 2), (1, 3)})
    check("Course example BR2", br2, {(0, 0), (1, 2), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)})
    print("\nAll tests passed.")


# ----------------------------------------------------------------------------
# 8. Main program
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pure-strategy Nash equilibria of a 2-player game")
    parser.add_argument("--test", action="store_true", help="run the tests on the built-in games")
    parser.add_argument("--list", action="store_true", help="list the built-in games")
    parser.add_argument("--example", type=int, help="analyse a built-in game (number)")
    parser.add_argument("--file", help="load a game from a JSON file")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return
    if args.list:
        for k, (title, *_rest) in EXAMPLES.items():
            print(f"  {k}) {title}")
        return

    print("=== Pure-strategy Nash equilibria ===")
    if args.file:
        A, B, rn, cn = load_json(args.file)
    elif args.example:
        A, B, rn, cn = EXAMPLES[args.example][1]()
        print(f"Built-in game {args.example}: {EXAMPLES[args.example][0]}")
    else:
        print("Game source: 1) keyboard  2) JSON file  3) random  4) built-in example")
        choice = read_int("Your choice: ", 1)
        if choice == 1:
            A, B, rn, cn = input_keyboard()
        elif choice == 2:
            A, B, rn, cn = input_file()
        elif choice == 3:
            A, B, rn, cn = input_random()
        else:
            for k, (title, *_rest) in EXAMPLES.items():
                print(f"  {k}) {title}")
            k = read_int("Game number: ", 1)
            A, B, rn, cn = EXAMPLES[k][1]()

    print("\n=== Game ===")
    show_game(A, B, rn, cn)
    analyse(A, B, rn, cn)


if __name__ == "__main__":
    main()
