"""
Game theory lab: iterated elimination of dominated strategies (strict / weak).

How a game is stored:
    game[s1, s2, ..., sn, i] = payoff of player i (0-indexed) for the profile (s1, ..., sn)
    names[i][j]              = name of strategy j of player i
    active[i]                = indices of the strategies still in the game for player i

Nothing is ever deleted from the numpy array: we only remove indices from `active`.

Usage:  python dominance.py                                      (interactive menu)
        python dominance.py --test                               (tests on examples 1 to 5)
        python dominance.py --file game.json --mode strict       (game loaded from a file)
"""
import sys
import json
import itertools
import numpy as np


# ----------------------------------------------------------------------------
# 1. Basic tools
# ----------------------------------------------------------------------------
def num_players(game):
    return game.ndim - 1


def default_names(ks):
    letters = "ABCDEFGH"
    return [[f"{letters[i] if i < len(letters) else 'P' + str(i + 1)}{j + 1}"
             for j in range(k)] for i, k in enumerate(ks)]


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


# ----------------------------------------------------------------------------
# 2. Display (R2)
# ----------------------------------------------------------------------------
def print_table(game, names, active, fixed):
    """Print one matrix: player 1 in rows, player 2 in columns.
    `fixed` = indices (in game) of the strategies of players 3..n."""
    rows, cols = active[0], active[1]
    header = ["1\\2"] + [names[1][c] for c in cols]
    body = []
    for r in rows:
        line = [names[0][r]]
        for c in cols:
            payoff = game[(r, c) + tuple(fixed)]
            line.append("(" + ", ".join(fmt(v) for v in payoff) + ")")
        body.append(line)
    table = [header] + body
    widths = [max(len(row[j]) for row in table) for j in range(len(header))]
    for row in table:
        print("  " + "  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))


def print_game(game, names, active):
    n = num_players(game)
    if n == 1:
        for s in active[0]:
            print(f"  {names[0][s]} : ({fmt(game[s, 0])})")
        return
    if n == 2:
        print_table(game, names, active, [])
        return
    # 3 players or more: one matrix for each combination of players 3..n
    for combo in itertools.product(*active[2:]):
        label = ", ".join(f"P{p + 3} = {names[p + 2][s]}" for p, s in enumerate(combo))
        print(f"  [{label}]")
        print_table(game, names, active, combo)
        print()


# ----------------------------------------------------------------------------
# 3. Dominance (R3, R4)
# ----------------------------------------------------------------------------
def payoffs_of(game, active, i, s):
    """Payoffs of player i when playing s, against every active combination of the others."""
    idx = [[s] if p == i else active[p] for p in range(num_players(game))]
    return game[np.ix_(*idx)][..., i].ravel()


def dominates(game, active, i, b, a, mode):
    """True if strategy b dominates strategy a for player i."""
    ub = payoffs_of(game, active, i, b)
    ua = payoffs_of(game, active, i, a)
    if mode == "strict":
        return bool(np.all(ub > ua))
    return bool(np.all(ub >= ua) and np.any(ub > ua))   # weak


def find_dominated(game, active, mode, player_order):
    """First dominated strategy found: (player, dominated, dominating) or None."""
    for i in player_order:
        if len(active[i]) <= 1:
            continue
        for a in active[i]:
            for b in active[i]:
                if a != b and dominates(game, active, i, b, a, mode):
                    return i, a, b
    return None


# ----------------------------------------------------------------------------
# 4. Iterated elimination with trace (R5, R6)
# ----------------------------------------------------------------------------
def iterated_elimination(game, names, mode="strict", player_order=None, verbose=True):
    n = num_players(game)
    if player_order is None:
        player_order = list(range(n))
    active = [list(range(game.shape[i])) for i in range(n)]
    step = 0
    while True:
        found = find_dominated(game, active, mode, player_order)
        if found is None:
            break
        i, a, b = found
        step += 1
        active[i].remove(a)
        if verbose:
            kind = "strictly" if mode == "strict" else "weakly"
            print(f"Step {step}: player {i + 1} removes {names[i][a]} "
                  f"({kind} dominated by {names[i][b]})")
            print_game(game, names, active)
            print()
    if verbose:
        if step == 0:
            print("No elimination is possible: the game is unchanged.\n")
        print("=== Final result ===")
        print_game(game, names, active)
        if all(len(s) == 1 for s in active):
            profile = tuple(active[i][0] for i in range(n))
            pay = ", ".join(fmt(v) for v in game[profile])
            prof = ", ".join(names[i][profile[i]] for i in range(n))
            print(f"\nA single profile remains: ({prof}) with payoffs ({pay}).")
        else:
            remaining = " x ".join("{" + ", ".join(names[i][s] for s in active[i]) + "}"
                                   for i in range(n))
            print(f"\nSeveral profiles remain: {remaining}. The game is not completely "
                  f"solved by this method.")
    return active


# ----------------------------------------------------------------------------
# 5. Game input (R1)
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
    n = read_int("Number of players: ", 2)
    ks = [read_int(f"Number of strategies of player {i + 1}: ") for i in range(n)]
    names = default_names(ks)
    for i in range(n):
        s = input(f"Strategy names of player {i + 1} "
                  f"(separated by spaces, Enter = {' '.join(names[i])}): ").split()
        if len(s) == ks[i]:
            names[i] = s
    game = np.zeros(tuple(ks) + (n,))
    for idx in np.ndindex(*ks):
        label = ", ".join(names[i][s] for i, s in enumerate(idx))
        while True:
            try:
                vals = [float(x) for x in input(f"Payoffs for ({label}) [{n} values]: ").split()]
                if len(vals) == n:
                    break
            except ValueError:
                pass
            print(f"  Enter exactly {n} numbers.")
        game[idx] = vals
    return game, names


def load_json(path):
    """JSON file: {"names": [[...], [...]], "payoffs": nested array of shape (k1,...,kn,n)}"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    game = np.array(data["payoffs"], dtype=float)
    names = data.get("names") or default_names(game.shape[:-1])
    return game, names


def input_file():
    return load_json(input("Path of the JSON file: ").strip())


def input_random():
    n = read_int("Number of players: ", 2)
    ks = [read_int(f"Number of strategies of player {i + 1}: ") for i in range(n)]
    lo = read_int("Minimum payoff: ", -10 ** 6)
    hi = read_int("Maximum payoff: ", lo)
    rng = np.random.default_rng()
    game = rng.integers(lo, hi + 1, size=tuple(ks) + (n,)).astype(float)
    return game, default_names(ks)


# ----------------------------------------------------------------------------
# 6. Test games (examples 1 to 5 of the lab sheet)
# ----------------------------------------------------------------------------
def example1():
    g = np.array([[(3, 1), (2, 3), (0, 0)],
                  [(1, 2), (1, 2), (5, 1)],
                  [(2, 2), (0, 1), (1, 0)]], dtype=float)
    return g, [["H", "M", "B"], ["G", "C", "D"]]


def example2():
    g = np.array([[(0, 0), (-1, 1), (1, -1)],
                  [(1, -1), (0, 0), (-1, 1)],
                  [(-1, 1), (1, -1), (0, 0)]], dtype=float)
    return g, [["Rock", "Paper", "Scissors"]] * 2


def example3():
    g = np.array([[(2, 1), (2, 1), (2, 0)],
                  [(2, 1), (1, 0), (2, 0)]], dtype=float)
    return g, [["H", "B"], ["G", "M", "D"]]


def example4():
    names = [["C", "N"]] * 3
    g = np.zeros((2, 2, 2, 3))
    for prof in itertools.product([0, 1], repeat=3):      # 0 = C, 1 = N
        c = prof.count(0)
        for i in range(3):
            g[prof + (i,)] = 2 * c - 3 if prof[i] == 0 else 2 * c
    return g, names


def example5():
    g = np.array([[(10, 10), (4, 9), (4, 11)],
                  [(5, 6), (8, 3), (3, 4)],
                  [(4, 0), (8, 1), (5, 3)]], dtype=float)
    return g, [["A1", "A2", "A3"], ["B1", "B2", "B3"]]


EXAMPLES = {1: example1, 2: example2, 3: example3, 4: example4, 5: example5}


def run_tests():
    def check(name, got, expected):
        ok = got == expected
        print(f"[{'OK' if ok else 'FAIL'}] {name} -> {got}")
        assert ok, f"{name}: expected {expected}"

    quiet = dict(verbose=False)
    g, n = example1(); check("Ex1 strict", iterated_elimination(g, n, "strict", **quiet), [[0], [1]])
    g, n = example2(); check("Ex2 strict", iterated_elimination(g, n, "strict", **quiet), [[0, 1, 2]] * 2)
    g, n = example2(); check("Ex2 weak", iterated_elimination(g, n, "weak", **quiet), [[0, 1, 2]] * 2)
    g, n = example3(); check("Ex3 strict", iterated_elimination(g, n, "strict", **quiet), [[0, 1], [0, 1]])
    g, n = example3(); check("Ex3 weak (player 1 first)", iterated_elimination(g, n, "weak", **quiet), [[0], [0, 1]])
    g, n = example3(); check("Ex3 weak (player 2 first)", iterated_elimination(g, n, "weak", [1, 0], **quiet), [[0, 1], [0]])
    g, n = example4(); check("Ex4 strict", iterated_elimination(g, n, "strict", **quiet), [[1], [1], [1]])
    g, n = example5(); check("Ex5 strict", iterated_elimination(g, n, "strict", **quiet), [[2], [2]])
    g, n = example5(); check("Ex5 weak", iterated_elimination(g, n, "weak", **quiet), [[2], [2]])
    print("\nAll tests passed.")


# ----------------------------------------------------------------------------
# 7. Main program
# ----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Iterated elimination of dominated strategies")
    parser.add_argument("--test", action="store_true", help="run the tests on examples 1 to 5")
    parser.add_argument("--file", help="load a game from a JSON file")
    parser.add_argument("--mode", choices=["strict", "weak"], help="strict or weak dominance")
    parser.add_argument("--reverse", action="store_true",
                        help="weak mode: process the players in reverse order")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    print("=== Iterated elimination of dominated strategies ===")
    if args.file:
        game, names = load_json(args.file)
    else:
        print("Game source: 1) keyboard  2) JSON file  3) random  4) lab sheet example")
        choice = read_int("Your choice: ", 1)
        if choice == 1:
            game, names = input_keyboard()
        elif choice == 2:
            game, names = input_file()
        elif choice == 3:
            game, names = input_random()
        else:
            k = read_int("Example number (1 to 5): ", 1)
            game, names = EXAMPLES[k]()

    if args.mode:
        mode = args.mode
    else:
        mode = "weak" if input("Mode: 1) strict dominance  2) weak dominance: ").strip() == "2" else "strict"

    order = list(range(num_players(game)))
    if mode == "weak":
        reverse = args.reverse or (not args.mode and
                                   input("Player order: 1) normal  2) reversed: ").strip() == "2")
        if reverse:
            order.reverse()
        print(f"(Order followed: players {[i + 1 for i in order]})")

    print("\n=== Initial game ===")
    print_game(game, names, [list(range(k)) for k in game.shape[:-1]])
    print()
    iterated_elimination(game, names, mode, order)


if __name__ == "__main__":
    main()
