import sys
import json
import random
import itertools
import numpy as np

#some utility functions
def num_players(game):
    return game.ndim - 1


def default_names(ks):
    letters = "ABCDEFGH"
    return [[f"{letters[i] if i < len(letters) else 'P' + str(i + 1)}{j + 1}"
             for j in range(k)] for i, k in enumerate(ks)]


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:g}"

#display functions
def print_table(game, names, active, fixed):
#player 1 = rows, player 2 = columns, other players(3..n) = fixed
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


#dominance functions
def payoffs_of(game, active, i, s):
# Return the payoffs of player i when playing strategy s, given the active strategies of all players.
    idx = [[s] if p == i else active[p] for p in range(num_players(game))]
    return game[np.ix_(*idx)][..., i].ravel()


def dominates(game, active, i, b, a, mode):
# Return True if strategy b dominates strategy a for player i, given the active strategies of all players.
    ub = payoffs_of(game, active, i, b)
    ua = payoffs_of(game, active, i, a)
    if mode == "strict":
        return bool(np.all(ub > ua))
    return bool(np.all(ub >= ua) and np.any(ub > ua))   # weak


def find_dominated(game, active, mode, player_order, rng=None):
# Return a tuple (i, a, b) if strategy a of player i is dominated by strategy b, or None if no dominated strategy is found.
# Without rng: the first one found, following player_order. With rng: one chosen at random among all dominated strategies.
    candidates = []
    for i in player_order:
        if len(active[i]) <= 1:
            continue
        for a in active[i]:
            for b in active[i]:
                if a != b and dominates(game, active, i, b, a, mode):
                    if rng is None:
                        return i, a, b
                    candidates.append((i, a, b))
                    break
    return rng.choice(candidates) if candidates else None

#iterated elimination of dominated strategies
def iterated_elimination(game, names, mode="strict", player_order=None, verbose=True, rng=None):
    n = num_players(game)
    if player_order is None:
        player_order = list(range(n))
    active = [list(range(game.shape[i])) for i in range(n)]
    step = 0
    while True:
        found = find_dominated(game, active, mode, player_order, rng)
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

#Game input functions
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
# Load a game from a JSON file. The file must contain a "payoffs" field (a list of lists of lists...) and may contain a "names" field (a list of lists of strings).
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

#Test examples from TP pdfS
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

#Main function
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Iterated elimination of dominated strategies")
    parser.add_argument("--test", action="store_true", help="run the tests on examples 1 to 5")
    parser.add_argument("--file", help="load a game from a JSON file")
    parser.add_argument("--mode", choices=["strict", "weak"], help="strict or weak dominance")
    parser.add_argument("--random", action="store_true", help="weak mode: random elimination order")
    parser.add_argument("--seed", type=int, help="seed for --random (to repeat a run)")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    print("=== Iterated elimination of dominated strategies ===")
    if args.file:
        game, names = load_json(args.file)
    else:
        print("Game source: 1) keyboard  2) JSON file  3) random  4) pdf examples")
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
    rng = None
    if mode == "weak":
        use_random = args.random
        if not args.mode:
            use_random = input("Elimination order: 1) default  2) random: ").strip() == "2"
        if use_random:
            seed = args.seed if args.seed is not None else random.randrange(10 ** 6)
            rng = random.Random(seed)
            print(f"(Order followed: random choice among the dominated strategies, seed = {seed})")
        else:
            print(f"(Order followed: players {[i + 1 for i in order]})")

    print("\n=== Initial game ===")
    print_game(game, names, [list(range(k)) for k in game.shape[:-1]])
    print()
    iterated_elimination(game, names, mode, order, rng=rng)


if __name__ == "__main__":
    main()
