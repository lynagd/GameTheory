"""
Game theory lab 3: pure-strategy Nash equilibria of a finite game with n players.

How a game is stored (same convention as the dominance program of TP 1):
    game[s1, s2, ..., sn, i] = payoff of player i (0-indexed) for the profile (s1, ..., sn)
    names[i][j]              = name of strategy j of player i
Player i may have any number of strategies k_i, and n can be any number >= 2.

The program:
    1. displays the game
    2. computes ALL best responses of every player (ties included)
    3. intersects them to get the pure Nash equilibria
    4. checks whether each equilibrium is Pareto-optimal

Usage:  python nash_n.py                       (interactive menu)
        python nash_n.py --test                (tests on the built-in games)
        python nash_n.py --list                (list the built-in games)
        python nash_n.py --example 3           (one of the built-in games)
        python nash_n.py --file game.json      (game loaded from a JSON file)
"""
import json
import argparse
import itertools
import numpy as np

EPS = 1e-9


# ----------------------------------------------------------------------------
# 1. Basic tools
# ----------------------------------------------------------------------------
def num_players(game):
    return game.ndim - 1


def default_names(ks):
    letters = "ABCDEFGH"
    return [[f"{letters[i] if i < len(letters) else 'P' + str(i + 1) + '_'}{j + 1}"
             for j in range(k)] for i, k in enumerate(ks)]


def fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.4g}"


def payoff_str(game, profile, stars=None):
    """(u1, u2, ..., un) for a profile; stars[i] True -> a * after u_i."""
    vals = game[tuple(profile)]
    return "(" + ", ".join(fmt(v) + ("*" if stars is not None and stars[i] else "")
                           for i, v in enumerate(vals)) + ")"


def profile_name(names, profile):
    return "(" + ", ".join(names[i][s] for i, s in enumerate(profile)) + ")"


def profiles_str(names, profiles):
    return "{" + ", ".join(profile_name(names, p) for p in sorted(profiles)) + "}"


# ----------------------------------------------------------------------------
# 2. Display (R2)
# ----------------------------------------------------------------------------
def print_table(rn, cn, cells):
    table = [["1\\2"] + list(cn)] + [[rn[i]] + list(cells[i]) for i in range(len(rn))]
    widths = [max(len(row[k]) for row in table) for k in range(len(table[0]))]
    for row in table:
        print("  " + "  ".join(c.ljust(widths[k]) for k, c in enumerate(row)))


def show_matrices(game, names, stars_masks=None):
    """Player 1 = rows, player 2 = columns; one matrix for each combination of the
    strategies of players 3..n (a single matrix when n = 2).
    stars_masks[i][profile] True -> the payoff of player i gets a * (best response)."""
    n = num_players(game)
    k = game.shape[:-1]
    for fixed in itertools.product(*[range(x) for x in k[2:]]):
        if n > 2:
            label = ", ".join(f"P{p + 3} = {names[p + 2][s]}" for p, s in enumerate(fixed))
            print(f"  [{label}]")
        cells = []
        for r in range(k[0]):
            row = []
            for c in range(k[1]):
                prof = (r, c) + fixed
                stars = None if stars_masks is None else [m[prof] for m in stars_masks]
                row.append(payoff_str(game, prof, stars))
            cells.append(row)
        print_table(names[0], names[1], cells)
        if n > 2:
            print()


# ----------------------------------------------------------------------------
# 3. Best responses (R3) and Nash equilibria (R4)
# ----------------------------------------------------------------------------
def best_response_masks(game):
    """masks[i] is a boolean array with the shape of the game: masks[i][s1..sn] is True
    when s_i is a best response of player i to the strategies of the others in the profile.
    All the maximizers are marked, so ties are kept."""
    masks = []
    for i in range(num_players(game)):
        u = game[..., i]
        best = u.max(axis=i, keepdims=True)         # best payoff for each combination of the others
        masks.append(u >= best - EPS)
    return masks


def best_responses(game, masks, i):
    """List of (others, [best strategies of player i]) for every combination of the
    strategies of the other players. `others` is the tuple of their strategies, in order."""
    k = game.shape[:-1]
    out = []
    for others in itertools.product(*[range(k[p]) for p in range(len(k)) if p != i]):
        idx = list(others)
        idx.insert(i, slice(None))
        out.append((others, [int(s) for s in np.flatnonzero(masks[i][tuple(idx)])]))
    return out


def nash_equilibria(game, masks=None):
    masks = best_response_masks(game) if masks is None else masks
    ne = np.logical_and.reduce(masks)
    return [tuple(int(x) for x in p) for p in np.argwhere(ne)]


# ----------------------------------------------------------------------------
# 4. Pareto analysis (R6)
# ----------------------------------------------------------------------------
def pareto_dominators(game, profile):
    """All profiles of the whole game that Pareto-dominate `profile`:
    every player gets at least as much, and at least one player gets strictly more."""
    u = game[tuple(profile)]
    ge = np.all(game >= u - EPS, axis=-1)
    gt = np.any(game > u + EPS, axis=-1)
    return [tuple(int(x) for x in p) for p in np.argwhere(ge & gt)]


# ----------------------------------------------------------------------------
# 5. Full analysis with display (R5)
# ----------------------------------------------------------------------------
def analyse(game, names, verbose=True):
    """Returns (ne, pareto_optimal_ne): two lists of profiles."""
    n = num_players(game)
    masks = best_response_masks(game)
    ne = nash_equilibria(game, masks)
    opt = [p for p in ne if not pareto_dominators(game, p)]

    if verbose:
        print("\n=== Best responses ===")
        for i in range(n):
            print(f"Player {i + 1} (best strategies against each combination of the others):")
            for others, best in best_responses(game, masks, i):
                label = ", ".join(names[p][s] for p, s in zip([q for q in range(n) if q != i], others))
                print(f"  R{i + 1}({label}) = {{{', '.join(names[i][s] for s in best)}}}")
        print()
        for i in range(n):
            br_i = [tuple(int(x) for x in p) for p in np.argwhere(masks[i])]
            print(f"BR{i + 1} = {profiles_str(names, br_i)}")

        print("\nBest-response matrix (* after u_i: the strategy of player i is a best response)")
        show_matrices(game, names, masks)

        print("=== Pure-strategy Nash equilibria (BR1, ..., BRn intersected) ===")
        if not ne:
            print("No Nash equilibrium in pure strategies exists in this game "
                  "(no profile is a best response for all the players at the same time).")
        else:
            print(f"NE = {profiles_str(names, ne)}")
            for p in ne:
                print(f"  {profile_name(names, p)} -> payoffs {payoff_str(game, p)}")

        print("\n=== Pareto analysis ===")
        if not ne:
            print("Nothing to analyse: there is no Nash equilibrium.")
        else:
            for p in ne:
                doms = pareto_dominators(game, p)
                head = f"{profile_name(names, p)} {payoff_str(game, p)}"
                if not doms:
                    print(f"  {head}: Pareto-optimal")
                else:
                    txt = ", ".join(f"{profile_name(names, d)} {payoff_str(game, d)}"
                                    + (" [Nash equilibrium]" if d in ne else "") for d in doms)
                    print(f"  {head}: Pareto-DOMINATED by {txt}")
            dominated = [p for p in ne if p not in opt]
            print(f"\nPareto-optimal Nash equilibria : {profiles_str(names, opt) if opt else 'none'}")
            print(f"Pareto-dominated Nash equilibria: "
                  f"{profiles_str(names, dominated) if dominated else 'none'}")
    return ne, opt


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
    """JSON file: {"names": [[...], [...], ...], "payoffs": nested array of shape (k1,...,kn,n)}
    payoffs[s1][s2]...[sn] is the list of the n payoffs of the profile (s1, ..., sn)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    game = np.array(data["payoffs"], dtype=float)
    if game.ndim < 3 or game.shape[-1] != game.ndim - 1:
        raise ValueError("payoffs must have shape (k1, ..., kn, n): one payoff per player.")
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
    return rng.integers(lo, hi + 1, size=tuple(ks) + (n,)).astype(float), default_names(ks)


# ----------------------------------------------------------------------------
# 7. Built-in games (R7)
# ----------------------------------------------------------------------------
def build(ks, payoff_fn, names=None):
    """Game from a function profile -> tuple of the n payoffs."""
    game = np.zeros(tuple(ks) + (len(ks),))
    for prof in itertools.product(*[range(k) for k in ks]):
        game[prof] = payoff_fn(prof)
    return game, names or default_names(ks)


def from_array(arr, names=None):
    game = np.array(arr, dtype=float)
    return game, names or default_names(game.shape[:-1])


def ex_course():          # the 2-player course example of TP 2
    return from_array([[(26, 10), (10, 2), (30, 8), (14, 0)],
                       [(2, 2), (2, 2), (22, 8), (22, 8)],
                       [(24, 8), (8, 0), (28, 8), (8, 2)],
                       [(4, 0), (0, 10), (20, 8), (16, 10)]])


def ex_prisoner():
    return from_array([[(-5, -5), (0, -10)], [(-10, 0), (-1, -1)]],
                      [["Confess", "Silent"]] * 2)


def ex_contribution():    # 3 players, example 4 of TP 1 (0 = C contribute, 1 = N do not)
    def pay(prof):
        c = prof.count(0)
        return tuple(2 * c - 3 if s == 0 else 2 * c for s in prof)
    return build([2, 2, 2], pay, [["C", "N"]] * 3)


def ex_cyclic_pennies():  # 3 players, cyclic matching pennies: no pure equilibrium
    def pay(p):
        return (1 if p[0] == p[1] else -1,        # player 1 wants to match player 2
                1 if p[1] == p[2] else -1,        # player 2 wants to match player 3
                1 if p[2] != p[0] else -1)        # player 3 wants to differ from player 1
    return build([2, 2, 2], pay, [["H", "T"]] * 3)


def ex_coordination3():   # 3 players: all Left = (2,2,2), all Right = (1,1,1), otherwise 0
    def pay(p):
        if p == (0, 0, 0):
            return (2, 2, 2)
        if p == (1, 1, 1):
            return (1, 1, 1)
        return (0, 0, 0)
    return build([2, 2, 2], pay, [["Left", "Right"]] * 3)


def ex_volunteer():       # 4 players: volunteer's dilemma, cost 0.4 for the volunteer
    def pay(p):
        out = []
        for i, s in enumerate(p):
            someone_else = any(p[j] == 0 for j in range(4) if j != i)
            if s == 0:
                out.append(0.6)                   # volunteer: gets 1 but pays 0.4
            else:
                out.append(1 if someone_else else 0)
        return tuple(out)
    return build([2, 2, 2, 2], pay, [["Volunteer", "Wait"]] * 4)


def ex_tp1_power():       # the 3-player game of TP 1 (solved there by iterated dominance)
    nm = ["Low", "Medium", "High"]
    return from_array(
        [[[[9, 4, 4], [8, 7, 6], [0, 5, 0]], [[6, 2, 0], [1, 3, 3], [0, 9, 3]], [[3, 6, 1], [6, 3, 3], [3, 3, 0]]],
         [[[2, 5, 1], [3, 7, 7], [6, 4, 5]], [[2, 7, 0], [0, 3, 6], [7, 1, 0]], [[2, 4, 1], [9, 3, 4], [3, 1, 9]]],
         [[[6, 2, 0], [9, 6, 9], [7, 3, 3]], [[3, 9, 4], [9, 9, 7], [7, 0, 5]], [[4, 0, 5], [7, 1, 8], [5, 1, 6]]]],
        [nm] * 3)


def ex_ties3():           # 3 players with 2, 3 and 2 strategies, ties in best responses
    return from_array(
        [[[[3, 1, 0], [2, 0, 2]], [[0, 3, 2], [3, 2, 2]], [[1, 3, 0], [0, 1, 2]]], [[[1, 3, 3], [1, 1, 2]], [[2, 1, 2], [3, 2, 1]], [[1, 1, 0], [0, 3, 1]]]],
        [["X1", "X2"], ["Y1", "Y2", "Y3"], ["Z1", "Z2"]])


# key: (title, builder, expected NE (profiles), expected Pareto-optimal NE)
EXAMPLES = {
    1: ("Course example of TP 2 (2 players, 4x4): 2 NE, one Pareto-dominated, ties", ex_course,
        {(0, 0), (1, 3)}, {(0, 0)}),
    2: ("Prisoner's dilemma (2 players): a single NE, Pareto-dominated", ex_prisoner,
        {(0, 0)}, set()),
    3: ("Contribution game (3 players): a single NE, Pareto-dominated", ex_contribution,
        {(1, 1, 1)}, set()),
    4: ("Cyclic matching pennies (3 players): no pure NE", ex_cyclic_pennies, set(), set()),
    5: ("Coordination (3 players): 2 NE, one Pareto-dominated by the other", ex_coordination3,
        {(0, 0, 0), (1, 1, 1)}, {(0, 0, 0)}),
    6: ("Volunteer's dilemma (4 players): 4 NE, all Pareto-optimal", ex_volunteer,
        {(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)},
        {(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)}),
    7: ("TP 1 game (3 players, 3x3x3): unique Pareto-optimal NE, found by iterated dominance", ex_tp1_power,
        {(2, 1, 1)}, {(2, 1, 1)}),
    8: ("Different strategy counts (3 players, 2x3x2) with ties: 2 NE, one Pareto-dominated", ex_ties3,
        {(0, 1, 1), (1, 2, 1)}, {(0, 1, 1)}),
}


def run_tests():
    def check(name, got, expected):
        ok = got == expected
        print(f"[{'OK' if ok else 'FAIL'}] {name}")
        assert ok, f"{name}: got {got}, expected {expected}"

    for k, (title, build_game, exp_ne, exp_opt) in EXAMPLES.items():
        game, names = build_game()
        ne, opt = analyse(game, names, verbose=False)
        check(f"Game {k} Nash equilibria   : {title}", set(ne), exp_ne)
        check(f"Game {k} Pareto-optimal NE : {title}", set(opt), exp_opt)

    # ties: in game 8 some players have several best responses to the same strategies of the others
    game, names = ex_ties3()
    masks = best_response_masks(game)
    ties = [(i, others, best) for i in range(3) for others, best in best_responses(game, masks, i)
            if len(best) > 1]
    check("Game 8 has best responses with several strategies", len(ties) > 0, True)
    # the 4-player game has 2^4 = 16 profiles and 4 equilibria
    game, names = ex_volunteer()
    check("Volunteer's dilemma has 16 profiles", game.shape, (2, 2, 2, 2, 4))
    # for n = 2 the program gives the sets of the lab sheet of TP 2
    game, names = ex_course()
    masks = best_response_masks(game)
    br = [{tuple(int(x) for x in p) for p in np.argwhere(m)} for m in masks]
    check("Course example BR1", br[0], {(0, 0), (0, 1), (0, 2), (1, 3)})
    check("Course example BR2", br[1], {(0, 0), (1, 2), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)})
    print("\nAll tests passed.")


# ----------------------------------------------------------------------------
# 8. Main program
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pure-strategy Nash equilibria of an n-player game")
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

    print("=== Pure-strategy Nash equilibria (n players) ===")
    if args.file:
        game, names = load_json(args.file)
    elif args.example:
        game, names = EXAMPLES[args.example][1]()
        print(f"Built-in game {args.example}: {EXAMPLES[args.example][0]}")
    else:
        print("Game source: 1) keyboard  2) JSON file  3) random  4) built-in example")
        choice = read_int("Your choice: ", 1)
        if choice == 1:
            game, names = input_keyboard()
        elif choice == 2:
            game, names = input_file()
        elif choice == 3:
            game, names = input_random()
        else:
            for k, (title, *_rest) in EXAMPLES.items():
                print(f"  {k}) {title}")
            game, names = EXAMPLES[read_int("Game number: ", 1)][1]()

    print(f"\n=== Game ({num_players(game)} players) ===")
    print("Each cell shows (u1, u2, ..., un). Player 1 = rows, player 2 = columns"
          + (", one matrix per combination of players 3..n." if num_players(game) > 2 else "."))
    show_matrices(game, names)
    analyse(game, names)


if __name__ == "__main__":
    main()
