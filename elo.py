import capture
import random
import pickle as pkl
import matplotlib.pyplot as plt
import os

def update_elo(winner_elo, loser_elo, point_diff=100, odds_diff=2, K=1):
    # Elo update using odds logic
    winner_elo += K * (1 - ((odds_diff ** (winner_elo / point_diff)) /
                     ((odds_diff ** (winner_elo / point_diff)) + (odds_diff ** (loser_elo / point_diff)))))
    loser_elo += K * (0 - ((odds_diff ** (loser_elo / point_diff)) /
                    ((odds_diff ** (winner_elo / point_diff)) + (odds_diff ** (loser_elo / point_diff)))))
    return winner_elo, loser_elo

def run_game(p1, p1_elo, p2, p2_elo):
    options = capture.readCommand(['-q', "-r", p1, "-b", p2])
    games = capture.runGames(**options)
    score = games[0].state.data.score
    if score > 0:
        print('p1 victory')
        p1_elo, p2_elo = update_elo(p1_elo, p2_elo)
    elif score < 0:
        print('p2 victory')
        p2_elo, p1_elo = update_elo(p2_elo, p1_elo)
    return p1_elo, p2_elo

def calculate_odds(p1_elo, p2_elo, point_diff=100, odds_diff=2):
    return (odds_diff ** (p1_elo / point_diff)) / ((odds_diff ** (p1_elo / point_diff)) + (odds_diff ** (p2_elo / point_diff)))

def elo_tournament(players, games_to_play=-1, elo_scores=None, elo_histories_existing=None):
    if games_to_play == -1:
        games_to_play = 50 * len(players)

    if elo_scores is None:
        elo_scores = {p: 500 for p in players}
    if elo_histories_existing is None:
        elo_histories = [[500] for _ in players]
    else:
        elo_histories = elo_histories_existing

    already_played = len(elo_histories[0]) - 1

    for g in range(already_played, games_to_play):
        p1, p2 = random.sample(players, 2)
        print(f'playing game {g+1} of {games_to_play}: {p1} vs {p2}')
        elo_p1 = elo_scores[p1]
        elo_p2 = elo_scores[p2]
        elo_scores[p1], elo_scores[p2] = run_game(p1, elo_p1, p2, elo_p2)

        for i, p in enumerate(players):
            elo_histories[i].append(elo_scores[p])

        print(elo_histories)
        with open('elo_scores', 'wb') as f:
            pkl.dump(elo_scores, f)
        with open('elo_histories', 'wb') as f:
            pkl.dump(elo_histories, f)

    return elo_scores, elo_histories

if __name__ == '__main__':
    players = ['heuristicTeam', 'ucbTeam', 'ucbheuristicTeam', 'raveTeam', 'raveHeuristicTeam']
    games_to_play = 20 * len(players)

    # Load previous data if it exists
    if os.path.exists('elo_scores') and os.path.exists('elo_histories'):
        print("Resuming previous tournament...")
        with open('elo_scores', 'rb') as f:
            elo_scores = pkl.load(f)
        with open('elo_histories', 'rb') as f:
            elo_histories = pkl.load(f)
    else:
        print("Starting fresh tournament...")
        elo_scores = None
        elo_histories = None

    scores, elo_histories = elo_tournament(players, games_to_play, elo_scores, elo_histories)

    # Save final state
    with open('elo_scores', 'wb') as f:
        pkl.dump(scores, f)
    with open('elo_histories', 'wb') as f:
        pkl.dump(elo_histories, f)

    # Plotting
    fig, ax = plt.subplots()
    ax.set_ylabel("Elo Score")
    ax.set_xlabel("Games")
    ax.set_title("Elo Score Evolutions")
    for i, p in enumerate(players):
        ax.plot(range(len(elo_histories[i])), elo_histories[i], label=p)

    ax.legend()
    fig.savefig("elo_scores_fig.png", dpi=300)
    fig.show()