import capture
import random

def update_elo(winner_elo, loser_elo, point_diff=100, odds_diff=2, K=1):
    #the update rule used here attributes a 2:1 odds of winning to each 100 points of difference by default
    #K can be used to make scores converge faster but they become more unstable
    winner_elo += K * (1 - (( odds_diff ** (winner_elo / point_diff) ) / ((odds_diff ** (winner_elo / point_diff)) + (odds_diff ** (loser_elo / point_diff)))))
    loser_elo += K * (-(( odds_diff ** (loser_elo / point_diff) ) / ((odds_diff ** (winner_elo / point_diff)) + (odds_diff ** (loser_elo / point_diff)))))
    return winner_elo, loser_elo

def run_game(p1, p1_elo, p2, p2_elo):
    #TODO add player options
    options = capture.readCommand(['-q', "-r", p1, "-b", p2])
    games = capture.runGames(**options)
    score = games[0].state.data.score
    if score > 0:
        p1_elo, p2_elo = update_elo(p1_elo, p2_elo)
    elif score < 0:
        p2_elo, p1_elo = update_elo(p2_elo, p1_elo)
    return p1_elo, p2_elo

def calculate_odds(p1_elo, p2_elo, point_diff=100, odds_diff=2):
    return ( odds_diff ** (p1_elo / point_diff) ) / ((odds_diff ** (p1_elo / point_diff)) + (odds_diff ** (p2_elo / point_diff)))


def elo_tournament(players, games_to_play=-1):
    if games_to_play == -1:
        games_to_play = 50*len(players)

    elo_scores = {p:500 for p in players}
    for g in range(games_to_play):
        p1, p2 = random.sample(players, 2)
        elo_p1 = elo_scores[p1]
        elo_p2 = elo_scores[p2]
        elo_p1, elo_p2 = run_game(p1, elo_p1, p2, elo_p2)

    return elo_scores

