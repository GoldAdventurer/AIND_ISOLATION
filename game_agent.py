"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import isolation


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # HEURISTIC improved_score.
    def difference_moves(myMoves, opponentMoves):
        # All legal moves are equal. We calculate the difference in the number
        # of legal moves between the player and the opponent
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        return float(len(myMoves) - len(opponentMoves))

    # HEURISTIC 1.
    # All legal moves are not equal. I assign a weight of 1/2 to moves around
    # a corridor which is one square thick around the board, i.e.
    # for 1<row<n-2 and 1<col<n-2, the move is assigned a weight of
    # (1- discount). For the rest of the board, the move is assigned
    # a weight of 1.0
    def weighted_nb_legal_moves(moves, discount):
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        weight = []
        for move in moves:
            if (move[0] == 0) or (move[0] == game.width - 1) or \
               (move[0] in(0, 1) and move[1] > 1 and
                    move[1] < game.height - 2) or\
               (move[0] == game.width - 2 and move[1] > 2 and
                    move[1] < game.height - 2):
                    weight.append(1 - discount)
            else:
                weight.append(1.0)
        res = sum(weight)
        return res

    def weighted_difference(discount, myMoves, opponentMoves):
        return weighted_nb_legal_moves(myMoves, discount) - \
                weighted_nb_legal_moves(opponentMoves, discount)

    # HEURISTIC 2
    # In this heuristic, we calculate the squares of the number of legal moves
    # for the player and the opponent.
    def quadratic_difference(myMoves, opponentMoves):
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        return float(len(myMoves) * len(myMoves) -
                     len(opponentMoves) * len(opponentMoves))

    # heuristic open_moves
    def open_moves(myMoves):
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        return float(len(myMoves))

    # HEURISTIC 3
    # In this heuristic, we asign weights to the previous heuristics values
    # and calculate the weighted average.
    def final_custom_score(diff_move_weight, weighted_difference_weight,
                           quadratic_difference_weight, open_moves_weight,
                           discount):
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        if diff_move_weight + weighted_difference_weight + \
           quadratic_difference_weight + open_moves_weight != 1.0:
                print("the weights in the custom_score function do not " +
                      "add up to 1.")
        if discount > 1 or discount < 0:
            print("The discount in the custom_score function must be " +
                  "between 0 and 1, included.")
        weights = [diff_move_weight, weighted_difference_weight,
                   quadratic_difference_weight, open_moves_weight]
        functions = [difference_moves(myMoves, opponentMoves),
                     weighted_difference(discount, myMoves, opponentMoves),
                     quadratic_difference(myMoves, opponentMoves),
                     open_moves(myMoves)]
        return (sum([i*j for i, j in zip(weights, functions)]))

    # HEURISTIC 4
    # In this heuristic we take the minimum of heuristic 1, 2, open_moves and
    # improved_score.
    def final_custom_score_min(discount):
        if game.is_loser(player):
            return float("-inf")

        if game.is_winner(player):
            return float("+inf")

        if discount > 1 or discount < 0:
            print("The discount in the custom_score function must be " +
                  "between 0 and 1, included.")
        functions = [difference_moves(myMoves, opponentMoves),
                     weighted_difference(discount, myMoves, opponentMoves),
                     quadratic_difference(myMoves, opponentMoves),
                     open_moves(myMoves)]
        return min(functions)

    # find opponent
    opponent = game.get_opponent(player)
    # determine the opponent's legal moves and the player's legal moves
    opponentMoves = game.get_legal_moves(opponent)
    myMoves = game.get_legal_moves(player)
    # return score (the first three numbers must be positive and add up to 1.0)
    # return final_custom_score_min(0.2)
    return final_custom_score(0, 1, 0, 0, 0.2)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left
        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        potentialMoves = list()
        if len(legal_moves) == 0:
            return (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.search_depth < 0:
                self.search_depth = int(1e10)

            if self.iterative:
                if self.method == 'minimax':
                    for d in range(1, self.search_depth + 1):
                        potentialMoves.append(self.minimax(game, d))
                elif self.method == 'alphabeta':
                    for d in range(1, self.search_depth + 1):
                        potentialMoves.append(self.alphabeta(game, d))
            else:
                if self.method == 'minimax':
                    potentialMoves.append(self.minimax(game,
                                                       self.search_depth))
                elif self.method == 'alphabeta':
                    potentialMoves.append(self.alphabeta(game,
                                                         self.search_depth))

        except Timeout:
            pass

        # Return the best move from the last completed search iteration
        if potentialMoves != []:
            bestMove = max(potentialMoves)
            return bestMove[1]
        else:
            return (-1, -1)

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        # initialize
        scores = []
        selectedMove = (-1, -1)
        # determine the player
        if maximizing_player:
            player = game.active_player
            scoreFinal = float("-inf")
        else:
            player = game.inactive_player
            scoreFinal = float("+inf")

        legal_moves = game.get_legal_moves()

        if len(legal_moves) == 0:
            score, selectedMove = self.score(game, player), (-1, -1)

        if depth == 1:
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                score = self.score(nextGame, player)
                scores.append(score)
                if (maximizing_player and score > scoreFinal) or \
                   (not maximizing_player and score < scoreFinal):
                    scoreFinal, selectedMove = score, move
            return scoreFinal, selectedMove

        elif depth > 1:
            for move in legal_moves:
                nextGame = game.forecast_move(move)
                score = self.minimax(nextGame, depth - 1,
                                     not maximizing_player)[0]
                scores.append(score)
                if (maximizing_player and score > scoreFinal) or \
                   (not maximizing_player and score < scoreFinal):
                    scoreFinal, selectedMove = score, move
        return scoreFinal, selectedMove

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"),
                  maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        # initialize
        scores = []
        explore = True
        selectedMove = (-1, -1)
        # select player and initialize the resulting score
        if maximizing_player:
            player = game.active_player
            scoreFinal = float("-inf")
        else:
            player = game.inactive_player
            scoreFinal = float("+inf")
        # return the score and (-1, -1) if there are no legal moves
        legal_moves = game.get_legal_moves()
        if len(legal_moves) == 0:
            return (self.score(game, player), (-1, -1))
        # calculate the score for depth = 1
        if depth == 1:
            for move in legal_moves:
                if explore:
                    nextGame = game.forecast_move(move)
                    score = self.score(nextGame, player)
                    scores.append(score)
                    # update beta, the final score and the selected move
                    if not maximizing_player:
                        beta = min(beta, score)
                        if score < scoreFinal:
                            selectedMove = move
                            scoreFinal = score
                    # update alpha, the final score and the selected move
                    else:
                        alpha = max(alpha, score)
                        if score > scoreFinal:
                            selectedMove = move
                            scoreFinal = score
                    # interrupt the search if alpha >= beta
                    if alpha >= beta:
                        explore = False
            return scoreFinal, selectedMove
        # proceed with recursion with depth > 1
        elif depth > 1:
            for move in legal_moves:
                if explore:
                    nextGame = game.forecast_move(move)
                    score = self.alphabeta(nextGame, depth - 1,
                                           alpha,
                                           beta, not maximizing_player)[0]
                    scores.append(score)
                    # update beta, the final score and the selected move
                    if not maximizing_player:
                        beta = min(beta, score)
                        if score < scoreFinal:
                            selectedMove = move
                            scoreFinal = score
                    # update alpha, the final score and the selected move
                    else:
                        alpha = max(alpha, score)
                        if score > scoreFinal:
                            selectedMove = move
                            scoreFinal = score
                    # interrupt the search if alpha >= beta
                    if alpha >= beta:
                        explore = False
            # return result
            return scoreFinal, selectedMove
