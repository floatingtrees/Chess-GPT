import chess
import chess.pgn
import random
class DataSampler:
    """A class to sample positions from PGN files."""
    def __init__(self, pgn_path):
        pgn = open(pgn_path)
        self.game = chess.pgn.read_game(pgn)
        self.board = self.game.board()
    def count_moves(self):
        """Count the total number of moves in the game."""
        return self.game.end().ply()  
    def get_position(self, move_number: int):
        """Get the board position at a specific move number. Reterns an FEN string."""
        total = 0
        board = self.game.board()
        for move in self.game.mainline_moves():
            board.push(move)
            total += 1
            if total == move_number:
                return board.fen()
        return board.fen()
    def get_random_position(self):
        """Get a random position from the game. Returns the FEN string of that position."""
        total_moves = self.count_moves()
        random_move = random.randint(1, total_moves)
        return self.get_position(random_move)