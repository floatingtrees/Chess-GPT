import chess
import chess.pgn
import random
import asyncio
import time

# This class can be used to sample positions from a file containing move sequences for multiple games.
class DataSampler:
    """A class to sample positions from a file containing sequences of moves for multiple games."""
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.move_sequences = [line.strip().split() for line in f if line.strip()]

    def count_games(self):
        """Return the number of games in the file."""
        return len(self.move_sequences)

    def count_moves(self, game_id):
        """Count the total number of moves in the specified game."""
        return len(self.move_sequences[game_id])

    def get_position(self, move_number: int, game_id):
        """Get the board position at a specific move number in a specific game. Returns an FEN string."""
        board = chess.Board()
        move_list = self.move_sequences[game_id]
        for i in range(min(move_number, len(move_list))):
            board.push_uci(move_list[i])
        return board.fen()

    def get_random_position(self):
        """Get a random position from a random game. Returns the FEN string of that position."""
        game_id = random.randint(0, self.count_games() - 1)
        total_moves = self.count_moves(game_id)
        random_move = random.randint(1, total_moves)
        return self.get_position(random_move, game_id)

    def get_random_game_moves(self):
        """Return the move list of a random game."""
        game_idx = random.randint(0, self.count_games() - 1)
        return self.move_sequences[game_idx]
    def get_random_positions(self, n):
        fens = []
        for i in range(n):
            fens.append(self.get_random_position())
        return fens
    async def get_random_positions_async(self, n):
        """Get n random positions asynchronously. Returns a list of FEN strings."""
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, self.get_random_position) for _ in range(n)]
        return await asyncio.gather(*tasks)

