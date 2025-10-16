import chess
import math
from stockfish import Stockfish

# --- Constants ---
FIXED_DEPTH = 15 #this can change if you want a stronger stockfish
WIN_PERCENT_NOISE_THRESHOLD = 3.0 #Sometimes playing a good move like in a 0.00 positiom changes the winpercentage by like 1% giving negative rewads
BEST_MOVE_BONUS = 0.05  # A small reward for playing the best move

# --- Helper Functions ---
def get_win_percentage(centipawns: int) -> float:
    """Converts centipawn evaluation to a win percentage (from 0 to 100)."""
    clipped_cp = max(-1500, min(1500, centipawns))
    return 50 + 50 * (2 / (1 + math.exp(-0.00368208 * clipped_cp)) - 1)

def _get_cp_from_eval(evaluation: dict) -> int:
    """Helper function to handle both 'cp' and 'mate' evaluations."""
    if evaluation['type'] == 'cp':
        return evaluation['value']
    elif evaluation['type'] == 'mate':
        return 30000 if evaluation['value'] > 0 else -30000
    return 0

# --- Main Reward Calculation Logic ---
def calculate_move_reward(stockfish: Stockfish , fen: str, move: str) -> float:
    """
    Calculates a scaled reward (-1.0 to 1.0) based on the change in win percentage for a move.
    """
    try:
        board = chess.Board(fen)

        if board.is_game_over():
            return -1.5  #Adds an extra -1.5 for moving when the game is over

        # The parse_san function will raise a ValueError for illegal or invalid moves
        parsed_move = board.parse_san(move)
        
        # This check is technically redundant if parse_san works, but good for safety
        if parsed_move not in board.legal_moves:
            return -3.0  # Large penalty for illegal moves

        #Centipawn of the position before the move
        stockfish.set_fen_position(fen)
        eval_before = stockfish.get_evaluation()
        cp_before = _get_cp_from_eval(eval_before)

        # 4. Make the Move and Check Post-Move Terminal States
        board.push(parsed_move)
        if board.is_checkmate():
            return 1.0  # Max reward for delivering checkmate
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0  # Neutral reward for a draw

        # Centipawns after the move is played
        stockfish.set_fen_position(board.fen())
        eval_after = stockfish.get_evaluation()
        cp_after = _get_cp_from_eval(eval_after)
        
        # checking to see who moved so the bot knows what is positive
        if board.turn == chess.BLACK: 
            win_percent_before = get_win_percentage(cp_before)
            win_percent_after = get_win_percentage(cp_after)
        else: 
            win_percent_before = get_win_percentage(-cp_before)
            win_percent_after = get_win_percentage(-cp_after)

        change = win_percent_after - win_percent_before
        reward = 0.0

        # 7. Noise threshold
        if abs(change) < WIN_PERCENT_NOISE_THRESHOLD:
            reward = BEST_MOVE_BONUS 
        #if needed we can actually check for best move
        # however i find this good enough and the centipawn change the wp by +-3% after getting the move
        else:
            # Scale the reward
            reward = change / 100.0

        return reward
    
    except chess.IllegalMoveError as e:
        # Catches illegal moves 
        print(f"Illegal move or invalid notation for '{move}': {e}")
        return -3.0

    except Exception as e:
        # Catches other unexpected errors and incorrect notations
        print(f"An unexpected error occurred in calculate_move_reward: {e}")
        return -5.0

# path to stockfish
stockfish_path = "C:\\Chess_Engines\\stockfish\\stockfish-windows-x86-64-avx2.exe"
stockfish = Stockfish(path=stockfish_path, depth=FIXED_DEPTH)

def reward(board_state: str, model_response: str) -> float:
    """
    This is the final reward function. It takes the board state (FEN) and
    the model's move (SAN) and returns the calculated reward.
    """
    return calculate_move_reward(stockfish, board_state, model_response)

if __name__ == '__main__':
    
    # Erigaisi vs Vokhidov 45th budapest olympiad
    erigaisi_fen = "4nk2/p1r1qpp1/1p5p/3B4/2P2Q1P/1P3R2/P4PK1/8 w - - 1 38"
    best_move_bf7 = "Bf7"
    second_best_re3 = "Re3" #move he actually played
    terrible_move_a3 = 'Qxh6'
    illegal_move = 'Bf8'
    wrong_notation_move = 'hello'
    
    print("\n--- Erigaisi vs Vokhidov Example ---")
    reward_best = reward(erigaisi_fen, best_move_bf7) #0.05
    reward_second_best = reward(erigaisi_fen, second_best_re3) #-0.2048
    reward_terrible = reward(erigaisi_fen, terrible_move_a3) #-0.5917
    reward_illegal = reward(erigaisi_fen,illegal_move) #-3.0
    reward_wrong_notation = reward(erigaisi_fen,wrong_notation_move) #-5.0
    
    
    print(f"Reward for the best move '{best_move_bf7}': {reward_best:.4f}")
    print(f"Reward for the second best move '{second_best_re3}': {reward_second_best:.4f}")
    print(f"Reward for the worst move '{terrible_move_a3}': {reward_terrible:.4f}")
    print(f"Reward for the illegal move '{illegal_move}': {reward_illegal:.4f}")
    print(f"Reward for the wrong notation move '{wrong_notation_move}': {reward_wrong_notation:.4f}")
    #tested it on many different positions seems ok

