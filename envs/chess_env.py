import chess
from typing import Optional, Union, Tuple
from enum import Enum


class BoardEnv:
    """
    A chess board env class to:
    - Load and manage board states
    - Generate prompts for LLM chess play
    - Parse LLM responses into chess moves
    """
    
    PIECE_LABELS = {
        chess.PAWN:   ("Pawn",   "Pawns"),
        chess.KNIGHT: ("Knight", "Knights"),
        chess.BISHOP: ("Bishop", "Bishops"),
        chess.ROOK:   ("Rook",   "Rooks"),
        chess.QUEEN:  ("Queen",  "Queens"),
        chess.KING:   ("King",   "Kings"),
    }
    
    def __init__(self, fen: Optional[str] = None):
        """
        Initialize the BoardEnv with a chess board.
        
        Args:
            fen: FEN string to initialize the board. If None, uses standard starting position.
        """
        self.board = chess.Board(fen) if fen else chess.Board()
    
    def load_board(self, fen: str) -> None:
        """
        Load a new board state from a FEN string.
        
        Args:
            fen: FEN string representing the board state
        """
        self.board = chess.Board(fen)
    
    def reset_to_starting_position(self) -> None:
        """Reset the board to the standard starting position."""
        self.board = chess.Board()
    
    def _format_side(self, color: chess.Color) -> str:
        """Format piece positions for one side (white or black)."""
        lines = []
        header = "White:" if color == chess.WHITE else "Black:"
        lines.append(header)
        for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
            squares = sorted(self.board.pieces(pt, color))  # SquareSet -> sorted list of ints
            if not squares:
                continue
            names = [chess.square_name(sq) for sq in squares]
            singular, plural = self.PIECE_LABELS[pt]
            label = singular if len(names) == 1 else plural
            lines.append(f"\t{label}: " + " ".join(names))
        return "\n".join(lines)
    
    def get_piece_lists(self) -> str:
        """Get formatted piece lists for both sides."""
        return self._format_side(chess.WHITE) + "\n\n" + self._format_side(chess.BLACK)
    
    def get_turn_to_move(self) -> str:
        """Get formatted string indicating whose turn it is."""
        return "White to move" if self.board.turn == chess.WHITE else "Black to move"
    
    def get_en_passant_info(self) -> str:
        """Get formatted en passant information."""
        has_ep = self.board.has_legal_en_passant()  # boolean
        target = chess.square_name(self.board.ep_square) if self.board.ep_square is not None else "-"
        # enumerate the actual legal EP moves (could be 0, 1, or 2)
        ep_moves = [m for m in self.board.legal_moves if self.board.is_en_passant(m)]
        movers = " ".join(sorted(chess.square_name(m.from_square) for m in ep_moves)) or "None"
        sans = ", ".join(self.board.san(m) for m in ep_moves) or "None"
        return (
            f"""<EN_PASSANT>
        Has legal EP now: {has_ep}
        Target square: {target}
        Capturing pawn(s): {movers}
        Moves (SAN notation): {sans}
        </EN_PASSANT>"""
        )
    
    def get_castling_info(self) -> str:
        """Get formatted castling rights and legal castle moves."""
        rights = {
            "white": {
                "kingside": self.board.has_kingside_castling_rights(chess.WHITE),
                "queenside": self.board.has_queenside_castling_rights(chess.WHITE),
            },
            "black": {
                "kingside": self.board.has_kingside_castling_rights(chess.BLACK),
                "queenside": self.board.has_queenside_castling_rights(chess.BLACK),
            }
        }
        # Castle moves that are legal *in the current position/turn*
        legal_castles = [m for m in self.board.legal_moves if self.board.is_castling(m)]
        castles_sans = [self.board.san(m) for m in legal_castles]
        return (
            f"""\t<CASTLING>
        Rights: {rights}
        Legal castle moves now (SAN notation): {', '.join(castles_sans) if castles_sans else 'None'}
    </CASTLING>"""
        )
    
    def get_misc_info(self) -> str:
        """Get formatted miscellaneous information (en passant and castling)."""
        return f"""<EN_PASSANT>\n{self.get_en_passant_info()}\n</EN_PASSANT>\n<CAN_CASTLE>\n{self.get_castling_info()}\n</CAN_CASTLE>"""
    
    def get_meta_guidance(self) -> str:
        """Get meta guidance instructions for the LLM."""
        return f"""<META_GUIDANCE>
    You are an expert chess player.
    You will be given board state within the <BOARD> tag.
    You will be given misc info (en passant, castling, etc.) within the <MISC_INFO> tag.
    You will be given the turn (white to move or black to move) within the <TURN> tag which indicates which player you are.
    </META_GUIDANCE>
    <INSTRUCTIONS>
    You will be RESPONDING with a final move in the format of <MOVE>[your move here as SAN notation string]</MOVE> where the contents of MOVE are SAN notation strings. E.g. to move the knight to c3 you will return <MOVE>Nc3</MOVE>.
    </INSTRUCTIONS>
    """
    
    def generate_prompt(self) -> str:
        """
        Generate a complete prompt for the LLM to play chess.
        
        Returns:
            Formatted string containing board state, misc info, and turn information
        """
        return f"""{self.get_meta_guidance()}\n<BOARD>\n{self.get_piece_lists()}\n</BOARD>\n<MISC_INFO>\n{self.get_misc_info()}\n</MISC_INFO>\n<TURN>\n{self.get_turn_to_move()}\n</TURN>"""
    
    def parse_llm_move(self, output: str) -> chess.Move:
        """
        Parse LLM output to extract a chess move from SAN format

        Pulls the last instance of the MOVE tags
        
        Args:
            output: LLM response string containing move in <MOVE>...</MOVE> tags
            
        Returns:
            chess.Move object, or chess.Move.null() if parsing fails
        """

        move_start_index = output.rfind("<MOVE>")
        move_end_index = output.rfind("</MOVE>")
        if move_start_index == -1 or move_end_index == -1:
            return chess.Move.null()
        move_str = output[move_start_index + 6:move_end_index]

        try:
            return self.board.parse_san(move_str)
        except (IndexError, chess.InvalidMoveError):
            return chess.Move.null()
    
    def make_move(self, move: chess.Move) -> bool:
        """
        Make a move on the board.
        
        Args:
            move: chess.Move object
            
        Returns:
            True if move was legal and made, False otherwise
        """        
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    class SimMoveStatus(Enum):
        ILLEGAL_MOVE = "illegal_move"
        UNPARSEABLE_MOVE = "unparseable_move"
        LEGAL_MOVE = "legal_move"

    def sim_move(self, move: chess.Move) -> Tuple[str, str, SimMoveStatus]:
        """
        Get the before and after board states of a move. Returns error object if move is illegal or unparseable.
        
        Will not push the move to the current board.

        Args:
            move: chess.Move object
            illegal_move_penalty: penalty for making an illegal move
            unparseable_move_penalty: penalty for unparseable move
            
        Returns:
            Tuple[str, str, SimMoveError]: before and after board states in FEN strings and error object
        """

        temp_board = self.board.copy()
        before_fen = temp_board.fen()
        after_fen = before_fen
        if move in temp_board.legal_moves:
            temp_board.push(move)
            after_fen = temp_board.fen()
            return before_fen, after_fen, self.SimMoveStatus.LEGAL_MOVE
        elif move == chess.Move.null(): 
            return before_fen, after_fen, self.SimMoveStatus.UNPARSEABLE_MOVE
        else:
            return before_fen, after_fen, self.SimMoveStatus.ILLEGAL_MOVE
    
    def get_legal_moves(self) -> list[chess.Move]:
        """Get list of all legal moves in the current position."""
        return list(self.board.legal_moves)
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.board.is_game_over()
    
    def get_result(self) -> str:
        """Get the game result."""
        return self.board.result()
    
    def get_fen(self) -> str:
        """Get the current board state as FEN string."""
        return self.board.fen()
    
    def __str__(self) -> str:
        """String representation of the board."""
        return str(self.board)
    
    def __repr__(self) -> str:
        """Detailed representation of the BoardEnv."""
        return f"BoardEnv(fen='{self.board.fen()}')"
    

# Example usage:
if __name__ == "__main__":
    # Example FEN from the original code
    FEN = "q3k2r/2pp1pp1/bbn2n1p/1p2p3/4P3/NBPP1N2/1P3PPP/2BQ1RK1 w k - 2 13"
    
    # Create BoardEnv instance
    board_env = BoardEnv(FEN)
    
    # Generate prompt for LLM
    prompt = board_env.generate_prompt()
    print("Generated prompt:")
    print(prompt)
    print("\n" + "="*50 + "\n")
    
    # Example of parsing LLM output
    llm_output = "yap yap yap <MOVE>Ne5</MOVE>"
    parsed_move = board_env.parse_llm_move(llm_output)
    print(f"Parsed move from LLM output '{llm_output}': {parsed_move}")
    
    # Example of simulating a move
    before_fen, after_fen, status = board_env.sim_move(parsed_move)
    print(f"Before move FEN: {before_fen}")
    print(f"After move FEN: {after_fen}")
    print(f"Status: {status}")

    # Example of making a move
    if parsed_move != chess.Move.null():
        print(f"Board before move:\n{board_env}")
        success = board_env.make_move(parsed_move)
        print(f"Move made successfully: {success}")
        print(f"Board after move:\n{board_env}")
