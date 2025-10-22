import chess.pgn

def extract_move_sequences(pgn_path, output_txt_path):
    with open(pgn_path, 'r') as pgn_file, open(output_txt_path, 'w') as out_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            moves = []
            node = game
            while node.variations:
                next_node = node.variation(0)
                moves.append(next_node.move.uci())
                node = next_node
            out_file.write(' '.join(moves) + '\n')

# Example usage:
extract_move_sequences('filtered_games.pgn', 'move_sequences.txt')
