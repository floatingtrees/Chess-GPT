import chess.pgn
import chess
import csv

def pgn_to_csv(pgn_file:str, csv_file:str = "games.csv") -> None:
    """
    Takes in a pgn file and outputs a csv file with same information, each row is 1 move

    Args:
        pgn_file: file of format .pgn
        csv_file: output csv file name

    Returns:
        None
    """
    with open(pgn_file, "r") as in_file:
        games_data = []

        while True:
            game = chess.pgn.read_game(in_file)
            if game is None:
                break

            games_data.append((game, dict(game.headers)))
    #headers for csv
    keys = list(games_data[0][0].headers.keys()) + ["fen_before", "fen_after", "next_san_move", "next_uci_move","move_number"]

    with open(csv_file, "w", newline="", encoding="utf-8") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=keys)
        writer.writeheader()
        for game, game_info in games_data:
            moveset_san, moveset_uci, fen_before_moveset, fen_after_moveset = get_moves_fen(game)

            for i in range(len(fen_before_moveset)):
                writer.writerow({**game_info, "fen_before": fen_before_moveset[i], "fen_after": fen_after_moveset[i],
                                 "next_san_move": moveset_san[i], "next_uci_move": moveset_uci[i], "move_number": -((i + 1)// -2)})
                # -(i+1//-2) is ceiling int division to find move number
    print(f"CSV file Created: {csv_file}")


def get_moves_fen(game: chess.pgn.Game) -> tuple[list[str], list[str], list[str],list[str]]:
    """
    Gets the moves in SAN and UCI format and FENs from board BEFORE the move

    Args:
        game: chess.pgn.Game object

    Returns:
        tuple[list[str], list[str], list[str], list[str]]
        san move array, uci move array, fen state before array, fen state after array
    """
    curr_board = chess.Board()
    san_moves = []
    uci_moves = []
    fen_before_states = []
    fen_after_states = []
    for move in game.mainline_moves():
        san_moves.append(curr_board.san(move))
        uci_moves.append(curr_board.uci(move))
        fen_before_states.append(curr_board.fen())
        curr_board.push(move)
        fen_after_states.append(curr_board.fen())
    return san_moves, uci_moves, fen_before_states, fen_after_states


if __name__ == "__main__":
    pgn_file_name = "Kasparov.pgn"
    with open(pgn_file_name, encoding = "utf-8") as test_pgn:
        test_game = chess.pgn.read_game(test_pgn)
    print(test_game)

    pgn_to_csv(pgn_file= "Kasparov.pgn", csv_file = "Kasparov.csv")
    game_moves_san, game_moves_uci, fen_before,fen_after = get_moves_fen(test_game)
    print(game_moves_san)
    print(game_moves_uci)

    print(fen_before)
    print(fen_after)
