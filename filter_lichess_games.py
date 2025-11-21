import re
import zstandard as zstd

INPUT_FILE = "lichess_db_standard_rated_2013-01.pgn.zst"
OUTPUT_FILE = "filtered_games.pgn"

def classify_time_control(tc):
    try:
        base, inc = map(int, tc.split('+'))
    except ValueError:
        return "Other"

    if base <= 180:
        return "Bullet"
    elif base <= 480:
        return "Blitz"
    elif base <= 1500:
        return "Rapid"
    else:
        return "Classical"

def process_game_fragment(game_data, out_file, count_total, count_kept):
    """Helper to process a single game and update counts."""
    text = game_data.decode('utf-8', errors='ignore')
    count_total[0] += 1
    
    match = re.search(r'\[TimeControl\s+"([^"]+)"\]', text)
    if match:
        tc = match.group(1)
        category = classify_time_control(tc)
        if category in ("Blitz", "Rapid", "Classical"):
            out_file.write(text.strip() + "\n\n")
            count_kept[0] += 1
            
    return count_total[0], count_kept[0]

def filter_games_from_zst(input_file, output_file):
    dctx = zstd.ZstdDecompressor()
    # Use mutable lists for counters to allow modification within the helper
    count_total = [0]
    count_kept = [0]

    with open(input_file, 'rb') as compressed, open(output_file, 'w', encoding='utf-8') as out:
        with dctx.stream_reader(compressed) as reader:
            buffer = b""
            while True:
                chunk = reader.read(2**20)
                if not chunk:
                    break
                buffer += chunk

                games = re.split(rb'(?=\[Event\s")', buffer)
                buffer = games.pop()  

                for g in games:
                    count_total[0], count_kept[0] = process_game_fragment(g, out, count_total, count_kept)

            # Process the final remaining game in 'buffer' after the loop
            if buffer:
                count_total[0], count_kept[0] = process_game_fragment(buffer, out, count_total, count_kept)

    print(f"✅ Processed {count_total[0]} games, kept {count_kept[0]} ({output_file})")

def main():
    filter_games_from_zst(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()
