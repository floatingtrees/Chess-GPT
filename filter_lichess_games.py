import re
import zstandard as zstd

INPUT_FILE = "lichess_db_standard_rated_2013-01.pgn.zst"
OUTPUT_FILE = "filtered_games.pgn"

def classify_time_control(tc):
    """Return the time control category (Bullet, Blitz, Rapid, Classical)."""
    try:
        base, inc = map(int, tc.split('+'))
    except ValueError:
        return "Other"

    # Base time in seconds
    if base <= 180:
        return "Bullet"
    elif base <= 480:
        return "Blitz"
    elif base <= 1500:
        return "Rapid"
    else:
        return "Classical"

def filter_games_from_zst(input_file, output_file):
    dctx = zstd.ZstdDecompressor()
    count_total = 0
    count_kept = 0

    with open(input_file, 'rb') as compressed, open(output_file, 'w', encoding='utf-8') as out:
        with dctx.stream_reader(compressed) as reader:
            buffer = b""
            while True:
                chunk = reader.read(2**20)  # read 1MB at a time
                if not chunk:
                    break
                buffer += chunk

                games = re.split(rb'(?=\[Event\s")', buffer)
                buffer = games.pop()  

                for g in games:
                    text = g.decode('utf-8', errors='ignore')
                    count_total += 1
                    match = re.search(r'\[TimeControl\s+"([^"]+)"\]', text)
                    if match:
                        tc = match.group(1)
                        category = classify_time_control(tc)
                        if category in ("Blitz", "Rapid", "Classical"):
                            out.write(text.strip() + "\n\n")
                            count_kept += 1

    print(f"✅ Processed {count_total} games, kept {count_kept} ({output_file})")

def main():
    filter_games_from_zst(INPUT_FILE, OUTPUT_FILE)

if __name__ == "__main__":
    main()
