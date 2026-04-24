"""
extract_pdfs.py
---------------
Extracts player statistics from PDF analysis files and converts them to the
standard CSV format used by the rugby ML project.

Output CSV format (same as existing game CSVs):
  Row 1:   player names (comma-separated)
  Rows 2-31: 30 stats in ROW_LABEL order (values only, comma-separated)

Handles:
  - Backs PDFs:    8 player columns
  - Forwards PDFs: 12 player columns + optional Scrum/LO/Maul aggregate columns (skipped)

Usage:
  python src/extract_pdfs.py
"""

import os
import re
import csv
import shutil

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber required. Run: pip install pdfplumber --break-system-packages")

# ── Row label mapping (matches data_loader.py ROW_LABELS) ──────────────────
# Maps each keyword pattern to its output row index (1-30)
# Order matters for Try vs Penalty Try disambiguation.

ROW_PATTERNS = [
    # (output_row_index, pattern_to_match_in_label_column)
    (1,  "Gainline +"),            # gainline_plus
    (1,  "Gainline+"),
    (2,  "Gainline 0"),            # gainline_zero
    (3,  "Unsuccessful Carry"),    # unsuccessful_carry
    (4,  "TOTAL METRES MADE"),     # total_metres_made
    (4,  "Total Metres Made"),
    (5,  "Defender Beaten"),       # defender_beaten
    (6,  "Linebreak made"),        # linebreak_made
    (6,  "Linebreak Made"),
    (7,  "Linebreak conceded"),    # linebreak_conceded
    (7,  "Linebreak Conceded"),
    (9,  "Penalty Try"),           # penalty_try — BEFORE "Try" to avoid substring match
    (8,  "Try"),                   # try (scored)
    (10, "Successful Pass"),       # successful_pass
    (11, "Unsuccessful Pass"),     # unsuccessful_pass
    (12, "Successful Offload"),    # successful_offload
    (13, "Unsuccessful Offload"),  # unsuccessful_offload
    (14, "Pos Attack Ruck"),       # support_pos_attack_ruck
    (15, "Neg Attack Ruck"),       # support_neg_attack_ruck
    (16, "Neutral Attack Ruck"),   # support_neutral_attack_ruck
    (17, "Positive Support"),      # in_possession_positive_support
    (18, "Ineffective Support"),   # in_possession_ineffective_support
    (19, "Dominant Tackle"),       # dominant_tackle
    (20, "Effective Tackle"),      # effective_tackle (NOT Dominant)
    (21, "Tackle assist"),         # tackle_assist
    (21, "Tackle Assist"),
    (22, "Missed Tackle"),         # missed_tackle
    (23, "Unsuccessful Tackle"),   # unsuccessful_tackle
    (24, "Positive barge"),        # positive_barge
    (24, "Positive Barge"),
    (25, "Ineffective barge"),     # ineffective_barge
    (25, "Ineffective Barge"),
    (26, "Turnover Won"),          # turnover_won
    (27, "Turnover lost"),         # turnover_lost
    (27, "Turnover Lost"),
    (28, "Pen For"),               # pen_for
    (29, "Pen Against"),           # pen_against
    (30, "Yellow Card"),           # yellow_card
]

# Rows to SKIP (section headers and derived totals)
SKIP_PATTERNS = [
    "ATTACK: BALL CARRIER", "Attack: Ball Carrier",
    "ATTACK: PASSING", "Attack: Passing",
    "ATTACK: SUPPORT", "Attack: Support",
    "DEFENCE", "Defence",
    "TURNOVERS", "Turnovers",
    "DISCIPLINE", "Discipline",
    "TOTAL POSITIVE CARRIES", "Total Positive Carries",
    "TOTAL POSITIVE TACKLE COUNT", "Total Positive Tackle Count",
    "TOTAL INEFFECTIVE TACKLE COUNT", "Total Ineffective Tackle Count",
]

# Known non-player columns in Forwards PDFs (set-piece aggregates)
NON_PLAYER_COLS = {"Scrum", "Tarf LO", "Maul", "Trin LO", "Lineout",
                   "Line Out", "LO", "SCRUM", "MAUL"}


def match_row_pattern(label: str) -> int | None:
    """Return the ROW_LABEL index (1-30) for a given label string, or None to skip."""
    label = label.strip()

    # Check skip patterns first
    for skip in SKIP_PATTERNS:
        if skip.lower() in label.lower():
            return None

    # Dominant Tackle must be matched before Effective Tackle
    if "Dominant Tackle" in label:
        return 19
    if "Effective Tackle" in label and "Dominant" not in label:
        return 20
    # Penalty Try before Try
    if "Penalty Try" in label:
        return 9
    if label.strip() == "Try" or "▶Try" in label or ">Try" in label:
        return 8
    # Positive barge before Ineffective barge
    if "Positive barge" in label or "Positive Barge" in label:
        return 24
    if "Ineffective barge" in label or "Ineffective Barge" in label:
        return 25
    # Positive Support before Ineffective Support
    if "Positive Support" in label:
        return 17
    if "Ineffective Support" in label:
        return 18

    for row_idx, pattern in ROW_PATTERNS:
        if pattern.lower() in label.lower():
            return row_idx

    return None  # unrecognised row — skip


def extract_game_table(pdf_path: str, position: str) -> tuple[list[str], dict[int, list[float]]]:
    """
    Extract player names and 30-row stats from a PDF.

    Returns
    -------
    players : list of player name strings
    stats   : dict mapping row_index (1-30) → list of values
    """
    players = []
    stats = {i: [] for i in range(1, 31)}
    found_header = False

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Try table extraction first
            tables = page.extract_tables()
            if not tables:
                continue

            # Use the largest table on the page
            table = max(tables, key=lambda t: len(t))

            for row_num, row in enumerate(table):
                if row is None or all(c is None or str(c).strip() == "" for c in row):
                    continue

                # Flatten cells
                cells = [str(c).strip() if c is not None else "" for c in row]

                # Find the player header row (first row with mostly non-numeric values
                # and no label in column 0, or where col 0 contains a game title)
                if not found_header:
                    # Look for the player name row: col[0] might be a game title or empty,
                    # and col[1:] contains player names
                    non_empty = [c for c in cells[1:] if c and not c.replace('.', '').isdigit()]
                    if len(non_empty) >= 3 and not any(
                        skip.lower() in cells[0].lower() for skip in SKIP_PATTERNS
                    ):
                        # Filter out non-player columns (set-piece aggregates)
                        raw_players = [c for c in cells[1:] if c and c not in NON_PLAYER_COLS
                                       and not c.replace('.', '').isdigit()]
                        if len(raw_players) >= 3:
                            players = raw_players
                            found_header = True
                            continue

                if not found_header:
                    continue

                # Data row: col[0] is the label, col[1:] are values
                label = cells[0]
                if not label:
                    continue

                row_idx = match_row_pattern(label)
                if row_idx is None:
                    continue

                # Extract numeric values for each player (skip non-player columns)
                values = []
                player_col = 0
                for c in cells[1:]:
                    if c in NON_PLAYER_COLS:
                        continue  # skip set-piece aggregate column header
                    # Try to parse numeric
                    try:
                        values.append(float(c) if c != "" else 0.0)
                    except ValueError:
                        values.append(0.0)
                    player_col += 1
                    if player_col >= len(players):
                        break

                # Pad/trim to exactly len(players) values
                while len(values) < len(players):
                    values.append(0.0)
                values = values[:len(players)]

                stats[row_idx] = values

    return players, stats


def write_csv(players: list[str], stats: dict[int, list[float]], out_path: str):
    """Write the extracted data in the standard CSV format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(players)
        for row_idx in range(1, 31):
            values = stats.get(row_idx, [0.0] * len(players))
            if not values:
                values = [0.0] * len(players)
            writer.writerow([int(v) if v == int(v) else v for v in values])
    print(f"  ✓ Wrote {out_path}")


def get_game_number(filename: str) -> int | None:
    """Extract AIL game number from PDF filename."""
    # Patterns: AIL1_..., AIL_1_..., AIL19..., AIL2_...
    m = re.search(r'AIL[_\s]*(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def process_season(analysis_dir: str, season_label: str, output_base: str):
    """
    Process all Backs and Forwards PDFs for a given season.

    Parameters
    ----------
    analysis_dir  : folder like .../Analysis/24-25
    season_label  : output folder name like '24_25'
    output_base   : root of data/raw/
    """
    for position, subdir in [("Back", "Backs"), ("Forward", "Forwards")]:
        pdf_folder = os.path.join(analysis_dir, subdir)
        if not os.path.isdir(pdf_folder):
            print(f"[WARNING] Not found: {pdf_folder}")
            continue

        out_folder = os.path.join(output_base, season_label, position)
        os.makedirs(out_folder, exist_ok=True)

        print(f"\n── {season_label} / {position} ──")
        for filename in sorted(os.listdir(pdf_folder)):
            if not filename.lower().endswith('.pdf'):
                continue
            game_num = get_game_number(filename)
            if game_num is None:
                print(f"  [SKIP] Can't parse game number: {filename}")
                continue

            out_path = os.path.join(out_folder, f"game{game_num}.csv")
            if os.path.exists(out_path):
                print(f"  [EXISTS] game{game_num}.csv — skipping")
                continue

            pdf_path = os.path.join(pdf_folder, filename)
            print(f"  Extracting: {filename} → game{game_num}.csv")
            try:
                players, stats = extract_game_table(pdf_path, position)
                if not players:
                    print(f"  [ERROR] No players found in {filename}")
                    continue
                missing = [i for i in range(1, 31) if not stats.get(i)]
                if missing:
                    print(f"  [WARN] Missing rows {missing} in {filename}")
                write_csv(players, stats, out_path)
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")


def delete_analysis_folder(analysis_root: str):
    """Delete the Analysis folder after successful extraction."""
    if os.path.isdir(analysis_root):
        shutil.rmtree(analysis_root)
        print(f"\nDeleted: {analysis_root}")
    else:
        print(f"[INFO] Already removed: {analysis_root}")


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.abspath(os.path.join(THIS_DIR, '..'))
    RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
    ANALYSIS_ROOT = os.path.join(RAW_DIR, 'Analysis')

    print("=" * 60)
    print("PDF → CSV Extraction for Rugby ML Dissertation")
    print("=" * 60)

    # Check what analysis folders exist
    if not os.path.isdir(ANALYSIS_ROOT):
        print(f"[ERROR] Analysis folder not found: {ANALYSIS_ROOT}")
        exit(1)

    seasons_found = []
    for season_folder in sorted(os.listdir(ANALYSIS_ROOT)):
        full_path = os.path.join(ANALYSIS_ROOT, season_folder)
        if os.path.isdir(full_path) and season_folder.startswith(('22', '23', '24')):
            seasons_found.append((full_path, season_folder))

    print(f"\nFound analysis seasons: {[s for _, s in seasons_found]}")

    # Process each season
    season_label_map = {
        "22-23": "22_23",
        "23-24": "23_24",
        "24-25": "24_25",
    }

    skipped_seasons = []
    for analysis_dir, season_folder in seasons_found:
        season_label = season_label_map.get(season_folder, season_folder.replace('-', '_'))

        # Skip 23-24 if CSVs already exist (check for game1.csv in Back and Forward)
        back_game1 = os.path.join(RAW_DIR, season_label, "Back", "game1.csv")
        fwd_game1  = os.path.join(RAW_DIR, season_label, "Forward", "game1.csv")
        if os.path.exists(back_game1) and os.path.exists(fwd_game1):
            print(f"\n[SKIP] {season_label} — CSVs already exist")
            skipped_seasons.append(season_label)
            continue

        process_season(analysis_dir, season_label, RAW_DIR)

    print("\n" + "=" * 60)
    print("Extraction complete.")

    # Verify outputs
    print("\nVerifying output files:")
    for season_label in season_label_map.values():
        for position in ["Back", "Forward"]:
            folder = os.path.join(RAW_DIR, season_label, position)
            if os.path.isdir(folder):
                csvs = [f for f in os.listdir(folder) if f.endswith('.csv')]
                print(f"  {season_label}/{position}: {len(csvs)} CSV files")

    # Delete Analysis folder
    print("\nRemoving Analysis folder...")
    delete_analysis_folder(ANALYSIS_ROOT)
    print("Done.")
