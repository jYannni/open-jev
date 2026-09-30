"""Build a small next-move dataset from Lichess puzzles for LoRA fine-tuning.

Reads one row group of the Lichess/chess-puzzles parquet on the Hugging Face
Hub over HTTP range requests (no full download), samples a few thousand
puzzles, turns each position into a prompt/completion pair, and writes
train/valid/test JSONL files in the format ``mlx_lm.lora`` expects.

Each example:
    prompt     -> ASCII board, FEN, side to move, legal moves (UCI), "Best move:"
    completion -> " e2e4"  (the puzzle's next solution move in UCI)

Lichess puzzle rows give the FEN *before* the opponent's last move; ``Moves``
starts with that opponent move followed by the solution. We replay the first
move, then emit one example per solution move (the side-to-move's moves only).

Usage:
    .venv/bin/python finetune/prepare_chess_data.py --puzzles 3000 --out data/chess
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import chess
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

PARQUET = "datasets/Lichess/chess-puzzles/data/train-00000-of-00003.parquet"
COLUMNS = ["PuzzleId", "FEN", "Moves", "Rating", "Popularity", "GameUrl"]


def fetch_puzzles(row_group: int = 0) -> list[dict]:
    """Read one row group (~490k puzzles, ~60 MB over the wire) from the Hub."""
    with HfFileSystem().open(PARQUET, "rb") as f:
        table = pq.ParquetFile(f).read_row_group(row_group, columns=COLUMNS)
    return table.to_pylist()


def board_ascii(board: chess.Board) -> str:
    """Rank 8 at top, files a-h left to right, '.' for empty squares."""
    lines = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            row.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1} " + " ".join(row))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def make_prompt(board: chess.Board) -> str:
    side = "white" if board.turn == chess.WHITE else "black"
    legal = " ".join(m.uci() for m in board.legal_moves)
    return (
        f"You are playing chess as {side}. Pick the best move.\n\n"
        f"Board (uppercase = white, lowercase = black):\n{board_ascii(board)}\n\n"
        f"FEN: {board.fen()}\n"
        f"Legal moves: {legal}\n"
        f"Best move:"
    )


def puzzle_examples(row: dict, max_per_puzzle: int) -> list[dict]:
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    board.push_uci(moves[0])  # opponent's move that sets up the puzzle
    out = []
    for i in range(1, len(moves), 2):  # solution moves by the side to move
        if len(out) >= max_per_puzzle:
            break
        target = chess.Move.from_uci(moves[i])
        if target not in board.legal_moves:
            break
        out.append(
            {
                "prompt": make_prompt(board),
                "completion": f" {moves[i]}",
                "puzzle_id": row["PuzzleId"],
                "game_url": row.get("GameUrl"),
                "rating": row["Rating"],
                "ply": (i - 1) // 2,
            }
        )
        board.push(target)
        if i + 1 < len(moves):
            board.push_uci(moves[i + 1])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzles", type=int, default=3000, help="puzzles to fetch")
    ap.add_argument("--max-rating", type=int, default=1600, help="skip harder puzzles")
    ap.add_argument("--min-popularity", type=int, default=70)
    ap.add_argument("--max-per-puzzle", type=int, default=2, help="solution moves used per puzzle")
    ap.add_argument("--out", default="data/chess")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valid-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    print("reading puzzles from the Hub...", file=sys.stderr)
    pool = [
        r
        for r in fetch_puzzles()
        if r["Rating"] <= args.max_rating and r["Popularity"] >= args.min_popularity
    ]
    print(f"{len(pool)} puzzles pass the filters; sampling {args.puzzles}", file=sys.stderr)
    kept = rng.sample(pool, min(args.puzzles, len(pool)))

    # Split by puzzle so no position leaks between train and test.
    rng.shuffle(kept)
    n_test = int(len(kept) * args.test_frac)
    n_valid = int(len(kept) * args.valid_frac)
    splits = {
        "test": kept[:n_test],
        "valid": kept[n_test : n_test + n_valid],
        "train": kept[n_test + n_valid :],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        examples = [ex for r in rows for ex in puzzle_examples(r, args.max_per_puzzle)]
        rng.shuffle(examples)
        with (out / f"{name}.jsonl").open("w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{name}: {len(rows)} puzzles -> {len(examples)} examples", file=sys.stderr)


if __name__ == "__main__":
    main()
