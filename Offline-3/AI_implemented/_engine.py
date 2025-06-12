"""
engine.py – backend process:
• waits for a 'Human Move:' in gamestate.txt
• replies with its own move and overwrites the file with 'AI Move:'
"""

import time, os, sys
import core, ai

FILE = "gamestate.txt"

def read_file_waiting_for(header: str) -> list[str]:
    """Block until the file starts with the desired header."""
    while True:
        try:
            with open(FILE, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            time.sleep(0.1); continue

        if lines and lines[0].strip() == header:
            return lines[1:]                  # return board only
        time.sleep(0.1)                       # wait & retry


def write_state(header: str, state: core.GameState):
    data = [header] + state.to_lines()
    with open(FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(data))


def main():
    print("AI Engine ready – waiting for human moves…")
    agent = ai.MinimaxAgent(player=2, depth=3)   # Blue

    # If file doesn't exist, create an empty board so GUI has something
    if not os.path.exists(FILE):
        empty = core.GameState(rows=9, cols=6)
        write_state("AI Move:", empty)

    while True:
        # 1. wait for human move
        board_lines = read_file_waiting_for("Human Move:")
        state = core.GameState.from_file(board_lines)
        state.current_player = 2               # it's AI's turn now

        if state.game_over:
            write_state("AI Move:", state)     # pass through final board
            print("Human already won – exiting.")
            break

        # 2. choose AI move
        r, c = agent.choose_move(state.clone())
        state.apply_move(r, c)                 # mutate state

        # 3. save result
        write_state("AI Move:", state)

        # 4. terminate if the AI just won
        if state.game_over:
            print("AI wins – exiting.")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
