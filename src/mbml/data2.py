import os
import json
import lzma
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data/lan"

def load_match(filepath):
    with lzma.open(filepath, "rt", encoding="utf-8") as f:
        return json.load(f)

def extract_frames_from_round(round_data, round_idx, match_id):
    frames_data = []
    winner = round_data["winningSide"] # "CT" or "T"

    # Basic round-level info
    base_info = {
        "round_idx": round_idx,
        "ctTeam": round_data["ctSide"]['teamName'],
        "tTeam": round_data["tSide"]['teamName'],
        "ctBuyType": round_data["ctBuyType"],
        "tBuyType": round_data["tBuyType"],
        "ctEqVal": round_data["ctRoundStartEqVal"],
        "tEqVal": round_data["tRoundStartEqVal"],
        "ctSpend": round_data["ctRoundSpendMoney"],
        "tSpend": round_data["tRoundSpendMoney"],
        "rnd_winner": winner,
        "match_id": match_id,
        "team1_score": [round_data["ctScore"] if round_data["roundNum"]<16 else round_data["tScore"]][0],
        "team2_score": [round_data["tScore"] if round_data["roundNum"]<16 else round_data["ctScore"]][0],
    }

    for frame in round_data['frames']:
        tick = frame.get("tick")
        seconds = frame.get("seconds")
        clock = frame.get("clockTime")

        # You can extract more state info here if needed
        frame_info = {
            "tick": tick,
            "seconds": seconds,
            "clock": clock,
            "bomb_planted": frame.get("bombPlanted", False),
            "num_util_ct": frame.get("ct", {}).get("totalUtility", 0),
            "num_util_t": frame.get("t", {}).get("totalUtility", 0),
            "num_ct_alive": frame.get("ct", {}).get("alivePlayers", []),
            "num_t_alive": frame.get("t", {}).get("alivePlayers", []),
        }

        # Merge frame + round info
        frames_data.append({**base_info, **frame_info})

    return frames_data

def extract_frames_from_match(match_data, match_id):
    all_frame_data = []

    for idx in range(len(match_data['gameRounds'])):
        rnd = match_data['gameRounds'][idx]
        round_frames = extract_frames_from_round(rnd, idx, match_id)
        all_frame_data.extend(round_frames)

        # # Update match score state
        # winner = rnd.get("winningSide")
        # if winner == 'CT' :
        #     if match_data['gameRounds'][idx]['roundNum'] > 16:
        #         match_state['tTeam'] += 1
        #     else:
        #         match_state['ctTeam'] += 1
        # elif winner == 'T':
        #     if match_data['gameRounds'][idx]['roundNum'] > 16:
        #         match_state['ctTeam'] += 1
        #     else: match_state['tTeam'] += 1

    return all_frame_data

def load_all_matches_to_frames(data_dir):
    all_frames = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".json.xz")][:10]

    for file in tqdm(files, desc="Loading matches"):
        path = os.path.join(data_dir, file)
        match_id = os.path.splitext(os.path.basename(file))[0]
        try:
            match = load_match(path)
            frames = extract_frames_from_match(match, match_id)
            all_frames.extend(frames)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    return pd.DataFrame(all_frames)

if __name__ == "__main__":
    df = load_all_matches_to_frames(DATA_DIR)
    print(df.head())
    df.to_csv("round_frame_data.csv", index=False)