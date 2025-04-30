import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter

from mbml.data import MBMLData

data = MBMLData(data_path="data/raw/0013db25-4444-452b-980b-7702dc6fb810.json")

events = data._get_kills_by_round(15)

df = pd.DataFrame(events)

# Get all unique player names
all_players = pd.unique(df[["attackerName", "victimName"]].values.ravel())

# Build player records
survival_records = []

# Max round time (or just take max seen in data)
max_time = df["seconds"].max()

for player in all_players:
    # Get player death, if any
    death_event = df[df["victimName"] == player]

    if not death_event.empty:
        # Player died: use first death time
        time = death_event["seconds"].min()
        event = 1
        team = death_event.iloc[0]["victimTeam"]
    else:
        # Player survived
        time = max_time
        event = 0
        # Guess team from attacker side
        attacker_rows = df[df["attackerName"] == player]
        team = attacker_rows.iloc[0]["attackerTeam"] if not attacker_rows.empty else "Unknown"

    survival_records.append({"player": player, "duration": time, "event": event, "team": team})

survival_df = pd.DataFrame(survival_records)


kmf = KaplanMeierFitter()

plt.figure(figsize=(10, 6))

for team in survival_df["team"].unique():
    mask = survival_df["team"] == team
    kmf.fit(durations=survival_df[mask]["duration"], event_observed=survival_df[mask]["event"], label=team)
    kmf.plot_survival_function()

plt.title("Survival Curves per Team")
plt.xlabel("Time (seconds)")
plt.ylabel("Probability of Survival")
plt.grid(True)
plt.show()
