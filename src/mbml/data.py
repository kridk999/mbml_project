import json
import lzma


class MBMLData:
    def __init__(self, data_path: str = None):
        self.data_path = data_path

        with lzma.open(self.data_path, "rt", encoding="utf-8") as f:
            self.data = json.load(f)

        self._add_tickseconds()

    def _get_round(self, round_number: int):
        assert round_number > 0, "Round number must be greater than 0"
        if round_number > len(self.data["gameRounds"]):
            print(f"Round number {round_number} exceeds the number of rounds in the data. Returning the last round.")
            round_number = len(self.data["gameRounds"])

        return self.data["gameRounds"][round_number - 1]

    def _convert_ticks_to_seconds(self, ticks: int, tick_rate: int = 128) -> float:
        return ticks / tick_rate

    def _get_kills_second(self, round_number: int):
        start_tick_of_round = self._get_round(round_number)["freezeTimeEndTick"]

        kill_dict = {}
        for kills in self._get_kills_by_round(round_number):
            kill_dict[kills["tick"]] = {
                "time_in_seconds": self._convert_ticks_to_seconds(kills["tick"] - start_tick_of_round),
                "killer": kills["attackerName"],
                "victim": kills["victimName"],
            }
        return kill_dict

    def _get_kills_by_round(self, round_number: int):
        round_data = self._get_round(round_number)
        kills = round_data.get("kills", [])
        return kills

    def _was_bomb_planted(self, round_number: int):
        t = False
        for row in data._get_round(round_number)["bombEvents"]:
            if row["bombAction"] == "plant":
                t = True
        return t

    def _add_tickseconds(self):
        for round in self.data["gameRounds"]:
            for kills in round["kills"]:
                kills["tickSeconds"] = self._convert_ticks_to_seconds(kills["tick"] - round["freezeTimeEndTick"])

    def _get_buys(self, round_number: int):
        round_data = self._get_round(round_number)
        buys = {"tTeam": round_data["tRoundSpendMoney"], "ctTeam": round_data["ctRoundSpendMoney"]}
        return buys

    def __len__(self):
        return len(self.data["gameRounds"])

    def __getitem__(self, key):
        return self.data[key]


if __name__ == "__main__":
    data = MBMLData("data/lan/03e1f233-579c-462d-ac0e-1635d4718ef8.json.xz")

    print(data)
