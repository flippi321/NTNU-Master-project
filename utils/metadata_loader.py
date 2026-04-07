import os
import pandas as pd

class MetaDataLoader:
    def __init__(self, root_path="data/metadata"):
        self.data_root = root_path
        self._combined = None
        self._index = {}

    def _get_id_from_path(self, path: str) -> str:
        filename = os.path.basename(path)
        return filename.split('_')[0]

    def _load(self):
        if self._combined is not None:
            return

        hunt3 = pd.read_csv(os.path.join(self.data_root, "HUNT3.csv"))
        hunt4 = pd.read_excel(os.path.join(self.data_root, "HUNT4.xlsx"))

        hunt3 = hunt3.rename(columns={
            "MR_HUNT_ID": "hunt_3_long_id",
            "Age_at_time_of_MRI": "age_hunt3"
        })
        hunt4 = hunt4.rename(columns={
            "HUNT4 MRI Participant number": "hunt4_id",
            "Long HUNT3 numbers": "hunt_3_long_id",
            "Sex": "sex",
            "Age at HUNT4 MRI": "age_hunt4"
        })

        self._combined = hunt4.merge(hunt3[["hunt_3_long_id", "age_hunt3"]], on="hunt_3_long_id", how="inner")

        self._combined["hunt_id"] = self._combined["hunt_3_long_id"].apply(lambda x: str(int(x))[-5:])
        self._combined["sex"] = self._combined["sex"].map({"M": 0, "F": 1})
        self._combined["age_hunt3"] = (self._combined["age_hunt3"] - self._combined["age_hunt3"].min()) / (self._combined["age_hunt3"].max() - self._combined["age_hunt3"].min())
        self._combined["age_hunt4"] = (self._combined["age_hunt4"] - self._combined["age_hunt4"].min()) / (self._combined["age_hunt4"].max() - self._combined["age_hunt4"].min())

        # Build O(1) lookup index
        self._index = self._combined.set_index("hunt_id").to_dict(orient="index")

    def get(self, hunt_path, long_id=False, sex=False, age_hunt3=False, age_hunt4=False, labeled=False):
        self._load()
        hunt_id = self._get_id_from_path(hunt_path)
        row = self._index.get(hunt_id)
        if row is None:
            print(f"No metadata found for hunt_id: {hunt_id}")
            return None
        result = {}
        if long_id:   result["hunt4_id"]  = row["hunt4_id"]
        if sex:       result["sex"]        = row["sex"]
        if age_hunt3: result["age_hunt3"]  = row["age_hunt3"]
        if age_hunt4: result["age_hunt4"]  = row["age_hunt4"]
        return result if labeled else list(result.values())

    def get_many(self, hunt_paths, long_id=False, sex=False, age_hunt3=False, age_hunt4=False):
        return [self.get(hpath, long_id=long_id, sex=sex, age_hunt3=age_hunt3, age_hunt4=age_hunt4) for hpath in hunt_paths]