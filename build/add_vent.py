import os
from tqdm import tqdm
import pandas as pd
import util
import multiprocessing as mp
from functools import partial
from sqlalchemy import create_engine
pd.options.mode.chained_assignment = None


def get_vent_data():
    engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')
    ventilation = pd.read_sql_query('select * from mimic_derived.ventilation', con=engine)
    return ventilation


def add_vent(subj, ventilation):
    patient_ts_files = list(filter(lambda x: x.find("timeseries") != -1, os.listdir(subj)))
    for index, ts in enumerate(patient_ts_files):
        path = os.path.join(subj, ts)

        # Load Data
        stay = pd.read_csv(os.path.join(subj, "stays.csv"))
        episode = pd.read_csv(path)

        # Calculate Ventilation Details
        stay_start = pd.to_datetime(stay.loc[index, 'intime'])
        stay_end = pd.to_datetime(stay.loc[index, 'outtime'])

        vent = ventilation[ventilation['stay_id'] == stay.loc[index, 'stay_id']]

        # Convert Dict to Categories
        vent_dict = {
            "Tracheostomy": 2,
            "InvasiveVent": 2,
            "NonInvasiveVent": 1,
            "SupplementalOxygen": 1,
            "None": 0,
            "HFNC": 1
        }
        episode['vent'] = 0
        if len(vent) > 0:
            vent["starttime"] = pd.to_datetime(vent["starttime"])
            vent["endtime"] = pd.to_datetime(vent["endtime"])

            vent = vent.loc[(vent['starttime'] >= stay_start) & (vent['starttime'] <= stay_end),]

            for i, row in vent.iterrows():
                vent_start = vent.loc[i, "starttime"]
                vent_end = vent.loc[i, "endtime"]
                vent_hour_start = (vent_start - stay_start)
                vent_hour_start = vent_hour_start.days * 24 + vent_hour_start.seconds / 3600
                vent_hour_end = (vent_end - stay_start)
                vent_hour_end = vent_hour_end.days * 24 + vent_hour_end.seconds / 3600

                # Add ventilation to ts
                episode.loc[(vent_hour_start <= episode['hours']) &
                            (vent_hour_end >= episode["hours"]), 'vent'] = vent_dict[vent.loc[i, "ventilation_status"]]
        episode.to_csv(path, index=False)


def main():
    path = "../data/output/"
    subjects = util.get_subjects(path)
    ventilation = get_vent_data()
    for subj in tqdm(subjects):
        add_vent(subj, ventilation)


if __name__ == "__main__":
    main()
