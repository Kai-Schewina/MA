from __future__ import absolute_import
from __future__ import print_function

import os
import sys
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Runtime 1-2h
# Multiprocessing speeds it up to about 20min in my case

from subject import read_stays, read_diagnoses, read_events, get_events_for_stay,\
    add_hours_elpased_to_events
from subject import convert_events_to_timeseries, get_first_valid_from_timeseries
from preprocessing import read_itemid_to_variable_map, map_itemids_to_variables, clean_events
from preprocessing import assemble_episodic_data


def extract_episodes(subjects_root_path, var_map, variables, subject_dir):
    dn = os.path.join(subjects_root_path, subject_dir)
    try:
        subject_id = int(subject_dir)
        if not os.path.isdir(dn):
            raise Exception
    except:
        print("Subject Directory " + str(subject_dir) + " does not exist.")
        return None

    try:
        stays = read_stays(dn)
        events = read_events(dn)
    except Exception as e:
        print(e)
        sys.stderr.write('Error reading from disk for subject: {}\n'.format(subject_id))
        return None

    episodic_data = assemble_episodic_data(stays)

    # cleaning and converting to time series
    events = map_itemids_to_variables(events, var_map)
    events = clean_events(events)
    if events.shape[0] == 0:
        # no valid events for this subject
        return None
    timeseries = convert_events_to_timeseries(events, variables=variables)

    # extracting separate episodes
    for i in range(stays.shape[0]):
        stay_id = stays.stay_id.iloc[i]
        intime = stays.intime.iloc[i]
        outtime = stays.outtime.iloc[i]

        episode = get_events_for_stay(timeseries, stay_id, intime, outtime)
        if episode.shape[0] == 0:
            # no data for this episode
            continue

        episode = add_hours_elpased_to_events(episode, intime).set_index('hours').sort_index(axis=0)
        if stay_id in episodic_data.index:
            episodic_data.loc[stay_id, 'weight'] = get_first_valid_from_timeseries(episode, 'weight')
            episodic_data.loc[stay_id, 'height'] = get_first_valid_from_timeseries(episode, 'height')
        episodic_data.loc[episodic_data.index == stay_id].to_csv(os.path.join(subjects_root_path, subject_dir,
                                                                              'episode{}.csv'.format(i + 1)),
                                                                 index_label='Icustay')
        columns = list(episode.columns)
        columns_sorted = sorted(columns, key=(lambda x: "" if x == "hours" else x))
        episode = episode[columns_sorted]
        episode.to_csv(os.path.join(subjects_root_path, subject_dir, 'episode{}_timeseries.csv'.format(i + 1)),
                       index_label='hours')


def main(subjects_root_path):
    var_map = read_itemid_to_variable_map('../resources/itemid_to_variable_map.csv')
    variables = var_map.variable.unique()

    pool = mp.Pool(processes=mp.cpu_count() - 1)

    subject_list = os.listdir(subjects_root_path)
    for _ in tqdm(pool.imap(partial(extract_episodes, subjects_root_path, var_map, variables), subject_list),
                  total=len(subject_list)):
        pass


if __name__ == "__main__":
    main("../data/output/")
