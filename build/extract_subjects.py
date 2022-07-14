from __future__ import absolute_import
from __future__ import print_function

from mimic3csv import *
from util import dataframe_from_csv

# Runtime: Slightly more than an hour


def main(mimic4_path, output_path, event_tables=['outputevents', 'labevents', 'chartevents'], verbose=True, test=False):

    os.makedirs(output_path, exist_ok=True)
    
    patients = read_patients_table(mimic4_path)
    admits = read_admissions_table(mimic4_path)
    stays = read_icustays_table(mimic4_path)

    if verbose:
        print('START:\n\ticustay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
              stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))
    
    stays = remove_icustays_with_transfers(stays)
    if verbose:
        print('REMOVE ICU TRANSFERS:\n\ticustay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
              stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))

    stays = merge_on_subject_admission(stays, admits)
    stays = merge_on_subject(stays, patients)
    stays = filter_admissions_on_nb_icustays(stays)
    if verbose:
        print('REMOVE MULTIPLE STAYS PER ADMIT:\n\ticustay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
              stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))
    
    stays = add_age_to_icustays(stays)
    stays = add_inunit_mortality_to_icustays(stays)
    stays = add_inhospital_mortality_to_icustays(stays)
    stays = filter_icustays_on_age(stays)
    if verbose:
        print('REMOVE PATIENTS AGE < 18:\n\ticustay_ids: {}\n\thadm_ids: {}\n\tsubject_ids: {}'.format(stays.stay_id.unique().shape[0],
              stays.hadm_id.unique().shape[0], stays.subject_id.unique().shape[0]))
    
    stays.to_csv(os.path.join(output_path, 'all_stays.csv'), index=False)
    
    if test:
        pat_idx = np.random.choice(patients.shape[0], size=1000)
        patients = patients.iloc[pat_idx]
        stays = stays.merge(patients[['subject_id']], left_on='subject_id', right_on='subject_id')
        event_tables = [event_tables[0]]
        print('Using only', stays.shape[0], 'stays and only', event_tables[0], 'table')
    
    subjects = stays.subject_id.unique()
    break_up_stays_by_subject(stays, output_path, subjects=subjects)

    itemids_file = False
    items_to_keep = set(
        [int(itemid) for itemid in dataframe_from_csv(itemids_file)['itemid'].unique()]) if itemids_file else None
    for table in event_tables:
        read_events_table_and_break_up_by_subject(mimic4_path, table, output_path, items_to_keep=items_to_keep,
                                                  subjects_to_keep=subjects)


if __name__ == "__main__":
    main(mimic4_path="E:\mimic-iv-1.0", output_path="../data/output_v2/", test=False)
