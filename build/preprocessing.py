from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from pandas import DataFrame, Series
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

###############################
# Non-time series build
###############################

g_map = {'F': 1, 'M': 2, 'OTHER': 3, '': 0}


def transform_gender(gender_series):
    global g_map
    return {'Gender': gender_series.fillna('').apply(lambda s: g_map[s] if s in g_map else g_map['OTHER'])}


e_map = {'ASIAN': 1,
         'BLACK': 2,
         'CARIBBEAN ISLAND': 2,
         'HISPANIC': 3,
         'SOUTH AMERICAN': 3,
         'WHITE': 4,
         'MIDDLE EASTERN': 4,
         'PORTUGUESE': 4,
         'AMERICAN INDIAN': 0,
         'NATIVE HAWAIIAN': 0,
         'UNABLE TO OBTAIN': 0,
         'PATIENT DECLINED TO ANSWER': 0,
         'UNKNOWN': 0,
         'OTHER': 0,
         '': 0}


def transform_ethnicity(ethnicity_series):
    global e_map

    def aggregate_ethnicity(ethnicity_str):
        return ethnicity_str.replace(' OR ', '/').split(' - ')[0].split('/')[0]

    ethnicity_series = ethnicity_series.apply(aggregate_ethnicity)
    return {'Ethnicity': ethnicity_series.fillna('').apply(lambda s: e_map[s] if s in e_map else e_map['OTHER'])}


def assemble_episodic_data(stays):
    data = {'Icustay': stays.stay_id, 'Age': stays.age, 'Length of Stay': stays.los,
            'Mortality': stays.mortality}
    data.update(transform_gender(stays.gender))
    data.update(transform_ethnicity(stays.ethnicity))
    data['height'] = np.nan
    data['weight'] = np.nan
    data = DataFrame(data).set_index('Icustay')
    data = data[['Ethnicity', 'Gender', 'Age', 'height', 'weight', 'Length of Stay', 'Mortality']]
    return data


###################################
# Time series build
###################################

def read_itemid_to_variable_map(fn, variable_column='level2'):
    var_map = pd.read_csv(fn, encoding="utf-8", sep=";").fillna('').astype(str)
    var_map.columns = map(str.lower, var_map.columns)
    var_map[variable_column] = var_map[variable_column].apply(lambda s: s.lower())
    var_map["count"] = var_map["count"].astype(int)
    var_map = var_map[(var_map[variable_column] != '') & (var_map["count"] > 0)]

    # Important ! Subsets only rows of the file where status is set to ready !
    var_map = var_map[(var_map.status == 'ready')]
    var_map.itemid = var_map.itemid.astype(int)
    var_map = var_map[[variable_column, 'itemid', 'mimic label']].set_index('itemid')
    return var_map.rename({variable_column: 'variable', 'mimic label': 'mimic_label'}, axis=1)


def map_itemids_to_variables(events, var_map):
    return events.merge(var_map, left_on='itemid', right_index=True)


# SBP: some are strings of type SBP/DBP
def clean_sbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(1))
    return v.astype(float)


def clean_dbp(df):
    v = df.value.astype(str).copy()
    idx = v.apply(lambda s: '/' in s)
    v.loc[idx] = v[idx].apply(lambda s: re.match('^(\d+)/(\d+)$', s).group(2))
    return v.astype(float)


# CRR: strings <3 normal or >3 abnormal
def clean_crr(df):
    v = Series(np.zeros(df.shape[0]), index=df.index)
    v[:] = np.nan
    df_value_str = df.value.astype(str)
    v.loc[(df_value_str == 'Normal <3 secs')] = 1.0
    v.loc[(df_value_str == 'Abnormal >3 secs')] = 2.0
    return v.astype(float)


# FIO2: many 0s, some 0<x<0.2 or 1<x<20
def clean_fio2(df):
    v = df.value.astype(float).copy()

    ''' The line below is the correct way of doing the cleaning, since we will not compare 'str' to 'float'.
    If we use that line it will create mismatches from the data of the paper in ~50 ICU stays.
    The next releases of the benchmark should use this line.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (v>1.0)

    ''' The line below was used to create the benchmark dataset that the paper used. Note this line will not work
    in python 3, since it may try to compare 'str' to 'float'.
    '''
    # idx = df.VALUEUOM.fillna('').apply(lambda s: 'torr' not in s.lower()) & (df.VALUE > 1.0)

    ''' The two following lines implement the code that was used to create the benchmark dataset that the paper used.
    This works with both python 2 and python 3.
    '''
    is_str = np.array(map(lambda x: type(x) == str, list(df.value)), dtype=np.bool)
    idx = df.valueuom.fillna('').apply(lambda s: 'torr' not in s.lower()) & (is_str | (~is_str & (v > 1.0)))

    v.loc[idx] = v[idx] / 100.
    v[v > 1] = v / 100
    return v


# GLUCOSE, PH: sometimes have ERROR as value
def clean_lab(df):
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)


# O2SAT: small number of 0<x<=1 that should be mapped to 0-100 scale
def clean_o2sat(df):
    # change "ERROR" to NaN
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan

    v = v.astype(float)
    idx = (v <= 1)
    v.loc[idx] = v[idx] * 100.
    return v


# Temperature: map Farenheit to Celsius, some ambiguous 50<x<80
def clean_temperature(df):
    v = df.value.astype(float).copy()
    idx = df.valueuom.fillna('').apply(lambda s: 'F' in s.lower()) | df.mimic_label.apply(lambda s: 'F' in s.lower()) | (v >= 79)
    v.loc[idx] = (v[idx] - 32) * 5. / 9
    return v


# Weight: some really light/heavy adults: <50 lb, >450 lb, ambiguous oz/lb
# Children are tough for height, weight
def clean_weight(df):
    v = df.value.astype(float).copy()
    # ounces
    idx = df.valueuom.fillna('').apply(lambda s: 'oz' in s.lower()) | df.mimic_label.apply(lambda s: 'oz' in s.lower())
    v.loc[idx] = v[idx] / 16.
    # pounds
    idx = idx | df.valueuom.fillna('').apply(lambda s: 'lb' in s.lower()) | df.mimic_label.apply(lambda s: 'lb' in s.lower())
    v.loc[idx] = v[idx] * 0.453592
    return v


# Height: some really short/tall adults: <2 ft, >7 ft)
# Children are tough for height, weight
def clean_height(df):
    v = df.value.astype(float).copy()
    idx = df.valueuom.fillna('').apply(lambda s: 'in' in s.lower()) | df.mimic_label.apply(lambda s: 'in' in s.lower())
    v.loc[idx] = np.round(v[idx] * 2.54)
    return v


# Positive end-expiratory pressure
# Afaik no need to clean it
def clean_peep(df):
    v = df.value.astype(float).copy()
    return v


def clean_hr_bpm_pao2_paco2_rr(df):
    v = df.value.astype(float).copy()
    return v


def clean_pulse(df):
    v = df.value.astype(str).copy()
    v[(v == 'Absent')] = 0.0
    v[(v == 'Strong/Palpable') | (v == 'Difficult to Palpate') | (v == 'Difficult Palpate')] = 1.0
    v[(v == 'Easily Palpable') | (v == 'Weak Palpable') | (v == 'Weak Palpate') | (v == 'Weak PalPable')] = 2.0
    v[(v == 'Doppler')] = 3.0
    v[(v == 'Not Applicable')] = np.nan
    return v.astype(float)


def clean_lactate(df):
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)


def clean_inr(df):
    v = df.value.copy()
    idx = v.apply(lambda s: type(s) is str and not re.match('^(\d+(\.\d*)?|\.\d+)$', s))
    v.loc[idx] = np.nan
    return v.astype(float)


clean_fns = {
    'capillary refill rate': clean_crr,
    'diastolic blood pressure': clean_dbp,
    'systolic blood pressure': clean_sbp,
    'fraction inspired oxygen': clean_fio2,
    'oxygen saturation': clean_o2sat,
    'glucose': clean_lab,
    'pH': clean_lab,
    'temperature': clean_temperature,
    'weight': clean_weight,
    # 'height': clean_height,
    # self-added
    'positive end-expiratory pressure': clean_peep,
    'heart rate': clean_hr_bpm_pao2_paco2_rr,
    'mean blood pressure': clean_hr_bpm_pao2_paco2_rr,
    'partial pressure of oxygen': clean_hr_bpm_pao2_paco2_rr,
    'partial pressure of carbon dioxide': clean_hr_bpm_pao2_paco2_rr,
    'respiratory rate': clean_hr_bpm_pao2_paco2_rr,
    'pulse': clean_pulse,
    # additional labs
    # albumin, bicarbonate, bilirubin, creatinine, hemoglobin, platelets, rbc, troponin, wbc is okay
    'hematocrit': clean_lab,
    'lactate': clean_lactate,
    'prothrombin time': clean_inr,
    'red blood cell distribution width': clean_inr,

}


def clean_events(events):
    global clean_fns
    for var_name, clean_fn in clean_fns.items():
        idx = (events.variable == var_name)
        df = events[idx]
        if df.shape[0] > 0:
            try:
                events.loc[idx, 'value'] = clean_fn(events[idx])
            except Exception as e:
                import traceback
                print("Exception in clean_events:", clean_fn.__name__, e)
                print(traceback.format_exc())
                print("number of rows:", np.sum(idx))
                print("values:", events[idx])
                exit()
    return events.loc[events.value.notnull()]
