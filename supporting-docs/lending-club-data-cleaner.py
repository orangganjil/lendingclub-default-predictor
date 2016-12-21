#!/usr/bin/python

from __future__ import division
import pandas as pd
import numpy as np

# Read the file into pandas
df = pd.read_csv("/path/to/csv/files/provided by/LendingClub.csv")

# Keep only the fields where the loan status is pretty binary (fully paid, defaulted, or charged off)
fullypaid = df['loan_status'] == "Fully Paid"
charged_off = df['loan_status'] == "Charged Off"
defaulted = df['loan_status'] == "Default"

df2 = df[fullypaid | charged_off | defaulted]

# Replace the original dataframe with the new, smaller dataframe
df = df2

# Clean up null values
df['annual_inc'].fillna(df['annual_inc'].mean(), inplace=True)

# Convert some DataFrames to appropriate dtypes
df['member_id'] = df['member_id'].astype(str)
df['int_rate'] = df['int_rate'].str[:-1].astype(np.float64)
df['int_rate'] = df['int_rate'] / 100
df['annual_inc'] = df['annual_inc'].astype(int64)
df['revol_util'] = df['revol_util'].str[:-1].astype(np.float64)
df['revol_util'] = df['revol_util'] / 100


# Clean up NaNs and other empty fields
df['revol_util'].fillna(df['revol_util'].mean(), inplace=True)
df['bc_util'].fillna(df['bc_util'].mean(), inplace=True)
df['num_rev_accts'].fillna(0, inplace=True)
df['num_accts_ever_120_pd'].fillna(0, inplace=True)
df['num_tl_30dpd'].fillna(0, inplace=True)
df['num_tl_90g_dpd_24m'].fillna(0, inplace=True)
df['num_tl_op_past_12m'].fillna(0, inplace=True)
df['pct_tl_nvr_dlq'].fillna(df['pct_tl_nvr_dlq'].mean(), inplace=True)
df['loan_status'].fillna('missing', inplace=True)
df['loan_amnt'].fillna(0, inplace=True)
df['int_rate'].fillna(df['int_rate'].mean(), inplace=True)
df['revol_bal'].fillna(df['revol_bal'].mean(), inplace=True)
df['chargeoff_within_12_mths'].fillna(0, inplace=True)
df['fico_range_high'].fillna(df['fico_range_high'].mean(), inplace=True)
df['fico_range_low'].fillna(df['fico_range_low'].mean(), inplace=True)
df['mths_since_last_delinq'].fillna(0, inplace=True)
df['installment'].fillna(0, inplace=True)
df['inq_last_6mths'].fillna(0, inplace=True)
df['mths_since_recent_inq'].fillna(0, inplace=True)
df['mths_since_last_major_derog'].fillna(0, inplace=True)
df['acc_open_past_24mths'].fillna(0, inplace=True)
df['mort_acc'].fillna(0, inplace=True)
df['mths_since_recent_bc_dlq'].fillna(0, inplace=True)
df['num_tl_120dpd_2m'].fillna(0, inplace=True)
df['percent_bc_gt_75'].fillna(0, inplace=True)
df['tot_hi_cred_lim'].fillna(df['tot_hi_cred_lim'].mean(), inplace=True)
df['total_bal_ex_mort'].fillna(df['total_bal_ex_mort'].mean(), inplace=True)
df['total_bc_limit'].fillna(df['total_bc_limit'].mean(), inplace=True)
df['total_il_high_credit_limit'].fillna(df['total_il_high_credit_limit'].mean(), inplace=True)
df['tot_cur_bal'].fillna(df['tot_cur_bal'].mean(), inplace=True)
df['open_acc_6m'].fillna(0, inplace=True)
df['dti'].fillna(df['dti'].mean(), inplace=True)
df['avg_cur_bal'].fillna(0, inplace=True)


# Add the "default" target feature (1 = defaulted or charged off; 0 = not defaulted)
df['default'] = 0
df.loc[df['loan_status'].str.contains("Charged Off|Default"), 'default'] = 1


# Create monthly income feature
monthly_inc = df['annual_inc'] / 12
df['monthly_inc'] = monthly_inc

# Create feature for ratio of annual income to outstanding principal
df['payment_ratio'] = (df['installment'] / df['monthly_inc'])
df['payment_ratio'].fillna(0, inplace=True)


# Encode some string values for use in ML models
from sklearn.preprocessing import LabelEncoder
var_mod = ['term','grade']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes


# Check for null values
for c in df:
    isnull = df[c].isnull().sum()
    print "%s = %s" % (c,isnull)

# Save the cleaned, prepped file for use in the ML model
df.to_csv("/path to save file/loans.csv")


# Pickle the label encoder for future use.
from sklearn.externals import joblib

joblib.dump(le, "/LendingClubData/pickled-models/lc-label-encoder.pkl", compress=3) 
