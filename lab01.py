import pandas as pd
import numpy as np

df = pd.read_csv("E:/hak/DuLieu/dulieuxettuyendaihoc.csv")

#5.1
# Tần số
freq_T1 = df['T1'].value_counts(dropna=False)

# Tần suất
freq_rate_T1 = df['T1'].value_counts(normalize=True, dropna=False)

print("Bảng tần số T1:")
print(freq_T1)

print("\nBảng tần suất T1:")
print(freq_rate_T1)

# số lượng dữ liệu thiếu
df['T1'].isnull().sum()

#5.2 thay thế
mean_T1 = df['T1'].mean()
df['T1'] = df['T1'].fillna(mean_T1)

df['T1'].isnull().sum()

#6
diem_cols = [
    'T1','L1','H1','S1','V1','X1','D1','N1',
    'T2','L2','H2','S2','V2','X2','D2','N2',
    'T6','L6','H6','S6','V6','X6','D6','N6',
    'DH1','DH2','DH3'
]
for col in diem_cols:
    df[col] = df[col].fillna(df[col].mean())

#7
df['TBM1'] = (df['T1']*2 + df['L1'] + df['H1'] + df['S1'] +
              df['V1']*2 + df['X1'] + df['D1'] + df['N1']) / 10

df['TBM2'] = (df['T2']*2 + df['L2'] + df['H2'] + df['S2'] +
              df['V2']*2 + df['X2'] + df['D2'] + df['N2']) / 10

df['TBM3'] = (df['T6']*2 + df['L6'] + df['H6'] + df['S6'] +
              df['V6']*2 + df['X6'] + df['D6'] + df['N6']) / 10

#8
def xep_loai(tbm):
    if tbm < 5:
        return 'Y'
    elif tbm < 6.5:
        return 'TB'
    elif tbm < 8:
        return 'K'
    elif tbm < 9:
        return 'G'
    else:
        return 'XS'

df['XL1'] = df['TBM1'].apply(xep_loai)
df['XL2'] = df['TBM2'].apply(xep_loai)
df['XL3'] = df['TBM3'].apply(xep_loai)

#9
def min_max_4(series):
    return (series - series.min()) / (series.max() - series.min()) * 4

df['US_TBM1'] = min_max_4(df['TBM1'])
df['US_TBM2'] = min_max_4(df['TBM2'])
df['US_TBM3'] = min_max_4(df['TBM3'])

#10
def ket_qua_xt(row):
    kt = row['KT']
    
    if kt in ['A', 'A1']:
        diem = (row['DH1']*2 + row['DH2'] + row['DH3']) / 4
    elif kt == 'B':
        diem = (row['DH1'] + row['DH2']*2 + row['DH3']) / 4
    else:
        diem = (row['DH1'] + row['DH2'] + row['DH3']) / 3
        
    return 1 if diem >= 5 else 0

df['KQXT'] = df.apply(ket_qua_xt, axis=1)

#11
df.to_csv("processed_dulieuxettuyendaihoc.csv", index=False)
