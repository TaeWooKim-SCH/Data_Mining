import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel("G:/내 드라이브/김태우/University/2학년 2학기/데이터마이닝/Assignment/Data/아파트(매매)_실거래가_20221104160600.xlsx", skiprows = 16)
df['계약일자'] = pd.to_datetime(df['계약년월'].astype(str) + df['계약일'].astype(str).str.zfill(2), format = '%Y%m%d')
print(df['계약일자'])

df[['계약일자']]