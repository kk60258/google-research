import pandas as pd

RAW_LABEL_PATH = '/home/jason/Downloads/Arisan_20210222_draw_BBox/DVR-798-CH1_20201123180600_20201123182600_res/label.csv'
TARGET_PATH = RAW_LABEL_PATH.replace('.csv', '_combine_class12.csv')

df = pd.read_csv(RAW_LABEL_PATH, dtype={'ImageID': str})

for index, row in df.iterrows():
  class1 = row.Class1 if row.Class1 != '/m/unset' else '/m/player'
  class2 = '_' + row.Class2[3:]
  df.at[index, 'LabelName'] = class1 + class2

df.to_csv(TARGET_PATH, index=False)
