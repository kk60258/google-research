import pandas as pd

RAW_LABEL_PATH = '/home/jason/Downloads/people_detection/viveland/Viveland-records-20210422_images_300x300_left_right_v1/label_left_right_testing_0503.csv'
TARGET_PATH = RAW_LABEL_PATH.replace('.csv', '_class2_v2.csv')

df = pd.read_csv(RAW_LABEL_PATH, dtype={'ImageID': str})

for index, row in df.iterrows():
  class2 = '/m/staff' if row.Class1 == '/m/staff' else row.Class2
  df.at[index, 'Class1'] = '/m/player'
  df.at[index, 'Class2'] = class2

df.to_csv(TARGET_PATH, index=False)
