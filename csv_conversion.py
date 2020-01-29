import pandas
import os
dir_path='E:/Python/Python_Intern_Gavs_ML/data set/'
csv_path='E:/Python/Python_Intern_Gavs_ML/csv/x.csv'
pandas.Series([open(dir_path+f,errors='ignore').readlines() for f in os.listdir(dir_path)]).to_csv(csv_path)
