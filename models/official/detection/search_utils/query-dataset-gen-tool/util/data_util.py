import pandas as pd
import os

def read_catalog():
    file_path = os.path.join('search_utils/query-dataset-gen-tool/temp/FK_catalog_04_01.csv')
    data = pd.read_csv(file_path, header=0)
#     data = pd.read_csv(file_path, header=0, skiprows=lambda i: i % 500 != 0)
    print("FK Catalog has", len(data),"items")
    return data
