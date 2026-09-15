import pandas as pd

def read_wells(path):
    return pd.read_csv(path)

def read_layers(path):
    return pd.read_excel(path)

def read_pumping_test(path):
    return pd.read_csv(path, sep="\t")

def print_table_info(name, table):
    print(f"\n{name}")
    print(table.shape)
    print(table.columns)
    print(table.dtypes)
    print(table.head())

class DatasetInfo:
    def __init__(self, name, table):
        self.name = name
        self.rows, self.columns = table.shape

    def describe(self):
        return f"{self.name}: {self.rows} строк, {self.columns} столбцов"