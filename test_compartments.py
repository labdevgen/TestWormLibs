import pandas as pd
from Expected_calculator import getExpectedByCompartments

dataset = "datasets.csv"
juicer_tools = "juicer/juicer_tools_1.19.02.jar"
resolution = 25000

datasets = pd.read_csv(dataset, sep="\t",
                       comment="#")
row = datasets.iloc[0]
hic = row.link
compartments = row.compartments
getExpectedByCompartments(hic,juicer_tools,25000,compartments_file=compartments,
                                 save_by_chr_dumps=True)