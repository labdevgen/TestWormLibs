import pandas as pd
from Expected_calculator import getExpectedByCompartments
from plot_functions import simple_plot, multiplots
from fit_functions import plot_ps, fit_linear_regression
import matplotlib.pyplot as plt

dataset = "datasets.csv"
juicer_tools = "juicer/juicer_tools_1.19.02.jar"
resolution = 25000


datasets = pd.read_csv(dataset, sep="\t",
                       comment="#")
row = datasets.query("name=='Acol'").iloc[0]
hic = row.link
compartments = row.compartments
result = getExpectedByCompartments(hic,juicer_tools,25000,compartments_file=compartments,
                                 save_by_chr_dumps=True)
colors = {"A":"red",
          "B":"blue",
          "all":"black",
          "AB":"green"}
plots = {}
for label,data in result.items():
    one_chr = list(data.keys())[0]
    data2 = {one_chr:data[one_chr]}
    # X, Y = plot_ps(data2, resolution, maxdist=20000000)
    # print (X[:5],Y[:5],data[one_chr][:5])
    # simple_plot(X,Y,color=colors[label],linewidth=0.5)

    if row.taxon == "vertebrate":
        crop_min = -2.2
        crop_max = 0.1
    else:
        crop_min = -1.75
        crop_max = -0.25
    X,Y = fit_linear_regression(data2, resolution=resolution,
                                crop_min=crop_min, crop_max=crop_max, max_plot_dist=25000000)
    plots[label] = [pd.DataFrame({"X":X,"Y":Y}),{"color":colors[label],"linewidth":0.5},label,"WT"]

multiplots(plots,shadow=False, average=False)
plt.show()