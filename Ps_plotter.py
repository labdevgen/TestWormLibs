import numpy as np
import matplotlib.pyplot as plt
import logging
logging.basicConfig(level = logging.INFO)
import pandas as pd
from fit_functions import plot_ps, fit_linear_regression, fit_ps_log_bins
from plot_functions import multiplots, multiplot_with_subplots
from Expected_calculator import dump

def process(data):
    # normalize probabilities to 1
    for chr in data:
        # last expected values on chrms are similar
        # let's clip expected values at this point
        t = np.subtract(data[chr][1:], data[chr][:-1])
        t = np.where(t==0)[0]
        if len(t) != 0:
            t = max(t)
            assert t > 3*len(data)/4
            data[chr] = data[chr][:t]

        # now normalize the sum to 1
        data[chr] = data[chr] / np.sum(data[chr])
    return data

def row2color(row):
    colors={"Anopheles":"springgreen",
            "Drosophila":"springgreen",
#            "culex": "limegreen",
            "culex": "springgreen",
            "Aedes": "springgreen",
            "Polypedium" : "springgreen",
            "chick":"springgreen",
            "mammals":"springgreen",
            "chick_Dekker":"springgreen"
    }

    special_styles = {
            "CME" : {"linestyle": "-", "color":"blue", "linewidth":2},
            "CIE": {"marker":"*","linestyle":":"},

            "Aedes": {"color":"orange", "linewidth":1},
            "Culex": {"color": "lightseagreen","linewidth":1},
            "RaoCondensinDegron": {"color": "blue", "linewidth":2},
            "2017Haarhuis_KO_WAPL1" : {"color":  "red",  "linewidth":2},
            "2017Haarhuis_Kontrol": {"color":  "red", "linestyle":"--", "linewidth":0.5},
            "LiverTAM": {"color": "yellow", "linestyle": "--", "linewidth": 1},
            "LiverNipbl": {"color": "red", "linestyle": "--", "linewidth": 1},
            "BonevCN":{"color":"purple"},
#            "DekkerCapH-": {"color":"red","marker" : "*", "linestyle":"--"},
#            "DekkerCAPHControl": {"color": "red", "marker" : "*"},
#            "DekkerCapH2-": {"color":"salmon", "marker":"^", "linestyle" : "--"},
#            "DekkerCAPH2Control": {"color": "salmon", "marker":"^"},
#            "DekkerSMC2-":{"color": "red"},
#            "DekkerSMC2Control": {"color": "red", "linestyle":"--"},
            "DekkerPrometo":{"color":"yellow", "linewidth":2},
            "DekkerSMC2-":{"linestyle":"-", "color":"red", "linewidth":2},
            "DekkerCapH-":{"linestyle":"--", "color":"red", "linewidth":1},
            "DekkerCapH2-":{"linestyle":":", "color":"red", "linewidth":1},
            "DmelCAP":{"linestyle":"-", "color":"red", "linewidth":2},
            "DmelRAD": {"linestyle": "-", "color":"blue", "linewidth":2},
#            "SextonDrosophila":{"linestyle": "-", "color":"black"},
#            "Kc167rowley": {"linestyle": "-", "color": "blue"},
#            "S2": {"linestyle": "-", "color": "red"},
#            "S2HeatShock": {"linestyle": "--", "color": "red"}
#        "Dvir": {"color": "white"},
#        "Dmel": {"color": "blue"}
#        "Dbus": {"color": "white"},
    }

    result = {"linewidth": 0.5}
    result["color"]=colors[row.subtaxon]
    if row["name"] in special_styles:
        for k,v in special_styles[row["name"]].items():
            result[k] = v

    return result

dataset = "datasets.csv"
juicer_tools = "juicer/juicer_tools_1.19.02.jar"
resolution = 10000
report = 170

datasets = pd.read_csv(dataset, sep="\t",
                       comment="#")

analysis = {
#    "Anopheles": datasets.query("(name in ['Acol','Amer','Aste','Aalb','Aatr'])"),
#    "test": datasets.query("(name in ['Acol','Amer'])")
#    "Other_insects": datasets.query("(subtaxon=='Drosophila' or subtaxon=='culex' or name=='Aedes')")
#    "mammals": datasets.query("(subtaxon=='mammals')"),
#    "chicken": datasets.query("(subtaxon=='chick')")
#    "Nipbl": datasets.query("(name in ['LiverWT','LiverTAM','LiverNipbl'])")
#    "Aedes":  datasets.query("name=='Aedes'")
#    "mammals_test": datasets.query("(name=='BonevNPC')")
    "all_maps_from_Gibcus_et_al": datasets.query("subtaxon=='chick_Dekker'")
}

multiplot = False # draw all graphs on one plot or draw multiple subplots

#for func in ["Ps_log"]:
#for func in ["Ps"]:
for func in ["Slope"]:
#for func in ["Ps","Slope"]:
    for suffix,species in analysis.items():
        plots = {}
        for ind in range(len(species)):
            row = species.iloc[ind]
            logging.info(row["name"])
            hic = row.link
            logging.info("Starting dump")
            data = dump(hic,juicer_tools,resolution)
            data = process(data)
            logging.info("Fitting...")
            # X,Y = fit_power_low(data)
            if row.taxon == "vertebrate":
                crop_min = -2.2
                crop_max = 0.1
                max_plot_dist = 35000000
#                max_plot_dist = 350000000
            else:
                crop_min = -1.75
                crop_max = -0.25
                max_plot_dist = 25000000
#                max_plot_dist = 45000000

            if func == "Slope":
                X,Y = fit_linear_regression(data, resolution=resolution,
                                            crop_min=crop_min, crop_max=crop_max, max_plot_dist=max_plot_dist)
            elif func == "Ps":
                X, Y = plot_ps(data, resolution=resolution)
            elif func == "Ps_log":
                X, Y = fit_ps_log_bins(data, resolution=resolution, logbase1 = 1.8, logbase2= 1.8)
            else:
                raise
            logging.info("Plotting...")
            plots[ind] = [pd.DataFrame({"X":X,"Y":Y}),row2color(row),row["name"],row["Genotype"]]
            logging.info("Done!")
            if ind >= report and ind % report == 0:
                #plt.show()
                break
        if multiplot:
            #multiplots(plots, shadow=(func=="Slope"), average=(func=="Slope"))
            #multiplots(plots, shadow=False, average=False)
            if func=="Slope":
                plt.gca().set_ylabel("Slope")
            elif func=="Ps":
                plt.gca().set_ylabel("Log(Normed Contact probability)")

            plt.gca().set_xlabel("Genomic distance")
        else:
            multiplot_with_subplots(plots, xlabel="Genomic distance", y_label="Slope")
        plt.tight_layout()
        plt.savefig("result_" + suffix +"_" + func +"_" + str(multiplot) + ".png", dpi=500)
        plt.clf()