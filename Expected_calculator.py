import straw
import pickle
import subprocess
import os
from hashlib import md5
import pandas as pd
import numpy as np

def get_vaid_chrms_from_straw(strawObj,
                            minchrsize,
                            excludechrms):
    # get chrm names first
    chrsizes = strawObj.chromDotSizes.data
    # example:
    # {'ALL': (0, 231617), 'X': (1, 26840812), '2R': (2, 61076761), '2L': (3, 48179558), '3R': (4, 53444961),
    # '3L': (5, 42074947)}

    if "ALL" in chrsizes:
        del chrsizes["ALL"]

    # get valid chrms
    valid_chrms = []

    for chr in chrsizes:
        if chrsizes[chr][1] > minchrsize and \
                not chr in excludechrms and \
                not chr.lower() in excludechrms and \
                not chr.upper() in excludechrms:
            valid_chrms.append(chr)

    assert len(valid_chrms) > 0
    return valid_chrms

def get_dumpPath(*args, root="."):
    description = "".join((map(str,args))).encode("utf-8")
    return os.path.join(root,md5(description).hexdigest())

# dump data to .expected file or reload from cash
def dump(file, juicerpath, resolution, minchrsize = 20000000,
         excludechrms = ("X","chrX","Y","chrY","Z","chrZ","W","chrW")
         ):

    hashfile = get_dumpPath(file,resolution,minchrsize,excludechrms,
                            root="data/all_chr_dumps")
    if os.path.isfile(hashfile):
        return pickle.load(open(hashfile,"rb"))

    strawObj = straw.straw(file)
    valid_chrms = get_vaid_chrms_from_straw(strawObj, minchrsize, excludechrms)
    # maxlenid = sorted(chrsizes.keys(),
    #                   key=lambda x: chrsizes[x][1],
    #                   reverse=True)[0]
    # valid_chrms = [maxlenid]
    # logging.info("Using chr "+str(maxlenid)+" for file "+file)

    # dump data for all valid chrms
    data = {}
    for chr in valid_chrms:
        chromosome_hash_file = get_dumpPath(file,resolution,minchrsize,excludechrms,chr,
                                            root="data/all_chr_dumps/")
        if not os.path.isfile(chromosome_hash_file):
            command = ["java","-jar",juicerpath,"dump","expected",
                        "KR",file,chr,"BP",str(resolution),chromosome_hash_file]
            logging.info(" ".join(command))
            p = subprocess.run(" ".join(command),shell=True,
                                    stdout = subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    check=True)
        chr_data = np.loadtxt(chromosome_hash_file)
        data[chr] = chr_data

    pickle.dump(data, open(hashfile,"wb"))
    return data

# compute expected from the dframe st1 - st2 - E1
def computeExpected(data, resolution):
    result = data.groupby(by="dist").aggregate(np.nanmean)
    # it might be that not all distance values are present. Add NaNs
    all_possible_values =  pd.DataFrame({"all_dist":range(0,max(result.index.values)+1,
                                                      resolution)})
    result = result.merge(all_possible_values,left_index=True,
                          right_on = "all_dist", how="outer").sort_index()["count"].fillna(0.)
    return result

def getExpectedByCompartments(file, juicerpath, resolution,
                              minchrsize = 20000000,
                    excludechrms = ("X","chrX","Y","chrY","Z","chrZ","W","chrW"),
                    compartments_file = None,
                    correlation_standard = None,
                    save_by_chr_dumps = False):
    dump_path = get_dumpPath([file,resolution,minchrsize,excludechrms,compartments_file,
                                  correlation_standard],
                             root="data/Expected_by_compartment/")
    if os.path.isfile(dump_path):
        return pickle.load(open(dump_path,"rb"))

    strawObj = straw.straw(file)
    valid_chrms = get_vaid_chrms_from_straw(strawObj, minchrsize, excludechrms)
    if compartments_file is None:
        # compute compartments on the fly
        raise NotImplementedError
        compartments_file = get_dumpPath(file,resolution,
                                         root="data/Expected_by_compartment/")+".E1"
        if not os.path.isfile(compartments_file):
            out = open(compartments_file,"w")
            for chr in valid_chrms:
                chr_file_dump = compartments_file+"."+chr
                cmd=["java","-jar",juicerpath,"eigenvector","KR",
                        file, chr,"BP",resolution,chr_file_dump]
                print (" ".join(cmd))
                subprocess.run(" ".join(cmd), shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               check=True)
                lines =open(chr_file_dump).readlines()
                out.write(lines)

    compartments = pd.read_csv(compartments_file,
                               sep="\t",
                               header=None,
                               names=["chr","st","end","E1"])
    compartments.dropna(inplace=True)
    before_drop = len(compartments)
    A_percentile = np.percentile(compartments.query("E1>=0").E1,10)
    B_percentile = np.percentile(compartments.query("E1<=0").E1,90)
    compartments.query("E1>=@A_percentile or E1<=@B_percentile", inplace=True)
    after_drop = len(compartments)
    print ("Dropped ",before_drop-after_drop," out of ",before_drop," unsertain compartmental bins")
    assert len(np.unique((compartments["end"]-compartments["st"]).values))==1
    assert abs (resolution - (compartments["end"]-compartments["st"]).values[0]) <= 1
    compartments.drop("end", axis="columns", inplace=True)

    results = {"A":{},"B":{},"AB":{},"all":{}}
    for chr in valid_chrms:

        # check that E1 were computed for this chrm
        E1chr = compartments.query("chr==@chr")
        if len(E1chr) == 0:
            print("Warning: no compartments for chr", chr)
            continue

        # get contacts from straw
        hic_chr, X1, X2 = strawObj.chromDotSizes.figureOutEndpoints(chr)
        matrxObj = strawObj.getNormalizedMatrix(hic_chr, hic_chr, "KR",
                                                "BP", resolution)

        assert matrxObj is not None
        contacts = matrxObj.getDataFromGenomeRegion(X1, X2, X1, X2)
        contacts = pd.DataFrame({"st1":contacts[0],"st2":contacts[1],"count":contacts[2]})
        contacts["st1"] = contacts["st1"]*resolution
        contacts["st2"] = contacts["st2"]*resolution
        contacts.dropna(inplace=True)

        contacts = contacts.merge(E1chr,how="inner",left_on="st1",right_on="st").rename(
            columns={"E1":"E1_st1"})
        contacts = contacts.merge(E1chr,how="inner",left_on="st2",right_on="st").rename(
            columns={"E1":"E1_st2"})
        contacts["dist"] = contacts["st2"]-contacts["st1"]
        assert np.all(contacts["dist"].values >= 0)

        # ready to compute expected
        A_interactions = contacts.query("E1_st1 > 0 and E1_st2 > 0")
        B_interactions = contacts.query("E1_st1 < 0 and E1_st2 < 0")
        AB_interactions = contacts.query("E1_st1 * E1_st2 < 0")

        for label,contact_data in zip(
                    ["A","B","AB","all"],
                    [A_interactions,B_interactions,AB_interactions,contacts]
                    ):
            Exp = computeExpected(contact_data, resolution)
            results[label][chr] = Exp.values

            if save_by_chr_dumps:
                chr_dump_path = get_dumpPath([file, resolution, minchrsize, excludechrms, compartments_file,
                                              correlation_standard, chr, label],
                                             root="data/Expected_by_compartment/") +"_"+ label+"_"+chr
                Exp.to_csv(chr_dump_path,sep="\t",header=False,index=False)
    assert len(results) != 0
    pickle.dump(results, open(dump_path,"wb"))
    return results