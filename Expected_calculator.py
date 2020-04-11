import straw
import pickle
import subprocess
import os
from hashlib import md5
import pandas as pd
import numpy as np
import logging

def get_vaid_chrms_from_straw(strawObj,
                            minchrsize,
                            excludechrms,
                            maxchrsize=1000000000000):
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
                chrsizes[chr][1] < maxchrsize and \
                not chr in excludechrms and \
                not chr.lower() in excludechrms and \
                not chr.upper() in excludechrms:
            valid_chrms.append(chr)

    assert len(valid_chrms) > 0
    return valid_chrms

def get_dumpPath(*args, root=".", create_dirs = False):
    description = "".join((map(str,args))).encode("utf-8")
    res = os.path.join(root,md5(description).hexdigest())
    if create_dirs and not os.path.isdir(os.path.dirname(res)):
        os.makedirs(os.path.dirname(res))
    return res


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
            retrie_number = 0
            while retrie_number < 10:
                try:
                    p = subprocess.run(" ".join(command),shell=True,
                                    stdout = subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    check=True)
                    break
                except subprocess.CalledProcessError as e:
                    print(e.output)
                    retrie_number += 1

        chr_data = np.loadtxt(chromosome_hash_file)
        data[chr] = chr_data

    pickle.dump(data, open(hashfile,"wb"))
    return data

# compute expected from the dframe st1 - st2 - E1
def computeExpected(data, resolution):
    result = data.groupby(by="dist").aggregate(np.nanmean)
    # it might be that not all distance values are present. Add NaNs
    print(result.head())
    print(result.dtypes)

    print(resolution)
    all_possible_values = pd.DataFrame({"all_dist":range(0,int(max(result.index.values))+1,
                                                      resolution)})
    result = result.merge(all_possible_values,left_index=True,
                          right_on = "all_dist", how="outer").sort_index()["count"].fillna(0.)
    return result

def get_contacts_using_juicer_dump(juicerpath,file,chr,resolution):
    # dump contacts from chromosome to temp file, then read if to Dframe
    command = ["java", "-jar", juicerpath, "dump", "observed",
               "KR", file, chr, chr, "BP", str(resolution), "temp.contacts"]
    print(" ".join(command))
    try:
        subprocess.run(" ".join(command), shell=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=True)
    except subprocess.CalledProcessError as e:
        print(e)
        exit(1)
    contacts = pd.read_csv("temp.contacts", sep="\t", header=None, names=["st1", "st2", "count"],
                           dtype={"st1": np.uint32, "st2": np.uint32, "count": np.float64})
    contacts["dist"] = contacts["st2"] - contacts["st1"]
    assert np.all(contacts["dist"].values >= 0)
    return contacts

def getExpectedByCompartments(file, juicerpath, resolution,
                              minchrsize = 20000000,
                              maxchrsize = 150000000,
                    excludechrms = ("X","chrX","Y","chrY","Z","chrZ","W","chrW"),
                    compartments_file = None,
                    correlation_standard = None,
                    save_by_chr_dumps = False):
    if pd.isnull(compartments_file) or pd.isna(compartments_file):
        compartments_file = None
    dump_path = get_dumpPath([file,resolution,minchrsize,excludechrms,compartments_file,
                                  correlation_standard],
                             root="data/Expected_by_compartment/",
                            create_dirs=True)
    if os.path.isfile(dump_path):
        return pickle.load(open(dump_path,"rb"))

    all_expected_hashfile = get_dumpPath(file,resolution,minchrsize,excludechrms,
                            root="data/all_chr_dumps",
                            create_dirs=True)

    # it takes a long time to get list of chrms frow hic-files located in cloud
    # let's try to load it from expected dumps
    if os.path.isfile(all_expected_hashfile) and compartments_file is not None:
        valid_chrms=list(pickle.load(open(all_expected_hashfile,"rb")).keys())
    else:
        strawObj = straw.straw(file)
        valid_chrms = get_vaid_chrms_from_straw(strawObj, minchrsize, excludechrms)

    if compartments_file is None:
        # compute compartments on the fly
        compartments_file = get_dumpPath(file,resolution,
                                         root="data/Expected_by_compartment/")+".E1"
        if not os.path.isfile(compartments_file):
            out = open(compartments_file,"a")
            for chr in valid_chrms:
                chr_file_dump = compartments_file+"."+chr+".juicerout"
                cmd=list(map(str,["java","-jar",juicerpath,"eigenvector","-p","KR",
                        file, chr,"BP",resolution,chr_file_dump]))
                print (" ".join(cmd))
                subprocess.run(" ".join(cmd), shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               check=True)
                lines=open(chr_file_dump).readlines()
                for ind,val in enumerate(lines):
                    out.write(str(ind*resolution) +"\t"+str((ind+1)*resolution) + "\t" + val)
                out.write("\n")

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
        print ("Processing chromosome ",chr)
        # check that E1 were computed for this chrm
        E1chr = compartments.query("chr==@chr")
        if len(E1chr) == 0:
            print("Warning: no compartments for chr", chr)
            continue

        # get contacts from straw
        # hic_chr, X1, X2 = strawObj.chromDotSizes.figureOutEndpoints(chr)
        # matrxObj = strawObj.getNormalizedMatrix(hic_chr, hic_chr, "KR",
        #                                         "BP", resolution)
        #
        # assert matrxObj is not None
        # contacts = matrxObj.getDataFromGenomeRegion(X1, X2, X1, X2)
        # contacts = pd.DataFrame({"st1":contacts[0],"st2":contacts[1],"count":contacts[2]})
        # contacts["st1"] = contacts["st1"]*resolution
        # contacts["st2"] = contacts["st2"]*resolution
        # contacts.dropna(inplace=True)

        contacts = get_contacts_using_juicer_dump(juicerpath,
                            file,chr,resolution)
        contacts = contacts.merge(E1chr,how="inner",left_on="st1",right_on="st").rename(
            columns={"E1":"E1_st1"})
        print (contacts.dtypes)
        contacts = contacts.merge(E1chr,how="inner",left_on="st2",right_on="st").rename(
            columns={"E1":"E1_st2"})
        print (contacts.dtypes)

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

def getExpected(file, juicerpath, resolution,
                              minchrsize = 20000000,
                    excludechrms = ("X","chrX","Y","chrY","Z","chrZ","W","chrW"),
                    save_by_chr_dumps = True):

    # get dump path and create dump dirs
    dump_path = get_dumpPath([file,resolution,minchrsize,excludechrms],
                             root="data/Expected/",
                             create_dirs=True)

    # search for dump and load if found
    if os.path.isfile(dump_path):
        return pickle.load(open(dump_path,"rb"))

    all_expected_hashfile = get_dumpPath(file,resolution,minchrsize,excludechrms,
                            root="data/all_chr_dumps",
                             create_dirs=True)

    # it takes a long time to get list of chrms frow hic-files located in cloud
    # let's try to load it from expected dumps
    if os.path.isfile(all_expected_hashfile):
        valid_chrms=list(pickle.load(open(all_expected_hashfile,"rb")).keys())
    else:
        strawObj = straw.straw(file)
        valid_chrms = get_vaid_chrms_from_straw(strawObj, minchrsize, excludechrms)

    results = {}
    label = "all_contacts"  # not to be confused with AA-AB-BA expected generated by other funcs
    results[label] = {}
    for chr in valid_chrms:
        print ("Processing chromosome ",chr)
        contacts = get_contacts_using_juicer_dump(juicerpath,
                            file,chr,resolution)
        # compute expected
        Exp = computeExpected(contacts, resolution)
        if save_by_chr_dumps:
                chr_dump_path = get_dumpPath([file, resolution, minchrsize, excludechrms, chr, label],
                                             root="data/Expected_by_compartment/",
                                             create_dirs = True
                                            ) +"_"+ label+"_"+chr
                Exp.to_csv(chr_dump_path,sep="\t",header=False,index=False)
        results["all_contacs"][chr] = Exp.values
    pickle.dump(results, open(dump_path,"wb"))

    return results