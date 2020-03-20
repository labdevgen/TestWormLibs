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

# dump data to .expected file or reload from cash
def dump(file, juicerpath, resolution, minchrsize = 20000000,
         excludechrms = ("X","chrX","Y","chrY","Z","chrZ","W","chrW")
         ):
    description = (file+str(resolution)+str(minchrsize)+str(excludechrms)).encode("utf-8")

    hashfile = os.path.join("data/all_chr_dumps",
                            md5(description).hexdigest())
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
        description_chr = (file+str(resolution)+str(minchrsize)+str(excludechrms)+chr).encode("utf-8")
        chromosome_hash_file = os.path.join("data/all_chr_dumps/",
                            md5(description_chr).hexdigest())
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
def computeExpected(data):
    pass

def getExpectedByCompartments(file, juicerpath, resolution,
                              minchrsize = 20000000,
                    excludechrms = ("X","chrX","Y","chrY","Z","chrZ","W","chrW"),
                    compartments_file = None,
                    correlation_standard = None):
    strawObj = straw.straw(file)

    valid_chrms = get_vaid_chrms_from_straw(strawObj, minchrsize, excludechrms)
    if compartments_file is None:
        # compute comparmtns on the fly
        raise NotImplementedError

    compartments = pd.read_csv(compartments_file,
                               sep="\t",
                               header=None,
                               names=["chr","st","end","E1"])
    compartments.dropna(inplace=True)
    assert len(np.unique((compartments["end"]-compartments["st"]).values))==1
    assert abs (resolution - (compartments["end"]-compartments["st"]).values[0]) <= 1
    compartments.drop("end", inplace=True)

    for chr in valid_chrms:
        assert chr in compartments.chr.values
        # get contacts from straw
        hic_chr, X1, X2 = strawObj.chromDotSizes.figureOutEndpoints(chr)
        matrxObj = strawObj.getNormalizedMatrix(hic_chr, hic_chr, "KR",
                                                "BP", resolution)

        assert matrxObj is not None
        contacts = matrxObj.getDataFromGenomeRegion(X1, X2, X1, X2)
        contacts = pd.DataFrame(list(contacts), columns="[chr1,st1,en1,chr2,st2,en2,count]")
        contacts = contacts.drop(["en1","en2"],inplace=True)

        E1chr = compartments.query("chr==@chr")

        contacts = contacts.merge(E1chr,how="innder",left_on="st1",right_on="st").rename(
            columns={"E1":"E1_st"})
        contacts = contacts.merge(E1chr,how="innder",left_on="st2",right_on="st").rename(
            columns={"E1":"E1_st2"})

        # ready to compute expected