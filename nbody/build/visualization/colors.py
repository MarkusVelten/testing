#! /usr/bin/env python3

import csv
import os
import glob
import re
from multiprocessing import cpu_count
from joblib import Parallel, delayed
import colorsys


def calculateColors(HForce, CForce, LForce, HForceMax, CForceMax, LForceMax):

    color = []

    for i in range(len(HForce)):

        HForceRGB = int(HForce[i]/HForceMax * 255)
        CForceRGB = int(CForce[i]/CForceMax * 255)
        LForceRGB = int(LForce[i]/LForceMax * 255)

        # colorsys functions take values in range [0..1]
        #ForcesHSV = colorsys.rgb_to_hsv(HForceRGB, CForceRGB, LForceRGB)

        color.append(f"{HForceRGB:03}"+f"{CForceRGB:03}"+f"{LForceRGB:03}")

    return (color)

# read csv result files from nbody simulation
# and return arrays for positions and forces
def read_csv(file):

    PosX = []
    PosY = []
    PosZ = []

    HForces = []
    CForces = []
    LForces = []

    with open(file) as FILE:
        inf = csv.reader(FILE)

        for row in inf:
            PosX.append(float(row[0]))
            PosY.append(float(row[1]))
            PosZ.append(float(row[2]))

            HForces.append(float(row[3]))
            CForces.append(float(row[4]))
            LForces.append(float(row[5]))

    return ( PosX, PosY, PosZ, HForces, CForces, LForces )


def work(infile, maxX, HForceMax, CForceMax, LForceMax):

    # read csv input file and extract data to arrays
    PosX, PosY, PosZ, HForces, CForces, LForces = read_csv(infile)

    # extract timestep
    timestep = re.findall(r'\d+', infile)

    # for individual particle
    color = calculateColors([HForces[maxX]], [CForces[maxX]], [LForces[maxX]], HForceMax, CForceMax, LForceMax)

    with open ("singleData" + str(timestep[0]) + ".csv", "w") as singleFile:
        singleFileWriter = csv.writer(singleFile, delimiter = ',')
        singleFileWriter.writerow([PosX[maxX], PosY[maxX], PosZ[maxX], color[0]])


    # the other particles
    # remove dedicated particle from remaining bunch
    del PosX[maxX]
    del PosY[maxX]
    del HForces[maxX]
    del CForces[maxX]
    del LForces[maxX]


    color = calculateColors(HForces, CForces, LForces, HForceMax, CForceMax, LForceMax)
    with open ("multipleData" + str(timestep[0]) + ".csv", "w", newline='') as multipleFile:
        multipleFileWriter = csv.writer(multipleFile, delimiter = ',')
        for i in range(len(PosX)):
            multipleFileWriter.writerow([PosX[i], PosY[i], PosZ[i], color[i]])


def main():

    num_cores = cpu_count()
    # load each result file for each time step and do the postprocessing on it
    path = '../'

    # find particle that is far away from center in first step
    PosX, PosY, PosZ, HForces, CForces, LForces = read_csv('../data0.csv')

    maxX = PosX.index( max(PosX) )

    # find largest force values
    HForceMax = 0
    CForceMax = 0
    LForceMax = 0

    for infile in glob.glob(os.path.join(path, 'data*.csv')):
        PosX, PosY, PosZ, HForces, CForces, LForces = read_csv(infile)

        HForceTempMax = max(HForces)
        CForceTempMax = max(CForces)
        LForceTempMax = max(LForces)

        HForceMax = max(HForceTempMax, HForceMax)
        CForceMax = max(CForceTempMax, CForceMax)
        LForceMax = max(LForceTempMax, LForceMax)

    # execute postprocessing in parallel
    Parallel(n_jobs = num_cores)(delayed(work)\
        (infile, maxX, HForceMax, CForceMax, LForceMax) \
        for infile in glob.glob(os.path.join(path, 'data*.csv')))


if __name__ == "__main__":
    main()

