#! /usr/bin/env python3

import vtk
import csv
import os
import glob
import re
from multiprocessing import cpu_count
from joblib import Parallel, delayed

# centers == positions
# force_data == forces
def generate_spheres(HForce, CForce, LForce, PosX, PosY, PosZ):
    data_phiH = vtk.vtkAppendPolyData()
    data_phiC = vtk.vtkAppendPolyData()
    data_phiL = vtk.vtkAppendPolyData()

    size = 1
    size_factor = 0.00001

    for i in range(len(PosX)):

        ForceSum = HForce[i] + CForce[i] + LForce[i]
        if ForceSum == 0:
            RelHForce = 0
            RelCForce = 0
            RelLForce = 0
        else:
            RelHForce = HForce[i] / ForceSum
            RelCForce = CForce[i] / ForceSum
            RelLForce = LForce[i] / ForceSum

        # create sphere-part for harmonic forces
        sphereH = vtk.vtkSphereSource()
        sphereH.SetRadius(size * size_factor)

        sphereH.SetCenter(PosX[i], PosY[i], PosZ[i])

        sphereH.SetStartPhi(0)
        sphereH.SetEndPhi(RelHForce * 180.0)

        sphereH.Update()

        # create sphere-part for coulomb forces
        sphereC = vtk.vtkSphereSource()
        sphereC.SetRadius(size * size_factor)

        sphereC.SetCenter(PosX[i], PosY[i], PosZ[i])

        sphereC.SetStartPhi(RelHForce * 180.0)
        sphereC.SetEndPhi((RelHForce + RelCForce) * 180.0)

        sphereC.Update()

        # create sphere-part for cooling-laser forces
        sphereL = vtk.vtkSphereSource()
        sphereL.SetRadius(size * size_factor)

        sphereL.SetCenter(PosX[i], PosY[i], PosZ[i])

        sphereL.SetStartPhi((RelHForce + RelCForce) * 180.0)
        sphereL.SetEndPhi((RelHForce + RelCForce + RelLForce) * 180.0)

        sphereL.Update()

        data_phiH.AddInputData(sphereH.GetOutput())
        data_phiC.AddInputData(sphereC.GetOutput())
        data_phiL.AddInputData(sphereL.GetOutput())

    data_phiH.Update()
    data_phiC.Update()
    data_phiL.Update()

    return  (data_phiH, data_phiC, data_phiL)


def write_vtp(file, data):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file)
    writer.SetInputData(data.GetOutput())
    writer.Update()

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


def work(infile):

    # read csv input file and extract data to arrays
    PosX, PosY, PosZ, HForces, CForces, LForces = read_csv(infile)

    # extract timestep
    timestep = re.findall(r'\d+', infile)

    p, pp, ppp = generate_spheres(HForces, CForces, LForces, \
        PosX, PosY, PosZ)
    write_vtp("spheres_phi_" + "HForces" + str(timestep[0]) + ".vtp", p)
    write_vtp("spheres_phi_" + "CForces" + str(timestep[0]) + ".vtp", pp)
    write_vtp("spheres_phi_" + "LForces" + str(timestep[0]) + ".vtp", ppp)


def main():

    num_cores = cpu_count()
    # load each result file for each time step and do the postprocessing on it
    path = '../'

    Parallel(n_jobs = num_cores)(delayed(work)(infile) \
        for infile in glob.glob(os.path.join(path, 'data*.csv')))


if __name__ == "__main__":
    main()

