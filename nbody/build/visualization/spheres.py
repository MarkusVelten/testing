#! /usr/bin/env python3

import vtk
import csv
import os
import glob
import re
from multiprocessing import cpu_count
from joblib import Parallel, delayed

def create_sphere(PosX, PosY, PosZ, force_p, size, mode="theta", size_factor=0.00001):
    s = vtk.vtkSphereSource()
    s.SetRadius(size * size_factor)
    # s.SetRadius(1)
    s.SetCenter(PosX, PosY, PosZ)
    if mode == "theta":
        s.SetStartTheta(0)
        s.SetEndTheta(force_p * 360.0)
    elif mode == "phi":
        s.SetStartPhi(0)
        s.SetEndPhi(force_p * 180.0)
    s.Update()

    ss = vtk.vtkSphereSource()
    ss.SetRadius(size * size_factor)
    # ss.SetRadius(1)
    ss.SetCenter(PosX, PosY, PosZ)
    if mode == "theta":
        ss.SetStartTheta(force_p * 360.0)
        ss.SetEndTheta(360.0)
    elif mode == "phi":
        ss.SetStartPhi(force_p * 180.0)
        ss.SetEndPhi(180.0)
    ss.Update()

    return (s, ss)

# centers == positions
# force_data == forces
def generate_spheres(HForce, CForce, LForce, PosX, PosY, PosZ, force):
    #data_theta1 = vtk.vtkAppendPolyData()
    #data_theta2 = vtk.vtkAppendPolyData()
    data_phi1 = vtk.vtkAppendPolyData()
    data_phi2 = vtk.vtkAppendPolyData()

    for i in range(len(PosX)):

        ForceSum = HForce[i] + CForce[i] + LForce[i]
        if force[i] == 0:
            RelForce = 0
        else:
            RelForce = force[i] / ForceSum

        #t, tt = create_sphere(PosX[i], PosY[i], PosZ[i], abs(RelForce), 1)
        p, pp = create_sphere(PosX[i], PosY[i], PosZ[i], RelForce, 1, "phi")
        #data_theta1.AddInputData(t.GetOutput())
        #data_theta2.AddInputData(tt.GetOutput())
        data_phi1.AddInputData(p.GetOutput())
        data_phi2.AddInputData(pp.GetOutput())

    #data_theta1.Update()
    #data_theta2.Update()
    data_phi1.Update()
    data_phi2.Update()

    #return  (data_theta1, data_theta2, data_phi1, data_phi2)
    return  (data_phi1, data_phi2)
    #return  (data_theta1, data_theta2)

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

    #if infile == '../data100.csv':
        #print(PosX, len(PosX))

    # extract timestep
    timestep = re.findall(r'\d+', infile)

    #t, tt, p, pp = generate_spheres(HForces, CForces, LForces, \
    #t, tt = generate_spheres(HForces, CForces, LForces, \
    p, pp = generate_spheres(HForces, CForces, LForces, \
        PosX, PosY, PosZ, HForces)
    #write_vtp("spheres_theta1_" + "HForces" + str(timestep[0]) + ".vtp", t)
    #write_vtp("spheres_theta2_" + "HForces" + str(timestep[0]) + ".vtp", tt)
    write_vtp("spheres_phi1_" + "HForces" + str(timestep[0]) + ".vtp", p)
    write_vtp("spheres_phi2_" + "HForces" + str(timestep[0]) + ".vtp", pp)

    #t, tt, p, pp = generate_spheres(HForces, CForces, LForces, \
    #t, tt = generate_spheres(HForces, CForces, LForces, \
    p, pp = generate_spheres(HForces, CForces, LForces, \
        PosX, PosY, PosZ, CForces)
    #write_vtp("spheres_theta1_" + "CForces" + str(timestep[0]) + ".vtp", t)
    #write_vtp("spheres_theta2_" + "CForces" + str(timestep[0]) + ".vtp", tt)
    write_vtp("spheres_phi1_" + "CForces" + str(timestep[0]) + ".vtp", p)
    write_vtp("spheres_phi2_" + "CForces" + str(timestep[0]) + ".vtp", pp)

    #t, tt, p, pp = generate_spheres(HForces, CForces, LForces, \
    #t, tt = generate_spheres(HForces, CForces, LForces, \
    p, pp = generate_spheres(HForces, CForces, LForces, \
        PosX, PosY, PosZ, LForces)
    #write_vtp("spheres_theta1_" + "LForces" + str(timestep[0]) + ".vtp", t)
    #write_vtp("spheres_theta2_" + "LForces" + str(timestep[0]) + ".vtp", tt)
    write_vtp("spheres_phi1_" + "LForces" + str(timestep[0]) + ".vtp", p)
    write_vtp("spheres_phi2_" + "LForces" + str(timestep[0]) + ".vtp", pp)


def main():

    num_cores = cpu_count()
    # load each result file for each time step and do the postprocessing on it
    path = '../'

    Parallel(n_jobs = num_cores)(delayed(work)(infile) \
        for infile in glob.glob(os.path.join(path, 'data*.csv')))




if __name__ == "__main__":
    main()

