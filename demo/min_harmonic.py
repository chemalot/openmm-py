#!/usr/bin/env python
###############################################################################
## The MIT License
##
## SPDX short identifier: MIT
##
## Copyright 2019 Genentech Inc. South San Francisco
##
## Permission is hereby granted, free of charge, to any person obtaining a
## copy of this software and associated documentation files (the "Software"),
## to deal in the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included
## in all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
## OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS IN THE SOFTWARE.
###############################################################################


import os
import sys
import openmm_py
import numpy as np
import torch
import torch.nn as nn
import simtk.openmm as mm

from sys import stdout, exit
from time import sleep
from simtk.openmm import app
from simtk.openmm import CustomNonbondedForce
from simtk import unit as u



def warn(*argv):
    # write to stderr
    print(*argv, file=sys.stderr, flush=True)


def createSystem(topology):
    # initil version taken from: simtk/openmm/app/forcefield.py
    sys = mm.System()
    for atom in topology.atoms():
        # Add the particle to the OpenMM system.
        #mass = self._atomTypes[typename].mass
        mass = atom.element.mass
        sys.addParticle(mass)

    # Set periodic boundary conditions.
    boxVectors = topology.getPeriodicBoxVectors()
    if boxVectors is not None:
        sys.setDefaultPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2])

    return sys


def Minimize(simulation, outFile, iters=0):
    simulation.minimizeEnergy(tolerance=0.001, maxIterations=iters)
    position = simulation.context.getState(getPositions=True).getPositions()
    energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
    with open(outFile, 'w') as outF:
        app.PDBFile.writeFile(simulation.topology, position, outF)
    warn( 'Energy at Minima is {:3.3f}'.format(energy._value))
    return simulation


class HarmonicModule(nn.Module):
    """ PyTorch Module computing a simple energy for a set of particle.
        The energy is computed as the square of the distance from the origin.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
       """ Compute energy as sum of distance to origin 
           x: tensor with xyz coordinates for all particles 1-n):
              x1,y1,z1, x2, ...., zn
       """
       # convert to matrix of xyz
       x = x.reshape((-1,3))

       # Energy of harmonic is sum(square of distances)
       return torch.sum(x * x)


class EnergyComputer(openmm_py.PyCall):
    """ This class implements the PyCall C++ interface.
        It's computeEnergyAndForces() method will be used as a callback from C++
        to compute energies and forces.
    """
    def __init__(self, nGPU=1):
        warn("PY__INIT EnergyComputer")
        super().__init__()

        self.device  = torch.device("cuda" if nGPU > 0 and torch.cuda.is_available() else "cpu")
        self.model   = HarmonicModule().to(self.device);
        
        self.model.eval()


    def computeEnergyAndForces(self, positions, includeForces, includeEnergy):
        """ positions: atomic postions in [nM]
                       numparitcal * 3 PyCall.FloatVector with
                       x1, y1, z1, x2, ... zn coordinates passed to us from openMM
            includeForces: boolean, if True force computation is requested.
            includeEnergy: boolean, if True energy computation is requested. 

            return: PyCall.NNPResult with energy [kJ/mol] and forces [kJ/mol/nm]
        """

        pos = np.array(positions, dtype=np.float32)
        warn("py computeEnergyAndForces {} positions[nm] {}"
             .format(type(positions),pos))

        # convert to pytorch tensor
        coords = torch.from_numpy(pos).to(device=self.device)

        # if forces are requested pytorch needs to know
        if includeForces: coords.requires_grad_(True)

        # compute energy
        pred = self.model.forward(coords)
        
        if includeForces:
            # use PyTorch autograd to compute:
            #     force = - derivative of enrgy wrt. coordinates
            pred.backward()
            forces = -coords.grad.cpu().numpy()
        else:
            forces = np.zeros(len(positions))


        # Return result in type openmm_py.NNPResult
        # this is a C++ struct with two fields: energy [kJ/Mol/nM]
        #    energy [kJ/Mol]
        #    force [kJ/Mol/nM]
        res = openmm_py.NNPResult();
        res.energy = pred.cpu().item()
        res.force = openmm_py.FloatVector(forces.tolist());
        warn("Py Energy {} Forces {}".format(pred,forces))

        return res;



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description="")


    parser.add_argument('-in', help='Molecule to minimize',
                        dest='inFile', metavar='pdb' ,  type=str, required=True)

    parser.add_argument('-out', help='optimized molecule output',
                        dest='outFile', metavar='pdb' ,  type=str, required=True)

    parser.add_argument('-nGPU' ,  metavar='n' ,  type=int, default=0,
                        help='number of GPUs, (currently only one supported)')


    args = parser.parse_args()

    # make parsed parameter local variables
    locals().update(args.__dict__)


    temperature = 298.15 * u.kelvin
    pdb = app.PDBFile(inFile)
    
    modeller = app.Modeller(pdb.topology, pdb.positions)
    
    topo = modeller.topology
    system = createSystem( modeller.topology )
    atomNum = []
    for atom in topo.atoms():
        atomNum.append(atom.element.atomic_number)
    warn(atomNum)
    
    
    #################################################
    # add PY force to system
    ecomputer = EnergyComputer(nGPU)

    #################################################
    # test ecomputer.computeEnergyAndForces
    #p = openmm_py.FloatVector([1,0,0, 0,0,0.96, 0,0,-0.028]);
    #ecomputer.computeEnergyAndForces(p, True, True)
    
    f = openmm_py.PYForce(ecomputer)
    system.addForce(f)
    #################################################

    
    integrator = mm.LangevinIntegrator(
        temperature, 1 / u.picosecond,  0.0005 * u.picoseconds)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation = Minimize(simulation,outFile,1000)
