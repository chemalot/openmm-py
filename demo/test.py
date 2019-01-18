#!/usr/bin/env python
from simtk.openmm import app, KcalPerKJ
import simtk.openmm as mm
from simtk.openmm import CustomNonbondedForce
from simtk import unit as u
from sys import stdout, exit
from math import sqrt


def warn(*argv):
    # write to stderr
    print(*argv, file=sys.stderr, flush=True)


# from /gstore/home/albertgo/.conda/envs/openMM2/lib/python3.6/site-packages/simtk/openmm/app/forcefield.py


system = mm.System()
pdb = app.PDBFile('OCCO.pdb')
modeller = app.Modeller(pdb.topology, pdb.positions)
topo = pdb.getTopology()

with open('PY-atomTypes.txt',"wt") as f:
    for atom in topo.atoms():
        ele = atom.element
        print(ele.atomic_number, file=f)
        system.addParticle(ele.mass)

f = PYForce('graph.pb')
system.addForce(f)


temperature = 298.15 * u.kelvin

integrator = mm.LangevinIntegrator(
    temperature, 1 / u.picosecond,  0.0005 * u.picoseconds)
simulation = app.Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation = Minimize(simulation,1000)
