OpenMM Python Force Plugin
============================

This is a plugin for [OpenMM](http://openmm.org) that allows force computations written in
Python to be integrated. Specifically Neural Net Potentials such as 
[TorchANI](https://github.com/aiqm/torchani) can be used with very little effort.

This code is released under the [MIT license](License.txt).


Installation
============

At present this plugin must be compiled from source. It uses CMake as its build
system.  Before compiling you must install the OpenMM package.

1. Create a directory in which to build the plugin.

2. Run the CMake GUI or ccmake, specifying your new directory as the build directory and the top
level directory of this project as the source directory.

3. Press "Configure".

4. Set OPENMM_DIR to point to the directory where OpenMM is installed.  This is needed to locate
the OpenMM header files and libraries.

5. Set CMAKE_INSTALL_PREFIX to the directory where the plugin should be installed.  Usually,
this will be the same as OPENMM_DIR, so the plugin will be added to your OpenMM installation.

6. If you plan to build the CUDA platform, make sure that CUDA_TOOLKIT_ROOT_DIR is set correctly
and that NN_BUILD_CUDA_LIB is selected.

7. Press "Configure" again if necessary, then press "Generate".
Alternatively, to steps 1-7 you can issue command line statements similar to the following:
```bash
mkdir pybuild
cd pybuild
\rm -rf ../pybuild/*

condaEnv=~/.conda/envs/openMM2
CUDA_DIR=/local/CUDA/9.0.176
export OPENMM_CUDA_COMPILER=$CUDA_DIR/bin/nvcc

make -DOPENMM_DIR=$condaEnv \
     -DCMAKE_INSTALL_PREFIX=$condaEnv \
     -DCUDA_TOOLKIT_ROOT_DIR=$CUDA_DIR \
     -DNN_BUILD_CUDA_LIB=1 \
      <path to checkeout openmm-py>
```

8. Use the build system you selected to build and install the plugin.  For example, if you
selected Unix Makefiles, type `make install` to install the plugin, and `make PythonInstall` to
install the Python wrapper.

Usage
=====

An simple example is given in [demo/min_harmonic.py](demo/min_harmonic.py).
It defines the energy as a simple quadratic function attracting all particles to
the origin. Energy and forces are computed using [pytorch](https://pytorch.org/).
To compute the forces, the autograd pytorch functionality is used.


The Energy and Forces must be computed in a class that extends ```openmm_py.PyCall```.
The ```PyCall.computeEnergyAndForces(self, positions, includeForces, includeEnergy)```
method is a callback function that will be called from openMM for each energy/force
calculation. Here is the main part of the [min_harmonic.py](demo/min_harmonic.py) example:

```python
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

        # convert to pytorch tensor
        coords = torch.from_numpy(pos).to(device=self.device)

        # if forces are requested pytorch needs to know
        if includeForces: coords.requires_grad_(True)

        # compute energy
        pred = self.model.forward(coords)

        if includeForces:
            # use PyTorch autograd to compute:
            #     force = - derivative of energy wrt. coordinates
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
```

The actual energy computation is done in the PyTorch module HarmonicModule.
The ```EnergyComputer``` class provides the callback framework to work
with the ```openmm_py.PYForce``` implementation of an 
[OpenMM::Force](https://simtk.org/api_docs/openmm/api6_0/python/classsimtk_1_1openmm_1_1openmm_1_1Force.html).
It also computes the forces using the PyTorch autograd framework.


To use this EnergyComputer in an OpenMM calculation we must instantiate it and
use the instance to create an instance of ```openmm_py.PYForce```. This
force is added to the ```OpenMM::System```:

```python
ecomputer = EnergyComputer(nGPU)
f = openmm_py.PYForce(ecomputer)
system.addForce(f)
```

You can see all of this working together by running:
```bash
cd demo
min_harmonic.py -in H2O.pdb -out H2O.min.pdb
```
This will minimize the single water molecule in H2O.pdb using our Harmonic force.
The effect of the harmonic force is that the minimized water in H2O.min.pdb will have
all atoms collapsed into the origin.

Acknowledgments
===============

Big parts of the is code were derived from the [openmm-nn](https://github.com/pandegroup/openmm-nn)
plugin by Peter Eastman. I would like to thank Peter for a great introduction to his plugin and
OpenMM development as a whole!

Petr Votava as instrumental in setting up the High Performance Computing enviroment and the OpenMM package at Genentch.

Literature:
===========

Eastman, P.; Swails, J.; Chodera, J. D.; McGibbon, R. T.; Zhao, Y.; Beauchamp, K. A.; Wang, L.-P.; Simmonett, A. C.; Harrigan, M. P.; Stern, C. D.; et al. [OpenMM 7: Rapid Development of High Performance Algorithms for Molecular Dynamics](https://doi.org/10.1371/journal.pcbi.1005659). PLOS Computational Biology 2017, 13 (7), e1005659.


License
=======
```
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
```

