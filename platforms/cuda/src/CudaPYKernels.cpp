/* -------------------------------------------------------------------------- *
 * The MIT License
 *
 * SPDX short identifier: MIT
 *
 * Copyright 2019 Genentech Inc. South San Francisco
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 * -------------------------------------------------------------------------- */
/* -------------------------------------------------------------------------- *
 * Portions of this software were derived from code originally developed
 * by Peter Eastman and copyrighted by Stanford University and the Authors
 * -------------------------------------------------------------------------- */


#include "CudaPYKernels.h"
#include "CudaPYKernelSources.h"
#include "PYForce.h"
#include "openmm_py.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <iostream>

#define PRINT_RESULTS 1
#define OPENMM_FLOAT 4



using namespace PYPlugin;
using namespace OpenMM;
using namespace std;

CudaCalcPYForceKernel::~CudaCalcPYForceKernel() {
}

void CudaCalcPYForceKernel::initialize(const System& system, const PYForce& force) {

    // cu is OpenMM::CudaContext&
    cu.setAsCurrent();
    usePeriodic = force.usesPeriodicBoundaryConditions();
    pyCall     = force.getPyCall();;

    int numParticles = system.getNumParticles();

    // pre-size position vector to bve reused each time execute is called
    pyPositions.resize(numParticles*3);
     
    //if (usePeriodic) {
    //    int64_t boxVectorsDims[] = {3, 3};
    //    boxVectorsTensor = TF_AllocateTensor(boxType, boxVectorsDims, 2, 9*TF_DataTypeSize(boxType));
    //}

    // Inititalize CUDA objects.
    // networkForces is OpenMM::CudaArray
    networkForces.initialize(cu, 3*numParticles, OPENMM_FLOAT, "networkForces");
    map<string, string> defines;
    defines["FORCES_TYPE"] = "float";
    CUmodule module = cu.createModule(CudaPYKernelSources::pyForce, defines);
    addForcesKernel = cu.getKernel(module, "addForces");
}

double CudaCalcPYForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3> pos;
    context.getPositions(pos);
    int numParticles = cu.getNumAtoms();
    for (int i = 0; i < numParticles; i++) {
        pyPositions[3*i]   = (float) pos[i][0];
        pyPositions[3*i+1] = (float) pos[i][1];
        pyPositions[3*i+2] = (float) pos[i][2];
    }

    if( PRINT_RESULTS ) {
        cerr << "Positions: ";
        for(float p : pyPositions) cerr << p << " ";
        cerr << pyPositions.size() << endl;
    }


    /*
    if (usePeriodic) {
        Vec3 box[3];
        cu.getPeriodicBoxVectors(box[0], box[1], box[2]);
        if (boxType == TF_FLOAT) {
            float* boxVectors = reinterpret_cast<float*>(TF_TensorData(boxVectorsTensor));
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    boxVectors[3*i+j] = box[i][j];
        }
        }
    */


    // Call python and compute forces and energy
    NNPResult *res = pyCall->computeEnergyAndForces(pyPositions, includeForces, includeEnergy);
    float energy = 0.0;
    if (includeEnergy) energy = res->energy;

    if (includeForces) {

cerr<<"1"<<endl;
        // Use cuda Kernel to upload forces to GPU
        networkForces.upload(res->force.data());
cerr<<"2"<<endl;
        int paddedNumAtoms = cu.getPaddedNumAtoms();
cerr<<"3"<<endl;
        void* args[] = {&networkForces.getDevicePointer(), &cu.getForce().getDevicePointer(),
                        &cu.getAtomIndexArray().getDevicePointer(), &numParticles, &paddedNumAtoms};
cerr<<"4"<<endl;
        cu.executeKernel(addForcesKernel, args, numParticles);
cerr<<"5"<<endl;
    }

    if( PRINT_RESULTS ) {
        cerr << "Energy[KJ/Mol]: " << energy << " forces[KJ/Mol/nM]: ";
        for (float f : res->force) cerr << f << " ";
    }

    return (double)energy;
}
