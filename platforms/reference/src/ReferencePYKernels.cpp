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


#include "ReferencePYKernels.h"
#include "PYForce.h"
#include "openmm_py.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include <iostream>

#define PRINT_RESULTS 1
#define OPENMM_FLOAT 4



using namespace PYPlugin;
using namespace OpenMM;
using namespace std;

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->positions);
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*) data->forces);
}

static Vec3* extractBoxVectors(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return (Vec3*) data->periodicBoxVectors;
}

ReferenceCalcPYForceKernel::~ReferenceCalcPYForceKernel() {
}

void ReferenceCalcPYForceKernel::initialize(const System& system, const PYForce& force) {
    std::cerr << "\nAGAG: ReferenceCalcPYForceKernel::initialize\n";

    usePeriodic = force.usesPeriodicBoundaryConditions();
    pyCall     = force.getPyCall();;

    int numParticles = system.getNumParticles();

    // pre-size position vector to bve reused each time execute is called
    pyPositions.resize(numParticles*3);
}

/**
 * This is where the actual energy and forces are computed by calling
 * the pyton callback.
 *
 */
double ReferenceCalcPYForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    std::cerr << "\nAGAG: ReferenceCalcPYForceKernel::execute\n";

    vector<Vec3>& pos = extractPositions(context);
    int numParticles = pos.size();
    for (int i = 0; i < numParticles; i++) {
        // python expects position in vactor<float>
        pyPositions[3*i]   = (float) pos[i][0];
        pyPositions[3*i+1] = (float) pos[i][1];
        pyPositions[3*i+2] = (float) pos[i][2];
    }
    if( PRINT_RESULTS ) {
        cerr << "Positions: ";
        for(float p : pyPositions) cerr << p << " ";
        cerr << "n: " << pyPositions.size() << endl;
    }

    /*
    if (usePeriodic) {
        Vec3* box = extractBoxVectors(context);
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
        vector<Vec3>& force = extractForces(context);
        for (int i = 0; i < numParticles; i++) {
            force[i][0] += (double)res->force[3*i];
            force[i][1] += (double)res->force[3*i+1];
            force[i][2] += (double)res->force[3*i+2];
        }
    }

    if( PRINT_RESULTS ) {
        cerr << "Energy[KJ/Mol]: " << energy << " forces[KJ/Mol/nM]: ";
        for (float f : res->force) cerr << f << " ";
    }
    cerr << endl;

    return (double)energy;
}
