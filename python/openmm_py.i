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
%module(directors="1",threads="1") openmm_py
%{
    #include "PYForce.h"
    #include "OpenMM.h"
    #include "OpenMMAmoeba.h"
    #include "OpenMMDrude.h"
    #include "openmm/RPMDIntegrator.h"
    #include "openmm/RPMDMonteCarloBarostat.h"

    #include "openmm_py.h"
    #include <vector>
%}

%feature("director") PyCall;
%import(module="simtk.openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_vector.i>


namespace std {
   %template(FloatVector) vector<float>;
};


using namespace std;

namespace PYPlugin {

    struct NNPResult {
        float energy;
        vector<float> force;
    };


    class PyCall {
    public:
       virtual ~PyCall() {};
       virtual NNPResult * computeEnergyAndForces(vector<float> positions, bool includeForces, bool includeEnergy) = 0;
    };


    class PYForce : public OpenMM::Force {
    public:
        PYForce(PyCall * callPy);
        void setUsesPeriodicBoundaryConditions(bool periodic);
        bool usesPeriodicBoundaryConditions() const;
    };
}
