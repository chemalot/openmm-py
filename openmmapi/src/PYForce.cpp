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


#include "PYForce.h"
// included in PYForce.h: #include "openmm_py.h"
#include "internal/PYForceImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include <iostream>
#include <vector>


using namespace PYPlugin;
using namespace OpenMM;
using namespace std;

PYForce::PYForce(PyCall *pyCall) : pyCall(pyCall), usePeriodic(false) {
}

PyCall * PYForce::getPyCall() const {
    return pyCall;
}

ForceImpl* PYForce::createImpl() const {
	OpenMM::ForceImpl* imp =  new PYForceImpl(*this);
   return imp;
}

void PYForce::setUsesPeriodicBoundaryConditions(bool periodic) {
    usePeriodic = periodic;
}

bool PYForce::usesPeriodicBoundaryConditions() const {
    return usePeriodic;
}
