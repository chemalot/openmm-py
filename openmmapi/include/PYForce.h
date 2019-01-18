#ifndef OPENMM_PYFORCE_H_
#define OPENMM_PYFORCE_H_

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


#include "openmm/Context.h"
#include "openmm/Force.h"
#include <string>
#include <vector>
#include "internal/windowsExportPY.h"
#include "openmm_py.h"

using namespace std;


namespace PYPlugin {

/**
 * This class implements forces that are defined by user-supplied pyhthon function call
 * It uses the TensorFlow library to perform the computations. */

class OPENMM_EXPORT_NN PYForce : public OpenMM::Force {
public:
    /**
     * Create a PYForce.  The network is accessd via a python function callback.
     *
     * @param pyCall the callback function
     */
    PYForce(PyCall *pyCall);

    /**
     * return callback to pytorch code.
     */
    PyCall * getPyCall() const;

    /**
     * Set whether this force makes use of periodic boundary conditions.  If this is set
     * to true, the TensorFlow graph must include a 3x3 tensor called "boxvectors", which
     * is set to the current periodic box vectors.
     */
    void setUsesPeriodicBoundaryConditions(bool periodic);

    /**
     * Get whether this force makes use of periodic boundary conditions.
     */
    bool usesPeriodicBoundaryConditions() const;

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    PyCall *pyCall;
    bool usePeriodic;
};

} // namespace NNPlugin

#endif /*OPENMM_PYFORCE_H_*/
