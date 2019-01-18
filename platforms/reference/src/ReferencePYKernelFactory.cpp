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


#include <exception>

#include "ReferencePYKernelFactory.h"
#include "ReferencePYKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include <vector>
#include <iostream>

using namespace PYPlugin;
using namespace OpenMM;
using namespace std;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    int argc = 0;
    vector<char**> argv = {NULL};
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferencePYKernelFactory* factory = new ReferencePYKernelFactory();
            platform.registerKernelFactory(CalcPYForceKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerPYReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferencePYKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    std::cerr << "\nAGAG: ReferencePYKernelFactory::createKernelImpl\n";

    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcPYForceKernel::Name())
        return new ReferenceCalcPYForceKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
