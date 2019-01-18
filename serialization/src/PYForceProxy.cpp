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


#include "PYForceProxy.h"
#include "PYForce.h"
#include "openmm/serialization/SerializationNode.h"
#include <string>
#include <vector>
#include <ostream>
#include <istream>
#include <fstream>
#include <iostream>

using namespace PYPlugin;
using namespace OpenMM;
using namespace std;

PYForceProxy::PYForceProxy() : SerializationProxy("PYForce") {
}

void PYForceProxy::serialize(const void* object, SerializationNode& node) const {
/* not serializable 
    node.setIntProperty("version", 1);
    const PYForce& force = *reinterpret_cast<const PYForce*>(object);
    node.setStringProperty("pySerFile", PY_SERIALIZATION_FILE);
    
    string pyArgs = force.getPyArgs();
    vector<int> atomTypes = force.getAtomTypes();

    ofstream ofs{PY_SERIALIZATION_FILE};
    if( ! ofs ) runtime_error("could not open " + PY_SERIALIZATION_FILE);
    ofs << pyArgs << '\n';
    for(int at:atomTypes)
        ofs << at << '\n';
*/
}

void* PYForceProxy::deserialize(const SerializationNode& node) const {
    throw OpenMMException("PYForce does not support serialization");
/*
    if (node.getIntProperty("version") != 1)
        throw OpenMMException("Unsupported version number");

    string pyArgs;
    vector<int> atomTypes;

    string serFile = node.getStringProperty("pySerFile");
    ifstream ifs{serFile};

    getline(ifs, pyArgs);

    for( int at; ifs >> at; )
        atomTypes.push_back(at);
    PYForce* force = new PYForce(pyArgs, atomTypes);
    return force;
*/
}
