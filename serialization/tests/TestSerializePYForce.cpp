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
#include "openmm/Platform.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/serialization/XmlSerializer.h"
#include <iostream>
#include <sstream>

using namespace PYPlugin;
using namespace OpenMM;
using namespace std;

extern "C" void registerPYSerializationProxies();

void testSerialization() {
    // Create a Force.

    // TODO pass in topology
    vector<int> dummy = { 8,1,1 };
    PYForce force("pyArgs", dummy);

    // Serialize and then deserialize it.

    stringstream buffer;
    XmlSerializer::serialize<PYForce>(&force, "Force", buffer);
    PYForce* copy = XmlSerializer::deserialize<PYForce>(buffer);

    // Compare the two forces to see if they are identical.

    PYForce& force2 = *copy;
    ASSERT_EQUAL(force.getPyArgs(), force2.getPyArgs());

    vector<int> atT1 = force.getAtomTypes();
    vector<int> atT2 = force2.getAtomTypes();
    for( int i=0; i<atT1.size(); i++ )
       ASSERT_EQUAL(atT1[i], atT2[i]);
}

int main() {
    try {
        registerPYSerializationProxies();
        testSerialization();
    }
    catch(const exception& e) {
        cerr << "exception: " << e.what() << endl;
        return 1;
    }
    cerr << "Done" << endl;
    return 0;
}
