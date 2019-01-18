#ifndef CALLPY_H
#define CALLPY_H

#include <vector>

using namespace std;

class PyCall {
public:
   virtual ~PyCall() {};
   virtual float computeEnergyAndForces(vector<float> positions, bool includeForces, bool includeEnergy) = 0;
};

