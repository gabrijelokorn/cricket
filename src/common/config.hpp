#pragma once
#include <string>
#include <iostream>

#include "json.hpp"

struct Config {
    int specMinFreq;
    int specMaxFreq;
    int windowSize;
    int hopSize;
    int overlapSize;
    int eventSize;
};

// Declare — exists somewhere, usable everywhere
extern Config gConfig;

// Load function declaration
bool loadConfig(const char* path);