#pragma once
#include <string>
#include <iostream>

#include "json.hpp"

enum class ClipFormat { PNG, NPY };

struct Config {
    int clipMinFreq;
    int clipMaxFreq;
    int windowSize;
    int hopSize;
    int overlapSize;
    int eventSize;
    int eventStep;
    int courtshipMinEvents;
    double courtshipMaxGap;
    std::string recordsPath;
    std::string courtshipClipsPath;
    std::string noiseClipsPath;
    std::string outputPath;
    ClipFormat clipFormat;
};

// Declare — exists somewhere, usable everywhere
extern Config gConfig;

// Load function declaration
bool loadConfig(const char* path);