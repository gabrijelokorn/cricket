#pragma once
#include <string>
#include <map>
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
    // Score cutoff per model — each model's scores are distributed differently,
    // so one shared value would mean different things for different models.
    std::map<std::string, double> thresholds;
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

// Threshold for the named model. Exits with an error if it has no entry —
// silently falling back to a default would scan with a meaningless cutoff.
double thresholdFor(const std::string &modelName);