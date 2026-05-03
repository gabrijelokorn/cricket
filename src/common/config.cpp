#include "config.hpp"

Config gConfig = {};

bool loadConfig(const char* path) {
    try {
        auto j = readJson(path);
        gConfig.specMinFreq = j.at("spec_min_freq").get<int>();
        gConfig.specMaxFreq = j.at("spec_max_freq").get<int>();
        gConfig.windowSize = j.at("window_size").get<int>();
        gConfig.overlapSize = j.at("overlap_size").get<int>();
        gConfig.eventSize = j.at("event_size").get<int>();
        gConfig.hopSize = gConfig.windowSize - gConfig.overlapSize;
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}