#include <cstdlib>

#include "config.hpp"

Config gConfig = {};

bool loadConfig(const char* path) {
    try {
        auto j = readJson(path);
        gConfig.clipMinFreq = j.at("clip_min_freq").get<int>();
        gConfig.clipMaxFreq = j.at("clip_max_freq").get<int>();
        gConfig.windowSize = j.at("window_size").get<int>();
        gConfig.overlapSize = j.at("overlap_size").get<int>();
        gConfig.eventSize = j.at("event_size").get<int>();
        gConfig.eventStep = j.at("event_step").get<int>();
        gConfig.hopSize = gConfig.windowSize - gConfig.overlapSize;
        gConfig.recordsPath = j.at("records_path").get<std::string>();
        gConfig.courtshipClipsPath = j.at("courtship_clips_path").get<std::string>();
        gConfig.noiseClipsPath = j.at("noise_clips_path").get<std::string>();
        gConfig.outputPath = j.at("output_path").get<std::string>();
        gConfig.courtshipMinEvents = j.at("courtship_min_events").get<int>();
        gConfig.courtshipMaxGap = j.at("courtship_max_gap").get<double>();
        gConfig.thresholds = j.at("thresholds").get<std::map<std::string, double>>();
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}

double thresholdFor(const std::string &modelName) {
    auto it = gConfig.thresholds.find(modelName);
    if (it == gConfig.thresholds.end()) {
        std::cerr << "No threshold configured for model '" << modelName
                  << "' — add it to \"thresholds\" in assets/config.json" << std::endl;
        std::exit(1);
    }
    return it->second;
}