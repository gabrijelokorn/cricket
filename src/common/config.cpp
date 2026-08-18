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
        gConfig.clipsBasePath = j.at("clips_base_path").get<std::string>();
        gConfig.courtshipMinEvents = j.at("courtship_min_events").get<int>();
        gConfig.courtshipMaxGap = j.at("courtship_max_gap").get<double>();
        return true;
    } catch (const std::exception &e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return false;
    }
}