#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <string>
#include <filesystem>
#include <vector>
#include <torch/script.h>

#include "Wav.hpp"

// --model <name>: model file under assets/models, without the .pt extension. Defaults to "cnn".
std::string parseModelArg(int argc, char **argv);

// True if the given flag (e.g. "--clips") is present anywhere in argv.
bool hasFlag(int argc, char **argv, const std::string &flag);

// Loads assets/models/<modelName>.pt, exits the program on failure.
torch::jit::script::Module loadModel(const std::string &modelName);

// Creates (if needed) and returns outputBase/<inputFolder's name>, mirroring the input folder's name.
std::filesystem::path ensureOutputFolder(const std::filesystem::path &inputFolder, const std::string &outputBase);

// Scores the event_size-wide window starting at the given frame.
bool isCourtshipWindow(Wav &w, torch::jit::script::Module &model, int frame);

// Slides across the whole recording. On a hit, skips ahead by (eventSize - eventStep)
// instead of the full eventSize, so overlapping windows still catch the same event
// continuing; on a miss, steps by eventStep for dense coverage.
std::vector<TimeInterval> scanWav(Wav &w, torch::jit::script::Module &model);

// Groups positive windows into courtships using courtship_max_gap / courtship_min_events.
std::vector<Courtship> groupCourtships(const std::vector<TimeInterval> &positives);

void writeClipsCsv(const std::vector<TimeInterval> &clips, const std::filesystem::path &path);
void writeCourtshipsCsv(const std::vector<Courtship> &courtships, const std::filesystem::path &path);

std::vector<TimeInterval> toTimeIntervals(const std::vector<Courtship> &courtships);

// Draws each span as a highlighted box over the full spectrogram and writes it out.
// Used for both the raw-clips PNG and the grouped-courtships PNG.
void saveMarkedPng(Wav &w, const std::vector<TimeInterval> &spans, const std::filesystem::path &path);

#endif // PIPELINE_HPP
