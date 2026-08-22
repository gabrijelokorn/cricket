#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include <string>
#include <filesystem>
#include <vector>
#include <torch/script.h>

#include "Wav.hpp"

// A positive window, plus the raw model score it was classified with (for debugging).
struct ScoredClip
{
    double start;
    double end;
    float score;
};

// --model <name>: model file under assets/models, without the .pt extension. Defaults to "cnn".
std::string parseModelArg(int argc, char **argv);

// Loads assets/models/<modelName>.pt, exits the program on failure.
torch::jit::script::Module loadModel(const std::string &modelName);

// Creates (if needed) and returns outputBase/<inputFolder's name>, mirroring the input folder's name.
std::filesystem::path ensureOutputFolder(const std::filesystem::path &inputFolder, const std::string &outputBase);

// Runs the model on the event_size-wide window starting at the given frame; returns the sigmoid score.
float scoreWindow(Wav &w, torch::jit::script::Module &model, int frame);

// Slides across the whole recording. On a hit, skips ahead by (eventSize - eventStep)
// instead of the full eventSize, so overlapping windows still catch the same event
// continuing; on a miss, steps by eventStep for dense coverage.
// threshold is resolved once by the caller via thresholdFor(modelName).
std::vector<ScoredClip> scanWav(Wav &w, torch::jit::script::Module &model, double threshold);

// Groups positive windows into courtships using courtship_max_gap / courtship_min_events.
std::vector<Courtship> groupCourtships(const std::vector<ScoredClip> &positives);

void writeClipsCsv(const std::vector<ScoredClip> &clips, const std::filesystem::path &path);
void writeCourtshipsCsv(const std::vector<Courtship> &courtships, const std::filesystem::path &path);

std::vector<TimeInterval> toTimeIntervals(const std::vector<Courtship> &courtships);
std::vector<TimeInterval> toTimeIntervals(const std::vector<ScoredClip> &clips);

// Draws each span as a highlighted box over the full spectrogram and writes it out.
// Used for both the raw-clips PNG and the grouped-courtships PNG.
void saveMarkedPng(Wav &w, const std::vector<TimeInterval> &spans, const std::filesystem::path &path);

#endif // PIPELINE_HPP
