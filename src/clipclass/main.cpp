#include <iostream>
#include <filesystem>
#include <torch/script.h>
#include <limits>

#include "Wav.hpp"
#include "config.hpp"
#include "gather.hpp"
#include "logger.hpp"
#include "pipeline.hpp"

int main(int argc, char **argv)
{
    std::string modelName = parseModelArg(argc, argv);
    bool saveClipsPng = hasFlag(argc, argv, "--clips");
    bool saveCourtshipsPng = hasFlag(argc, argv, "--courtships");

    std::cout << "Hello, from ClipClass!" << std::endl;
    if (!loadConfig("../assets/config.json"))
    {
        std::cerr << "Failed to load config!" << std::endl;
        return 0;
    }

    torch::jit::script::Module model = loadModel(modelName);
    Logger::Info("Loaded model %s.pt", modelName.c_str());

    std::vector<std::string> folders = openFolderDialog();
    if (folders.empty())
    {
        Logger::Warn("No folders selected.");
        return 0;
    }

    for (const std::string &folderStr : folders)
    {
        std::filesystem::path inputFolder(folderStr);
        std::filesystem::path outputFolder = ensureOutputFolder(inputFolder, gConfig.outputPath);

        // One Wav at a time — loading every recording's spectrogram into memory
        // simultaneously would OOM once folders start holding months of audio.
        for (const std::string &wavPath : findWavFiles(folderStr))
        {
            Wav w(wavPath, rawWav);
            if (!w.getSpec())
            {
                Logger::Warn("Failed to convert %s to spectrogram — skipping", w.getRecName().c_str());
                continue;
            }

            std::vector<TimeInterval> positives = scanWav(w, model);
            std::vector<Courtship> courtships = groupCourtships(positives);

            std::filesystem::path base = outputFolder / w.getRecName();
            writeClipsCsv(positives, base.string() + ".clips.csv");
            writeCourtshipsCsv(courtships, base.string() + ".courtships.csv");

            if (saveClipsPng)
                saveMarkedPng(w, positives, base.string() + ".clips.png");
            if (saveCourtshipsPng)
                saveMarkedPng(w, toTimeIntervals(courtships), base.string() + ".courtships.png");

            Logger::Info("%s: %d positive window(s), %d courtship(s)",
                         w.getRecName().c_str(), (int)positives.size(), (int)courtships.size());
        }
    }

    return 1;
}