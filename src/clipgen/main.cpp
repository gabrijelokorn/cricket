#include <iostream>
#include <filesystem>

#include "Wav.hpp"
#include "config.hpp"
#include "gather.hpp"

ClipFormat parseFormatArg(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--format" && i + 1 < argc)
        {
            std::string format = argv[i + 1];
            if (format == "png") return ClipFormat::PNG;
            if (format == "npy") return ClipFormat::NPY;
            std::cerr << "Invalid --format value: " << format << " (expected 'png' or 'npy')" << std::endl;
            std::exit(1);
        }
    }
    return ClipFormat::NPY;
}

std::string parseSplitArg(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--split" && i + 1 < argc)
        {
            std::string split = argv[i + 1];
            if (split == "train" || split == "valid")
                return split;
            std::cerr << "Invalid --split value: " << split << " (expected 'train' or 'valid')" << std::endl;
            std::exit(1);
        }
    }
    return "train";
}

int main(int argc, char **argv)
{
    ClipFormat format = parseFormatArg(argc, argv);
    std::string split = parseSplitArg(argc, argv);

    std::cout << "Hello, from ClipGen!" << std::endl;
    if (!loadConfig("../assets/config.json"))
    {
        std::cerr << "Failed to load config!" << std::endl;
        return 0;
    }

    gConfig.clipFormat = format;
    gConfig.courtshipClipsPath = gConfig.clipsBasePath + "/" + split + "/courtship";
    gConfig.noiseClipsPath     = gConfig.clipsBasePath + "/" + split + "/noise";
    std::filesystem::create_directories(gConfig.courtshipClipsPath);
    std::filesystem::create_directories(gConfig.noiseClipsPath);

    std::vector<Wav> spectrograms = getSpectrograms();
    for (Wav w : spectrograms)
    {
        w.exportLabeledCourtship();
        w.exportLabeledNoise();
        w.exportSpectrogram();
    }

    return 1;
}