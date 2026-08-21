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

int main(int argc, char **argv)
{
    ClipFormat format = parseFormatArg(argc, argv);

    std::cout << "Hello, from ClipGen!" << std::endl;
    if (!loadConfig("../assets/config.json"))
    {
        std::cerr << "Failed to load config!" << std::endl;
        return 0;
    }

    gConfig.clipFormat = format;
    std::filesystem::create_directories(gConfig.courtshipClipsPath);
    std::filesystem::create_directories(gConfig.noiseClipsPath);

    for (const std::string &f : openFileDialog(gConfig.recordsPath))
    {
        Wav w(f);
        if (!w.getSpec())
        {
            Logger::Warn("Failed to convert %s to spectrogram — skipping", w.getRecName().c_str());
            continue;
        }
        Logger::Info("Successfully converted %s to spectrogram", w.getRecName().c_str());

        w.exportLabeledCourtship();
        w.exportLabeledNoise();
        // w.exportSpectrogram();
    }

    return 1;
}