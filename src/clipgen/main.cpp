#include <iostream>
#include <filesystem>

#include "Wav.hpp"
#include "args.hpp"
#include "config.hpp"
#include "gather.hpp"

ClipFormat parseFormatArg(int argc, char **argv)
{
    std::string format = flagValue(argc, argv, "--format", "npy");

    if (format == "png") return ClipFormat::PNG;
    if (format == "npy") return ClipFormat::NPY;

    std::cerr << "Invalid --format value: " << format << " (expected 'png' or 'npy')" << std::endl;
    std::exit(1);
}

void printUsage()
{
    std::cerr << "Nothing to do — pass at least one of:\n"
              << "  --clips           export labeled courtship and noise clips\n"
              << "  --spectrograms           export a full spectrogram PNG per recording\n"
              << "\nOptional:\n"
              << "  --format png|npy  clip file format (default: npy, applies to --clips)\n";
}

int main(int argc, char **argv)
{
    bool exportClips = hasFlag(argc, argv, "--clips");
    bool exportSpecs = hasFlag(argc, argv, "--spectrograms");

    // Without this the program would run the whole spectrogram computation and
    // then write nothing, which looks like a failure rather than a missing flag.
    if (!exportClips && !exportSpecs)
    {
        printUsage();
        return 1;
    }

    ClipFormat format = parseFormatArg(argc, argv);

    std::cout << "Hello, from ClipGen!" << std::endl;
    if (!loadConfig("../assets/config.json"))
    {
        std::cerr << "Failed to load config!" << std::endl;
        return 0;
    }

    gConfig.clipFormat = format;

    if (exportClips)
    {
        std::filesystem::create_directories(gConfig.courtshipClipsPath);
        std::filesystem::create_directories(gConfig.noiseClipsPath);
    }

    for (const std::string &f : openFileDialog(gConfig.recordsPath))
    {
        Wav w(f);
        if (!w.getSpec())
        {
            Logger::Warn("Failed to convert %s to spectrogram — skipping", w.getRecName().c_str());
            continue;
        }
        Logger::Info("Successfully converted %s to spectrogram", w.getRecName().c_str());

        if (exportClips)
        {
            w.exportLabeledCourtship();
            w.exportLabeledNoise();
        }

        if (exportSpecs)
            w.exportSpectrogram();
    }

    return 1;
}
