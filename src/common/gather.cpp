#include <nfd.hpp>
#include <iostream>

#include "Wav.hpp"
#include "gather.hpp"

std::vector<std::string> openFileDialog()
{
    NFD::Guard guard;
    NFD::UniquePathSet paths;

    // Get the absolute path of ../assets/record relative to the executable
    std::filesystem::path exePath = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path defaultPath = std::filesystem::weakly_canonical(exePath.parent_path() / "../assets/records");

    nfdfilteritem_t filters[] = {{"JSON", "json"}, {"All Files", "*"}};
    nfdresult_t result = NFD::OpenDialogMultiple(paths, filters, 2, defaultPath.c_str());

    std::vector<std::string> files;

    if (result == NFD_OKAY)
    {
        nfdpathsetsize_t count;
        NFD::PathSet::Count(paths, count);

        for (nfdpathsetsize_t i = 0; i < count; i++)
        {
            NFD::UniquePathSetPath path;
            NFD::PathSet::GetPath(paths, i, path);
            files.push_back(path.get());
        }
    }
    else if (result == NFD_CANCEL)
    {
        std::cout << "User cancelled." << std::endl;
    }
    else
    {
        std::cerr << "Error: " << NFD::GetError() << std::endl;
    }

    return files;
}

std::vector<Wav> getSpectrograms()
{
    auto files = openFileDialog();
    std::vector<Wav> wavs;

    for (const auto &f : files)
    {
        Wav wav = Wav(f);
        wav.getSpec();
        wavs.emplace_back(wav);
    }
    return wavs;
}