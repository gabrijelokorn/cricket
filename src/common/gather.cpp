#include <nfd.hpp>
#include <iostream>
#include <algorithm>
#include <filesystem>

#include "Wav.hpp"
#include "gather.hpp"

std::vector<std::string> openFileDialog(const std::string &defaultPath)
{
    NFD::Guard guard;
    NFD::UniquePathSet paths;

    // Get the absolute path of defaultPath relative to the executable
    std::filesystem::path exePath = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path resolvedPath = std::filesystem::weakly_canonical(exePath.parent_path() / defaultPath);

    nfdfilteritem_t filters[] = {{"JSON", "json"}, {"All Files", "*"}};
    nfdresult_t result = NFD::OpenDialogMultiple(paths, filters, 2, resolvedPath.c_str());

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

std::vector<std::string> openFolderDialog()
{
    NFD::Guard guard;
    NFD::UniquePathSet paths;

    // Get the absolute path of ../assets/records relative to the executable
    std::filesystem::path exePath = std::filesystem::canonical("/proc/self/exe");
    std::filesystem::path defaultPath = std::filesystem::weakly_canonical(exePath.parent_path() / "../assets/records");

    nfdresult_t result = NFD::PickFolderMultiple(paths, defaultPath.c_str());

    std::vector<std::string> folders;

    if (result == NFD_OKAY)
    {
        nfdpathsetsize_t count;
        NFD::PathSet::Count(paths, count);

        for (nfdpathsetsize_t i = 0; i < count; i++)
        {
            NFD::UniquePathSetPath path;
            NFD::PathSet::GetPath(paths, i, path);
            folders.push_back(path.get());
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

    return folders;
}

std::vector<std::string> findWavFiles(const std::string &folder)
{
    std::vector<std::string> wavFiles;

    for (const auto &entry : std::filesystem::directory_iterator(folder))
    {
        if (!entry.is_regular_file())
            continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".wav")
            wavFiles.push_back(entry.path().string());
    }

    return wavFiles;
}