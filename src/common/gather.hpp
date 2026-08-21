#ifndef GATHER_HPP
#define GATHER_HPP

#include <string>
#include <vector>

std::vector<std::string> openFileDialog(const std::string &defaultPath);
std::vector<std::string> openFolderDialog();
std::vector<std::string> findWavFiles(const std::string &folder);

#endif // GATHER_HPP    