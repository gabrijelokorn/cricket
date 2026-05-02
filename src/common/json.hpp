#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <stdexcept>

using json = nlohmann::json;

inline json readJson(const std::string &path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Could not open file: " + path);
    return json::parse(f);
}