#include "args.hpp"

bool hasFlag(int argc, char **argv, const std::string &flag)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag)
            return true;
    return false;
}

std::string flagValue(int argc, char **argv, const std::string &flag, const std::string &fallback)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag && i + 1 < argc)
            return argv[i + 1];
    return fallback;
}
