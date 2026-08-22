#ifndef ARGS_HPP
#define ARGS_HPP

#include <string>

// True if the flag (e.g. "--clips") appears anywhere in argv.
bool hasFlag(int argc, char **argv, const std::string &flag);

// The value following a flag (e.g. "--format npy" -> "npy"), or fallback if the
// flag is absent or has nothing after it.
std::string flagValue(int argc, char **argv, const std::string &flag, const std::string &fallback = "");

#endif // ARGS_HPP
