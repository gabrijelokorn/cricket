#include <iostream>

#include "Wav.hpp"
#include "config.hpp"
#include "gather.hpp"

int main()
{
    std::cout << "Hello, from ClipGen!" << std::endl;
    if (!loadConfig("../assets/config.json"))
    {
        std::cerr << "Failed to load config!" << std::endl;
        return 0;
    }

    std::vector<Wav> spectrograms = getSpectrograms();
    for (Wav w : spectrograms)
    {
        w.clipCourtship();
        w.clipNoise();
    }

    return 1;
}