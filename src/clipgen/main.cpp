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
    }

    // Wav wav = Wav("../assets/records/raven.251202.125905.json");
    // wav.getSpec();

    return 1;
}