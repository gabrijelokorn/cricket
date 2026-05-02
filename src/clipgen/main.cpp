#include <iostream>

#include "Wav.hpp"
#include "config.hpp"

int main()
{
    std::cout << "Hello, from ClipGen!" << std::endl;

    if (!loadConfig("../assets/config.json")) {
        std::cerr << "Failed to load config!" << std::endl;
    }

    std::cout << "Config loaded: " << std::endl;
    std::cout << "Min Frequency: " << gConfig.minFreq << std::endl;
    
    return 1;
}