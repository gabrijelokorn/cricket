#include <iostream>
#include <torch/script.h>

#include "Wav.hpp"
#include "config.hpp"
#include "gather.hpp"

int main()
{
    std::cout << "Hello, from ClipClass!" << std::endl;
    if (!loadConfig("../assets/config.json"))
    {
        std::cerr << "Failed to load config!" << std::endl;
        return 0;
    }

    torch::jit::script::Module model = torch::jit::load("../cricket.pt");
    model.eval();

    // torch::Tensor input = torch::from_blob(pixels, {1, 1, 300, 16});

    std::vector<Wav> spectrograms = getSpectrograms();
    for (Wav w : spectrograms)
    {
        // for (int i = 0; i < w.getWavNumTimeFrames() - gConfig.eventSize; i += gConfig.eventSize)
        // {
        //     w.exportClip(w.getClipByFrame(i), "../trash/" + std::to_string(i) + ".png");
        // }
    }

    return 1;
}