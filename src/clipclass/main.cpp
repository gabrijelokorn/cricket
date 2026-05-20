#include <iostream>
#include <torch/script.h>
#include <limits>

#include "Wav.hpp"
#include "config.hpp"
#include "gather.hpp"
struct Courtship {
    std::vector<TimeInterval> events;
    double start;
    double end;
};

int main()
{
    std::cout << "Hello, from ClipClass!" << std::endl;
    if (!loadConfig("../assets/config.json"))
    {
        std::cerr << "Failed to load config!" << std::endl;
        return 0;
    }

    // Load the TorchScript model
    torch::jit::script::Module model = torch::jit::load("../cricket.pt");
    model.eval();

    // Load wav recordings as spectrograms
    std::vector<Wav> spectrograms = getSpectrograms();
    for (Wav w : spectrograms)
    {
        // We convert the spectrogram to a 3-channel BGR image for visualization.
        // This way the detected events can be highlighted in green.
        cv::Mat display;
        cv::Mat spec8u;
        w.getMSpec().convertTo(spec8u, CV_8U);
        cv::cvtColor(spec8u, display, cv::COLOR_GRAY2BGR);

        std::vector<TimeInterval> detectedEvents;
        for (int i = 0; i < w.getWavNumTimeFrames() - gConfig.eventSize; i += gConfig.eventSize + gConfig.eventStep)
        {
            // Normalize the spectrogram clip to match the values from trained model
            cv::Mat clip = w.getClipByFrame(i);
            cv::Mat normalized;
            clip.convertTo(normalized, CV_32F, 1.0 / 255.0);

            // Convert the normalized spectrogram clip to a Torch tensor
            torch::Tensor input = torch::from_blob(
                                      normalized.data, {1, 1, normalized.rows, normalized.cols})
                                      .clone();
            // Run the model and get the output score
            float score = torch::sigmoid(
                              model.forward(std::vector<torch::jit::IValue>{input}).toTensor())
                              .item<float>();

            if (score > 0.5f)
            {
                cv::Mat roi = display(cv::Range::all(), cv::Range(i, i + gConfig.eventSize));
                cv::Mat overlay = roi.clone();
                cv::rectangle(overlay, cv::Point(0, 0), cv::Point(overlay.cols, overlay.rows),
                              cv::Scalar(0, 255, 0), cv::FILLED);
                cv::addWeighted(overlay, 0.2, roi, 0.8, 0, roi); // 20% green, 80% original

                detectedEvents.push_back({w.specTimeFrameToMs(i),
                                          w.specTimeFrameToMs(i + gConfig.eventSize)});

                i += gConfig.eventSize - gConfig.eventStep;
            }
        }

        // put the detected courtships onto new spectrogram and export the image
        // std::vector<Courtship> courtships;

        // for (TimeInterval ti : detectedEvents)
        // {

        // }

        // cv::imwrite("../detected_" + w.getRecName() + ".png", display);
    }

    return 1;
}