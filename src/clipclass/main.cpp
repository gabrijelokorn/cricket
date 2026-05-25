#include <iostream>
#include <torch/script.h>
#include <limits>

#include "Wav.hpp"
#include "config.hpp"
#include "gather.hpp"
#include "logger.hpp"

int main()
{
    std::cout << "Hello, from ClipClass!" << std::endl;
    if (!loadConfig("../assets/config.json"))
    {
        std::cerr << "Failed to load config!" << std::endl;
        return 0;
    }

    // Load the TorchScript model
    torch::jit::script::Module model = torch::jit::load("../assets/models/cricket.pt");
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
            cv::Mat clip = w.getClipAtFrame(i);
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
                detectedEvents.push_back({w.frameToTime(i),
                                          w.frameToTime(i + gConfig.eventSize)});
                i += gConfig.eventSize - gConfig.eventStep;
            }
        }

        // Sort detected events by start time
        std::sort(detectedEvents.begin(), detectedEvents.end(), [](const TimeInterval &a, const TimeInterval &b)
                  { return a.start < b.start; });

        // Put the detected courtships onto new spectrogram and export the image
        std::vector<Courtship> courtships;
        Courtship courtship;
        courtship.addEvent(detectedEvents.front());

        for (auto it = detectedEvents.begin() + 1; it != detectedEvents.end(); ++it)
        {
            TimeInterval ti = *it;
            // Add this event to the ongoing courtship
            if (((ti.start - courtship.events.back().start)) * 1000 <= gConfig.courtshipMaxGap)
            {
                courtship.addEvent(ti);
                // Save if last event
                if (std::next(it) == detectedEvents.end())
                    courtships.push_back(courtship);
            }
            else
            {
                // Save the courtship if enough events occured
                if (courtship.events.size() >= gConfig.courtshipMinEvents)
                    courtships.push_back(courtship);
                // Create new courtship object if the gap is too big
                courtship = Courtship();
                courtship.addEvent(ti);
            }
        }
        Logger::Info("Detected %d courtship(s) in recording %s", (int)courtships.size(), w.getRecName().c_str());

        // Create spectrogram with detected courtships highlighted and export it
        for (const auto &c : courtships)
        {
            // Mark individual events in a lighter green
            for (const auto &ev : c.events)
            {
                int evStart = std::max(0, w.timeToFrame(ev.start));
                int evEnd = std::min(display.cols, w.timeToFrame(ev.end));
                if (evStart >= evEnd)
                    continue;

                cv::Mat evRoi = display(cv::Range::all(), cv::Range(evStart, evEnd));
                cv::Mat evOverlay = evRoi.clone();
                cv::rectangle(evOverlay, cv::Point(0, 0), cv::Point(evOverlay.cols, evOverlay.rows),
                              cv::Scalar(0, 0, 200), cv::FILLED);
                cv::Mat evBlended;
                cv::addWeighted(evOverlay, 0.2, evRoi, 0.8, 0, evBlended);
                evBlended.copyTo(evRoi);
            }

            int startFrame = std::max(0, w.timeToFrame(c.events.front().start));
            int endFrame = std::min(display.cols, w.timeToFrame(c.events.back().end));

            if (startFrame >= endFrame)
            {
                Logger::Warn("Skipping — invalid range %d", 1);
                continue;
            }

            cv::Mat roi = display(cv::Range::all(), cv::Range(startFrame, endFrame));
            cv::Mat overlay = roi.clone();
            cv::rectangle(overlay, cv::Point(0, 0), cv::Point(overlay.cols, overlay.rows),
                          cv::Scalar(0, 255, 0), cv::FILLED);
            cv::Mat blended;
            cv::addWeighted(overlay, 0.2, roi, 0.8, 0, blended);
            blended.copyTo(roi);
        }

        cv::imwrite("../detected_" + w.getRecName() + ".png", display);
    }

    return 1;
}