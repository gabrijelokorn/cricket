#include <iostream>
#include <fstream>
#include <iomanip>

#include "pipeline.hpp"
#include "config.hpp"

std::string parseModelArg(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
            return argv[i + 1];
    }
    return "cnn";
}

bool hasFlag(int argc, char **argv, const std::string &flag)
{
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == flag)
            return true;
    return false;
}

torch::jit::script::Module loadModel(const std::string &modelName)
{
    std::string modelPath = "../assets/models/" + modelName + ".pt";
    try
    {
        torch::jit::script::Module model = torch::jit::load(modelPath);
        model.eval();
        return model;
    }
    catch (const c10::Error &e)
    {
        std::cerr << "Error loading model at " << modelPath << ": " << e.what() << std::endl;
        std::exit(1);
    }
}

std::filesystem::path ensureOutputFolder(const std::filesystem::path &inputFolder, const std::string &outputBase)
{
    std::filesystem::path outFolder = std::filesystem::path(outputBase) / inputFolder.filename();
    std::filesystem::create_directories(outFolder);
    return outFolder;
}

float scoreWindow(Wav &w, torch::jit::script::Module &model, int frame)
{
    cv::Mat clip = w.trimFrequencyRange(w.getClipAtFrame(frame));

    cv::Mat normalized;
    clip.convertTo(normalized, CV_32F, 1.0 / 255.0);
    if (!normalized.isContinuous())
        normalized = normalized.clone();

    torch::Tensor input = torch::from_blob(normalized.data, {1, 1, normalized.rows, normalized.cols}).clone();
    return torch::sigmoid(model.forward(std::vector<torch::jit::IValue>{input}).toTensor()).item<float>();
}

std::vector<ScoredClip> scanWav(Wav &w, torch::jit::script::Module &model, double threshold)
{
    std::vector<ScoredClip> positives;
    int numFrames = w.getWavNumTimeFrames();

    int i = 0;
    while (i + gConfig.eventSize <= numFrames)
    {
        float score = scoreWindow(w, model, i);
        if (score > threshold)
        {
            positives.push_back({w.frameToTime(i), w.frameToTime(i + gConfig.eventSize), score});
            i += gConfig.eventSize - gConfig.eventStep;
        }
        else
        {
            i += gConfig.eventStep;
        }
    }

    return positives;
}

std::vector<Courtship> groupCourtships(const std::vector<ScoredClip> &positives)
{
    std::vector<Courtship> courtships;
    if (positives.empty())
        return courtships;

    double maxGap = gConfig.courtshipMaxGap / 1000.0;

    Courtship current;
    current.addEvent({positives[0].start, positives[0].end});

    for (size_t k = 1; k < positives.size(); ++k)
    {
        const ScoredClip &p = positives[k];
        if (p.start - current.end > maxGap)
        {
            if ((int)current.events.size() >= gConfig.courtshipMinEvents)
                courtships.push_back(current);
            current = Courtship();
            current.addEvent({p.start, p.end});
        }
        else
        {
            current.addEvent({p.start, p.end});
        }
    }
    if ((int)current.events.size() >= gConfig.courtshipMinEvents)
        courtships.push_back(current);

    return courtships;
}

void writeClipsCsv(const std::vector<ScoredClip> &clips, const std::filesystem::path &path)
{
    std::ofstream out(path);
    out << "start_time,end_time,score\n";
    for (const ScoredClip &c : clips)
        out << std::fixed << std::setprecision(3) << c.start << "," << c.end << "," << c.score << "\n";
}

void writeCourtshipsCsv(const std::vector<Courtship> &courtships, const std::filesystem::path &path)
{
    std::ofstream out(path);
    out << "start_time,end_time,num_events\n";
    for (const Courtship &c : courtships)
        out << std::fixed << std::setprecision(3) << c.start << "," << c.end << "," << c.events.size() << "\n";
}

std::vector<TimeInterval> toTimeIntervals(const std::vector<Courtship> &courtships)
{
    std::vector<TimeInterval> spans;
    for (const Courtship &c : courtships)
        spans.push_back({c.start, c.end});
    return spans;
}

std::vector<TimeInterval> toTimeIntervals(const std::vector<ScoredClip> &clips)
{
    std::vector<TimeInterval> spans;
    for (const ScoredClip &c : clips)
        spans.push_back({c.start, c.end});
    return spans;
}

void saveMarkedPng(Wav &w, const std::vector<TimeInterval> &spans, const std::filesystem::path &path)
{
    cv::Mat spec8u;
    w.getMSpec().convertTo(spec8u, CV_8U);
    cv::Mat display;
    cv::cvtColor(spec8u, display, cv::COLOR_GRAY2BGR);

    for (const TimeInterval &s : spans)
    {
        int startFrame = std::max(0, w.timeToFrame(s.start));
        int endFrame = std::min(display.cols, w.timeToFrame(s.end));
        if (startFrame >= endFrame)
            continue;

        cv::Mat roi = display(cv::Range::all(), cv::Range(startFrame, endFrame));
        cv::Mat overlay = roi.clone();
        cv::rectangle(overlay, cv::Point(0, 0), cv::Point(overlay.cols, overlay.rows),
                      cv::Scalar(0, 255, 0), cv::FILLED);
        cv::Mat blended;
        cv::addWeighted(overlay, 0.05, roi, 0.95, 0, blended);
        blended.copyTo(roi);
    }

    cv::imwrite(path.string(), display);
}
