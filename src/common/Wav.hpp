#ifndef WAV_HPP
#define WAV_HPP

#include <string>
#include <sndfile.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>

#include "json.hpp"
#include "config.hpp"
#include "logger.hpp"

class Wav;
struct TimeInterval;
struct FrameInterval;
struct TimeInterval
{
    double start;
    double end;

    FrameInterval toFrameInterval(const Wav &w) const;
};

struct FrameInterval
{
    int start;
    int end;

    TimeInterval toTimeInterval(const Wav &w) const;
};

struct Courtship
{
    std::vector<TimeInterval> events;
    double start;
    double end;

    void addEvent(const TimeInterval &ti)
    {
        events.push_back(ti);
        start = events.front().start;
        end = events.back().end;
    }
};

class Wav
{
private:
    std::string mRecPath;
    std::string mRecName;

    int mWavSampleRate;
    int mWavFreqRes;
    int mWavNumFreqBins;
    int mWavNumTimeFrames;
    int mWavMinFreq;
    int mWavMaxFreq;
    int mWavFrames;
    int mWavChannels;
    int mWavDuration;

    // wav data
    std::vector<double> mSoundData;
    cv::Mat mSpec;

    std::vector<TimeInterval> mLabeledCourtship;
    std::vector<TimeInterval> mLabeledNoise;

public:
    Wav() = default;
    ~Wav() = default;

    Wav(const std::string &rPath);

    void setRecPath(const std::string &recPath) { mRecPath = recPath; }
    std::string getRecPath() const { return mRecPath; }
    void setRecName(const std::string &recName) { mRecName = recName; }
    std::string getRecName() const { return mRecName; }

    void setWavSamplerate(int sampleRate) { mWavSampleRate = sampleRate; }
    int getWavSamplerate() const { return mWavSampleRate; }
    void setWavFreqRes(int freqRes) { mWavFreqRes = freqRes; }
    int getWavFreqRes() const { return mWavFreqRes; }
    void setWavNumFreqBins(int numBins) { mWavNumFreqBins = numBins; }
    int getWavNumFreqBins() const { return mWavNumFreqBins; }
    void setWavNumTimeFrames(int numTFrames) { mWavNumTimeFrames = numTFrames; }
    int getWavNumTimeFrames() const { return mWavNumTimeFrames; }

    void setWavMinFreq(int minFreq) { mWavMinFreq = minFreq; }
    int getWavMinFreq() const { return mWavMinFreq; }
    void setWavMaxFreq(int maxFreq) { mWavMaxFreq = maxFreq; }
    int getWavMaxFreq() const { return mWavMaxFreq; }
    void setWavFrames(int frames) { mWavFrames = frames; }
    int getWavFrames() const { return mWavFrames; }
    void setWavChannels(int channels) { mWavChannels = channels; }
    int getWavChannels() const { return mWavChannels; }
    void setWavDuration(int duration) { mWavDuration = duration; }
    int getWavDuration() const { return mWavDuration; }

    std::vector<TimeInterval> getLabeledCourtship() const { return mLabeledCourtship; }
    std::vector<TimeInterval> getLabeledNoise() const { return mLabeledNoise; }

    // Conversions
    double freqToBin(double freq);
    double frameToTime(int f) const;
    int timeToFrame(double t) const;

    cv::Mat trimFrequencyRange(cv::Mat spec);
    FrameInterval trimFrameInterval(FrameInterval fi);
    TimeInterval trimTimeInterval(TimeInterval ti);

    cv::Mat getClipAtFrame(int start);
    cv::Mat getClipByFrameInterval(FrameInterval fi);
    cv::Mat getClipByTimeInterval(TimeInterval ti);
    void exportClip(cv::Mat clip, const std::string rPath);
    void exportSpectrogram();
    void clipCourtship();
    void clipNoise();

    bool getSpec();

    cv::Mat getMSpec() { return this->mSpec; }
};

#endif // WAV_HPP