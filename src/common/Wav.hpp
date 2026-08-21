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

// Tag to select the raw-.wav-file constructor over the JSON-label one — both
// take a single string, so the compiler needs a second argument to tell them apart.
struct RawWavTag {};
inline constexpr RawWavTag rawWav{};

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
    // Construct directly from a .wav file, with no JSON label file — for scanning unlabeled recordings.
    Wav(const std::string &wavPath, RawWavTag);

    void setRecPath(const std::string &recPath) { mRecPath = recPath; }
    std::string getRecPath() const { return mRecPath; }
    // Derived from the wav filename itself — e.g. "raven.251208.052755.wav" -> "raven.251208_052755"
    std::string getRecName() const;

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
    void exportSpectrogram();
    void exportLabeledCourtship();
    void exportLabeledNoise();

    bool getSpec();

    cv::Mat getMSpec() { return this->mSpec; }

private:
    void exportLabeledClips(const std::vector<TimeInterval> &labeled, const std::string &outDir);
    void loadAudio();
};

#endif // WAV_HPP