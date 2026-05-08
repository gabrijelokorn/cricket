#ifndef WAV_HPP
#define WAV_HPP

#include <string>
#include <sndfile.h>
#include <fftw3.h>
#include <opencv2/opencv.hpp>
#include <cmath>

#include "json.hpp"
#include "config.hpp"

struct TimeInterval
{
    double start;
    double end;
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

    std::vector<TimeInterval> mCourtship;
    std::vector<TimeInterval> mNoise;

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

    std::vector<TimeInterval> getCourtship() const { return mCourtship; }
    std::vector<TimeInterval> getNoise() const { return mNoise; }

    int getTimeFrame(double ms);
    void clip(TimeInterval t, const std::string rPath);
    void clipCourtship();
    void clipNoise();

    double getFreqBin(double freq);
    bool getSpec();
};

#endif // WAV_HPP