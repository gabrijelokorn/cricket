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
    
    int mSampleRate;
    int mFreqRes;
    int mNumBins;
    int mNumTFrames;
    
    int mWavMinFreq;
    int mWavMaxFreq;
    int mWavFrames;
    int mWavChannels;
    int mDuration;

    // wav data
    std::vector<double> mSoundData;
    cv::Mat spec;

    std::vector<TimeInterval> mCourtship;
    std::vector<TimeInterval> mNoise;
public:
    Wav() = default;
    ~Wav() = default;

    Wav(const std::string& rPath);

    void setRecPath(const std::string& recPath) { mRecPath = recPath; }
    std::string getRecPath() const { return mRecPath; }
    void setRecName(const std::string& recName) { mRecName = recName; }
    std::string getRecName() const { return mRecName; }

    void setSamplerate(int sampleRate) { mSampleRate = sampleRate; }
    int getSamplerate() const { return mSampleRate; }
    void setFreqRes(int freqRes) { mFreqRes = freqRes; }
    int getFreqRes() const { return mFreqRes; }
    void setNumBins(int numBins) { mNumBins = numBins; }
    int getNumBins() const { return mNumBins; }
    void setNumTFrames(int numTFrames) { mNumTFrames = numTFrames; }
    int getNumTFrames() const { return mNumTFrames; }

    void setWavMinFreq(int minFreq) { mWavMinFreq = minFreq; }
    int getWavMinFreq() const { return mWavMinFreq; }
    void setWavMaxFreq(int maxFreq) { mWavMaxFreq = maxFreq; }
    int getWavMaxFreq() const { return mWavMaxFreq; }
    void setWavFrames(int frames) { mWavFrames = frames; }
    int getWavFrames() const { return mWavFrames; }
    void setWavChannels(int channels) { mWavChannels = channels; }
    int getWavChannels() const { return mWavChannels; }
    void setDuration(int duration) { mDuration = duration; }
    int getDuration() const { return mDuration; }

    bool getSpec();
};

#endif // WAV_HPP