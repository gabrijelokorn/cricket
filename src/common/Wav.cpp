#include <iostream>
#include <filesystem>

#include "cnpy.h"
#include "Wav.hpp"

/**
 * @brief A function for smoothening STFT.
 *
 * @param window_size is number of frames FFT is calculated upon
 * @return std::vector<double> window of spectrogram values
 */
std::vector<double> hanning_window(int window_size)
{
    std::vector<double> window(window_size);
    for (int n = 0; n < window_size; ++n)
    {
        window[n] = 0.5 * (1 - cos((2 * M_PI * n) / (window_size - 1)));
    }
    return window;
}
/**
 * @brief Convert values from complex to magnitude
 *
 * @param complex_number
 * @return double
 */
double complex2magnitude(fftw_complex complex_number)
{
    return sqrt(complex_number[0] * complex_number[0] + complex_number[1] * complex_number[1]);
}

double Wav::freqToBin(double freq)
{
    return (freq / (double)this->getWavMaxFreq()) * (double)this->getWavNumFreqBins();
}

std::string Wav::getRecName() const
{
    // e.g. "raven.251208.052755.wav" -> stem "raven.251208.052755" -> "raven.251208_052755"
    std::string stem = std::filesystem::path(this->mRecPath).stem().string();
    size_t firstDot = stem.find('.');
    if (firstDot == std::string::npos)
        return stem;

    std::string rest = stem.substr(firstDot + 1);
    std::replace(rest.begin(), rest.end(), '.', '_');
    return stem.substr(0, firstDot) + "." + rest;
}

int Wav::timeToFrame(double t) const
{
    return t / (double)this->getWavDuration() * (double)this->mSpec.cols;
}

double Wav::frameToTime(int f) const
{
    return ((double)(f * gConfig.hopSize) / (double)this->getWavSamplerate());
}

FrameInterval TimeInterval::toFrameInterval(const Wav &w) const
{
    return FrameInterval{w.timeToFrame(start), w.timeToFrame(end)};
}

TimeInterval FrameInterval::toTimeInterval(const Wav &w) const
{
    return TimeInterval{w.frameToTime(start), w.frameToTime(end)};
}

void Wav::exportSpectrogram()
{
    std::string rPath = gConfig.recordsPath + "/" + this->getRecName() + "_spectrogram.png";
    cv::imwrite(rPath, this->mSpec);
}

FrameInterval Wav::trimFrameInterval(FrameInterval fi)
{
    if (!gConfig.eventSize)
    {
        Logger::Warn("Event size: %d", gConfig.eventSize);
        return fi;
    }

    if (fi.end - fi.start <= gConfig.eventSize)
    {
        int diff = gConfig.eventSize - (fi.end - fi.start);
        int padLeft = diff / 2;
        int padRight = diff - padLeft; // takes the extra 1 if diff is odd

        fi.start -= padLeft;
        fi.end += padRight;
    }
    else
    {
        int diff = (fi.end - fi.start) - gConfig.eventSize;
        int trimLeft = diff / 2;
        int trimRight = diff - trimLeft;

        fi.start += trimLeft;
        fi.end -= trimRight;
    }

    fi.start = std::max(0, fi.start);
    fi.end = std::min(this->mSpec.cols, fi.end);

    return fi;
}

TimeInterval Wav::trimTimeInterval(TimeInterval ti)
{
    if (!gConfig.eventSize)
    {
        Logger::Warn("Event size: %d", gConfig.eventSize);
        return ti;
    }

    // Convert to frames and normalize
    FrameInterval fi = ti.toFrameInterval(*this);
    fi = this->trimFrameInterval(fi);

    return fi.toTimeInterval(*this);
}

cv::Mat Wav::trimFrequencyRange(cv::Mat spec)
{
    int startRow = std::max(0, (int)std::floor(this->freqToBin(gConfig.clipMinFreq)));
    int endRow = std::min(this->getWavNumFreqBins(), (int)std::ceil(this->freqToBin(gConfig.clipMaxFreq)));
    spec(cv::Range(this->getWavNumFreqBins() - endRow, this->getWavNumFreqBins() - startRow), cv::Range::all()).copyTo(spec);

    return spec;
}

cv::Mat Wav::getClipAtFrame(int start)
{
    return this->mSpec.colRange(start, start + gConfig.eventSize);
}

cv::Mat Wav::getClipByFrameInterval(FrameInterval fi)
{
    return this->mSpec.colRange(fi.start, fi.end);
}

cv::Mat Wav::getClipByTimeInterval(TimeInterval ti)
{
    FrameInterval fi = ti.toFrameInterval(*this);
    return this->getClipByFrameInterval(fi);
}

void Wav::exportLabeledClips(const std::vector<TimeInterval> &labeled, const std::string &outDir)
{
    int count = 0;

    for (const TimeInterval &t : labeled)
    {
        TimeInterval reframed = this->trimTimeInterval(t);
        cv::Mat clip = this->getClipByTimeInterval(reframed);
        clip = this->trimFrequencyRange(clip);

        if (clip.cols != gConfig.eventSize)
        {
            Logger::Warn("Skipping clip with %d frames (expected %d) — likely near a recording boundary: %s [%f, %f]",
                         clip.cols, gConfig.eventSize, this->getRecName().c_str(), reframed.start, reframed.end);
            continue;
        }

        std::string ext = (gConfig.clipFormat == ClipFormat::PNG) ? "png" : "npy";
        long startMs = std::lround(reframed.start * 1000.0);
        long endMs = std::lround(reframed.end * 1000.0);
        std::string rPath = outDir + "/" + this->getRecName()
            + "." + std::to_string(startMs) + "_" + std::to_string(endMs)
            + "." + std::to_string(gConfig.clipMinFreq) + "_" + std::to_string(gConfig.clipMaxFreq)
            + "." + ext;

        if (gConfig.clipFormat == ClipFormat::PNG)
        {
            cv::imwrite(rPath, clip);
        }
        else
        {
            if (!clip.isContinuous())
                clip = clip.clone();
            cnpy::npy_save(rPath, clip.ptr<float>(0), {(size_t)clip.rows, (size_t)clip.cols}, "w");
        }

        count++;
    }

    Logger::Info("Exported %d clips from %s to %s", count, this->getRecPath().c_str(), outDir.c_str());
}

void Wav::exportLabeledTicks()
{
    this->exportLabeledClips(this->mLabeledTicks, gConfig.ticksClipsPath);
}

void Wav::exportLabeledNoise()
{
    this->exportLabeledClips(this->mLabeledNoise, gConfig.noiseClipsPath);
}

bool Wav::getSpec()
{
    std::vector<std::vector<double>> data_spec(this->getWavNumTimeFrames(), std::vector<double>(this->getWavNumFreqBins(), 0.0));
    std::vector<double> smoothing_window = hanning_window(gConfig.windowSize);

    double *in = (double *)fftw_malloc(sizeof(double) * gConfig.windowSize);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * this->getWavNumFreqBins());
    fftw_plan plan = fftw_plan_dft_r2c_1d(gConfig.windowSize, in, out, FFTW_ESTIMATE);

    // sample_index - index of the first sample in the current window
    for (int sample_index = 0; sample_index + gConfig.windowSize <= this->mSoundData.size(); sample_index += gConfig.hopSize)
    {
        // Fill input buffer with windowed samples, averaging across channels
        for (int i = 0; i < gConfig.windowSize; ++i)
        {
            try
            {
                double sample_value = this->mSoundData.at(sample_index + i + 0); // No channel offset needed
                in[i] = sample_value * smoothing_window[i];
            }
            catch (const std::out_of_range &e)
            {
                std::cerr << "[ERROR] mSoundData out of range at sample_index="
                          << sample_index << " i=" << i << " : " << e.what() << "\n";
                in[i] = 0.0;
            }
        }

        fftw_execute(plan);

        // Extract magnitude for each frequency bin
        for (int bin = 0; bin < this->getWavNumFreqBins(); ++bin)
            data_spec[sample_index / gConfig.hopSize][bin] = log10(complex2magnitude(out[bin]) + 1e-2); // Add small value to avoid log(
    }

    // // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    // Per-bin baseline: median of each frequency bin across this file, so a loud event in one bin can't skew the normalization of every other bin.
    std::vector<double> binBaseline(this->getWavNumFreqBins());
    std::vector<double> column(this->getWavNumTimeFrames());

    for (int j = 0; j < this->getWavNumFreqBins(); j++)
    {
        for (int i = 0; i < this->getWavNumTimeFrames(); i++)
            column[i] = data_spec[i][j];

        std::nth_element(column.begin(), column.begin() + column.size() / 2, column.end());
        binBaseline[j] = column[column.size() / 2];
    }

    // Fixed range, same for every file, so the same physical event maps to the same output value regardless of what else happened in that recording.
    const double RANGE_LOW = 0.0;
    const double RANGE_HIGH = 3.0;

    this->mSpec.create(std::vector<int>{this->getWavNumFreqBins(), this->getWavNumTimeFrames()}, CV_32F);

    for (int i = 0; i < this->getWavNumTimeFrames(); i++)
    {
        for (int j = 0; j < this->getWavNumFreqBins(); j++)
        {
            double rel = data_spec[i][j] - binBaseline[j];
            double norm = std::clamp((rel - RANGE_LOW) / (RANGE_HIGH - RANGE_LOW), 0.0, 1.0);
            this->mSpec.at<float>(j, i) = (float)(norm * 255);
        }
    }
    cv::flip(this->mSpec, this->mSpec, 0);

    return true;
}

Wav::Wav(const std::string &rPath)
{
    try
    {
        json wav_json = readJson(rPath);

        this->setRecPath(wav_json.at("rec_path").get<std::string>());

        if (wav_json.contains("ticks"))
            for (auto &entry : wav_json["ticks"])
                this->mLabeledTicks.push_back({entry["start_time"], entry["end_time"]});

        if (wav_json.contains("noise"))
            for (auto &entry : wav_json["noise"])
                this->mLabeledNoise.push_back({entry["start_time"], entry["end_time"]});
    }

    catch (const std::exception &e)
    {
        std::cerr << "Error loading Wav file: " << e.what() << std::endl;
    }

    this->loadAudio();
}

Wav::Wav(const std::string &wavPath, RawWavTag)
{
    this->setRecPath(wavPath);
    this->loadAudio();
}

void Wav::loadAudio()
{
    SF_INFO wav_info{};
    SNDFILE *wav_file = sf_open(this->getRecPath().c_str(), SFM_READ, &wav_info);
    if (!wav_file)
    {
        std::cerr << "Error opening file: " << this->getRecPath() << std::endl;
        return;
    }
    std::vector<float> wav_data(wav_info.frames * wav_info.channels);
    sf_readf_float(wav_file, wav_data.data(), wav_info.frames);
    sf_close(wav_file);

    // Set parameters for computing the spectrogram
    this->setWavSamplerate(wav_info.samplerate);
    this->setWavFreqRes(wav_info.samplerate / gConfig.windowSize);
    this->setWavNumFreqBins(gConfig.windowSize / 2 + 1);
    this->setWavNumTimeFrames((wav_info.frames - gConfig.windowSize) / gConfig.hopSize + 1);

    this->setWavMinFreq(0);
    this->setWavMaxFreq(wav_info.samplerate / 2);
    this->setWavFrames(wav_info.frames);
    this->setWavChannels(wav_info.channels);
    this->setWavDuration(wav_info.frames / wav_info.samplerate);

    this->mSoundData.resize(wav_info.frames);
    for (int i = 0; i < this->mSoundData.size(); ++i)
    {
        this->mSoundData[i] = wav_data[i * wav_info.channels + 1];
    }
}