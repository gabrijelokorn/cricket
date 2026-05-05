#include <iostream>

#include "Wav.hpp"

std::vector<double> hanning_window(int N)
{
    std::vector<double> window(N);
    for (int n = 0; n < N; ++n)
    {
        window[n] = 0.5 * (1 - cos((2 * M_PI * n) / (N - 1)));
    }
    return window;
}
double complex2magnitude(fftw_complex complex_number)
{
    return sqrt(complex_number[0] * complex_number[0] + complex_number[1] * complex_number[1]);
}

bool Wav::getSpec()
{
    std::vector<std::vector<double>> data_spec(this->getNumTimeFrames(), std::vector<double>(this->getNumFreqBins(), 0.0));
    std::vector<double> smoothing_window = hanning_window(gConfig.windowSize);

    double *in = (double *)fftw_malloc(sizeof(double) * gConfig.windowSize);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * this->getNumFreqBins());
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
        for (int bin = 0; bin < this->getNumFreqBins(); ++bin)
            data_spec[sample_index / gConfig.hopSize][bin] = log10(complex2magnitude(out[bin]) + 1e-2); // Add small value to avoid log(
    }

    // // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();

    for (const auto &frame : data_spec)
        for (double v : frame)
        {
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }

    cv::Mat mat(this->getNumFreqBins(), this->getNumTimeFrames(), CV_32F);

    for (int i = 0; i < this->getNumTimeFrames(); i++)
    {
        for (int j = 0; j < this->getNumFreqBins(); j++)
        {
            double norm = (data_spec[i][j] - minVal) / (maxVal - minVal);
            mat.at<float>(j, i) = (float)(norm * 255);
        }
    }
    cv::flip(mat, mat, 0);

    double minBin = (double)gConfig.specMinFreq / (double)this->getWavMaxFreq() * (double)this->getNumFreqBins();
    int startRow = std::max(0, (int)std::floor(minBin));
    double maxBin = (double)gConfig.specMaxFreq / (double)this->getWavMaxFreq() * (double)this->getNumFreqBins();
    int endRow = std::min(this->getNumFreqBins(), (int)std::ceil(maxBin));

    // mat = mat.rowRange(this->getNumFreqBins() - endRow, this->getNumFreqBins() - startRow);

    cv::imwrite("../spec.png", mat);

    return true;
}

Wav::Wav(const std::string &rPath)
{
    try
    {
        json wav_json = readJson(rPath);

        this->setRecPath(wav_json.at("rec_path").get<std::string>());
        this->setRecName(wav_json.at("rec_name").get<std::string>());

        if (wav_json.contains("courtship"))
            for (auto &entry : wav_json["courtship"])
                mCourtship.push_back({entry["start_time"], entry["end_time"]});

        if (wav_json.contains("noise"))
            for (auto &entry : wav_json["noise"])
                mNoise.push_back({entry["start_time"], entry["end_time"]});
    }

    catch (const std::exception &e)
    {
        std::cerr << "Error loading Wav file: " << e.what() << std::endl;
    }

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
    this->setSamplerate(wav_info.samplerate);
    this->setFreqRes(wav_info.samplerate / gConfig.windowSize);
    this->setNumFreqBins(gConfig.windowSize / 2 + 1);
    this->setNumTimeFrames((wav_info.frames - gConfig.windowSize) / gConfig.hopSize + 1);

    this->setWavMinFreq(0);
    this->setWavMaxFreq(wav_info.samplerate / 2);
    this->setWavFrames(wav_info.frames);
    this->setWavChannels(wav_info.channels);
    this->setDuration(wav_info.frames / wav_info.samplerate);

    this->mSoundData.resize(wav_info.frames);
    for (int i = 0; i < this->mSoundData.size(); ++i)
    {
        this->mSoundData[i] = wav_data[i * wav_info.channels + 1];
    }
}