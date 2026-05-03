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
    std::vector<std::vector<double>> data_spec(this->getNumTFrames(), std::vector<double>(this->getNumBins(), 0.0));
    std::vector<double> smoothing_window = hanning_window(gConfig.windowSize);

    double *in = (double *)fftw_malloc(sizeof(double) * gConfig.windowSize);
    fftw_complex *out = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * this->getNumBins());
    fftw_plan plan = fftw_plan_dft_r2c_1d(gConfig.windowSize, in, out, FFTW_ESTIMATE);

    // sample_index - index of the first sample in the current window
    for (int sample_index = 0; sample_index + gConfig.windowSize <= this->getWavFrames(); sample_index += gConfig.hopSize)
    {
        // Fill input buffer with windowed samples, averaging across channels
        // for (int i = 0; i < gConfig.windowSize; ++i)
        // {
        //     double sample_value = 0.0;
        //     // Only first channel.
        //     for (int ch = 0; ch < this->getWavChannels(); ++ch)
        //     {
        //         if (ch == this->getWavChannels())
        //             continue;
        //         sample_value += mSoundData[(sample_index + i) * this->getWavChannels() + ch];
        //     }
        //     in[i] = sample_value * smoothing_window[i];
        // }
        for (int i = 0; i < gConfig.windowSize; ++i)
        {
            try
            {
                double sample_value = mSoundData.at(sample_index + i); // no channel offset needed
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
        for (int bin = 0; bin < this->getNumBins(); ++bin)
            data_spec[sample_index / gConfig.hopSize][bin] = 20 * log10(complex2magnitude(out[bin]) + 1e-2);
    }

    // // Cleanup
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    double minVal = std::numeric_limits<double>::infinity();
    double maxVal = -std::numeric_limits<double>::infinity();

    this->spec = cv::Mat(this->getNumTFrames(), this->getNumBins(), CV_32F, cv::Scalar(0.0));

    for (const auto &frame : data_spec)
        for (double v : frame)
        {
            minVal = std::min(minVal, v);
            maxVal = std::max(maxVal, v);
        }

    // cv::Mat mat(wav.getMSpectrogram()[0].size(), wav.getMSpectrogram().size(), CV_32F);
    cv::Mat mat(this->getNumBins(), this->getNumTFrames(), CV_32F);

    for (int i = 0; i < this->getNumTFrames(); i++)
    {
        for (int j = 0; j < this->getNumBins(); j++)
        {
            double norm = (data_spec[i][j] - minVal) / (maxVal - minVal);
            mat.at<float>(j, i) = (float)(norm * 255);
        }
    }

    cv::flip(mat, mat, 0);

    double minBin = (double)this->getWavMinFreq() / (double)this->getWavMaxFreq() * (double)this->getNumBins();
    double maxBin = (double)this->getWavMaxFreq() / (double)this->getWavMaxFreq() * (double)this->getNumBins();
    int startRow = std::max(0, (int)std::floor(minBin));
    int endRow = std::min(this->getNumBins(), (int)std::ceil(maxBin));

    mat = mat.rowRange(this->getNumBins() - endRow, this->getNumBins() - startRow);

    cv::imwrite("./spec.png", mat);

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
    this->setNumBins(this->getFreqRes() / 2 + 1);
    this->setNumTFrames((wav_info.frames - gConfig.windowSize) / gConfig.hopSize + 1);

    this->setWavMinFreq(0);
    this->setWavMaxFreq(wav_info.samplerate / 2);
    this->setWavFrames(wav_info.frames);
    this->setWavChannels(wav_info.channels);
    this->setDuration(wav_info.frames / wav_info.samplerate);

    mSoundData.resize(wav_info.frames);
    for (int i = 0; i < wav_info.frames; ++i)
    {
        mSoundData[i] = wav_data[i * wav_info.channels + 0]; // extract channel 0 only
    }
}