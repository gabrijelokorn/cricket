# Cricket

Detects house cricket courtship signals in audio recordings using a CNN. Method and results are described in [Cricket.pdf](Cricket.pdf).

![Example of recognized courtship](./assets/presentation/courship_sounds_example.png)
*Spectrogram of a recording. Recognized courtship sounds are marked green.*

## How it works

- A courtship is a sequence of **ticks** (short, high-frequency chirps) and **pulses** (low-frequency). Ticks at least 3 s apart belong to different courtships, so the task reduces to detecting ticks.
- Recordings are 48 kHz stereo WAV (audio + vibration channel).
- Each recording becomes a spectrogram (STFT window 256, hop 128) cropped to 5–24 kHz, where ticks live.
- A CNN classifies 24-column clips (~64 ms) as tick or noise (pulses, alarm calls, background).
- `clipclass` slides the clip across a recording and groups detected ticks into courtships.

All parameters are in `assets/config.json`.

**Results:** leave-one-session-out cross-validation over 7 recording days (5,570 clips) gives balanced accuracy **0.9905** (STFT window 256).

## Setup

Requires Conan, CMake and [libtorch](https://pytorch.org/get-started/locally/) (path set via `CMAKE_PREFIX_PATH` in `CMakeLists.txt`). Training requires Python with `torch` and `numpy`.

From the project root:

```bash
conan install . --build=missing -s build_type=Debug --output-folder=build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake
cmake --build .
```

`CMakeLists.txt` forces a Debug build, so the Conan build type must match.

## Usage

Run the C++ programs from the *build* directory and the Python scripts from *src/train*. Inputs are picked through a file dialog.

### 1. Label

Each `assets/records/<rec>.wav` has a `<rec>.json` listing its labeled `ticks` and `noise` intervals (`start_time`, `end_time` in seconds).

### 2. Generate clips (clipgen)

Cuts the labeled intervals into training clips in `assets/clips/`.

```bash
./clipgen --clips [--format npy|png]
./clipgen --spectrograms               # full spectrogram PNG per recording
cmake --build . --target clean-clips   # delete generated clips
```

### 3. Train (Python)

Model architectures are registered in `models.py`; `--model` picks one (default: `cnn`).

```bash
python3 stats.py [--by date|recording]   # labeled clip counts
python3 evaluation.py [--model all]      # LOSO cross-validation → output/evaluation/<model>/
python3 main.py                          # train on all clips → assets/models/<model>.pt
```

### 4. Classify (clipclass)

Classifies every `.wav` in the selected folders and writes `<rec>.clips.csv` and `<rec>.courtships.csv` to `output/<folder>/`.

```bash
./clipclass [--model cnn] [--clips] [--courtships]   # flags also save marked spectrogram PNGs
```
