# Cricket
Software to recognize cricket courtship sounds from nature recordings.

## Example of recognized courtships sounds
 - Picutre below shows spectrogram of audio recording.
    ![Example of recognized courtship](./assets/presentation/courship_sounds_example.png)
    - Recognized singular courtship sounds are marked as green area.

## Project Setup (development)
### Conan
Run the following command in project root directory:
```
conan install . --build=missing -s build_type=Release --output-folder=build
```

### Cmake build
Run the following command in the *build* directory:
```
cmake ..
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
