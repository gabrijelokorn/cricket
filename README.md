# cricket
Software to recognize courtship sounds from nature recordings.

## Project Setup
### Conan
```
conan install . --output-folder=build --build=missing
```

### Cmake build
```
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake
```
