# cricket
Software to recognize courtship sounds from nature recordings.

## Project Setup
### Conan
Run the following command in project root directory:
```
conan install . --output-folder=build --build=missing
```

### Cmake build
Run the following command in the *build* directory:
```
cmake ..
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
```
