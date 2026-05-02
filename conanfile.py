from conan import ConanFile

class CricketConan(ConanFile):
    generators=(
        "CMakeDeps",
        "CMakeToolchain"
    )

    settings = (
        "os",
        "build_type",
        "compiler",
        "arch"
    )

    def requirements(self):
        self.requires("nlohmann_json/3.12.0")
        # self.requires("fftw/3.3.10")
        # self.requires("libsndfile/1.0.31")
        # self.requires("opencv/3.4.20")

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.5]")
        
