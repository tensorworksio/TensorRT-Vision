#include <tasks/detection.hpp>
#include <arch/yolo/detector.hpp>

namespace det
{
    std::unique_ptr<trt::DetectionProcessor> DetectorFactory::create(const std::string &config_file)
    {
        std::ifstream file(config_file);
        auto data = nlohmann::json::parse(file, nullptr, true, true);
        std::string arch = data["detector"]["architecture"].get<std::string>();
        std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);

        if (arch == "yolo")
            return YoloFactory::create(data);
        throw std::runtime_error("Unknown detector architecture: " + arch);
    }
} // namespace det
