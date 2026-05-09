#include <tasks/segmentation.hpp>
#include <arch/yolo/segmenter.hpp>

namespace seg
{
    std::unique_ptr<trt::DetectionProcessor> SegmenterFactory::create(const std::string &config_file)
    {
        std::ifstream file(config_file);
        auto data = nlohmann::json::parse(file, nullptr, true, true);
        std::string arch = data["segmenter"]["architecture"].get<std::string>();
        std::transform(arch.begin(), arch.end(), arch.begin(), ::tolower);

        if (arch == "yolo")
            return YoloFactory::create(data);
        throw std::runtime_error("Unknown segmenter architecture: " + arch);
    }
} // namespace seg
