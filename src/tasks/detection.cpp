#include <tasks/detection.hpp>
#include <arch/yolo/detector.hpp>

namespace det
{
    std::unique_ptr<trt::DetectionProcessor> DetectorFactory::create(const std::string &config_file)
    {
        auto arch = loadConfig<det::DetectorArch>(config_file, "detector");
        return rfl::visit([](auto config) -> std::unique_ptr<trt::DetectionProcessor> {
            return det::YoloFactory::create(std::move(config));
        }, arch);
    }
} // namespace det
