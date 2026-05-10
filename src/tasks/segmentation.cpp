#include <tasks/segmentation.hpp>
#include <arch/yolo/segmenter.hpp>

namespace seg
{
    std::unique_ptr<trt::DetectionProcessor> SegmenterFactory::create(const std::string &config_file)
    {
        auto arch = loadConfig<seg::SegmenterArch>(config_file);
        return rfl::visit([](auto config) -> std::unique_ptr<trt::DetectionProcessor> {
            return seg::YoloFactory::create(std::move(config));
        }, arch);
    }
} // namespace seg
