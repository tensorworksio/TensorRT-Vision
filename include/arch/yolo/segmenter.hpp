#pragma once

#include <types/detection.hpp>
#include <arch/yolo/config.hpp>
#include <tasks/segmentation.hpp>

namespace seg
{
    enum class YoloVersion { YOLOv8, YOLOv11, UNKNOWN };

    inline YoloVersion getYoloVersion(const std::string &name)
    {
        std::string lower = name;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        if (lower == "yolov8")  return YoloVersion::YOLOv8;
        if (lower == "yolov11") return YoloVersion::YOLOv11;
        return YoloVersion::UNKNOWN;
    }

    using YoloConfig = YoloBaseConfig;
    using SegmenterArch = rfl::TaggedUnion<"architecture", YoloConfig>;

    class Yolo : public Segmenter<trt::MultiOutput>
    {
    public:
        Yolo(const YoloConfig &t_config)
            : Segmenter<trt::MultiOutput>(t_config.engine), config(t_config) {}
        virtual ~Yolo() = default;
        const YoloConfig &getConfig() const { return config; }
        const std::string getClassName(int class_id) const
        {
            return (static_cast<size_t>(class_id) < config.class_names.size())
                       ? config.class_names[class_id]
                       : std::to_string(class_id);
        }

    protected:
        const YoloConfig config;

    private:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) override;
        std::vector<Detection> postprocess(const trt::MultiOutput &engineOutputs) override;
    };

    using Yolov8  = Yolo;
    using Yolov11 = Yolo;

    class YoloFactory
    {
    public:
        static std::unique_ptr<Yolo> create(YoloConfig config)
        {
            if (config.class_names_file)
                config.class_names = loadClassNamesFromFile(*config.class_names_file);

            switch (getYoloVersion(config.name))
            {
            case YoloVersion::YOLOv8:  return std::make_unique<Yolov8>(config);
            case YoloVersion::YOLOv11: return std::make_unique<Yolov11>(config);
            default: throw std::runtime_error("Unsupported yolo version: " + config.name);
            }
        }
    };

} // namespace seg
