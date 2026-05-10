#pragma once

#include <types/detection.hpp>
#include <utils/config_utils.hpp>
#include <engine/processor.hpp>
#include <engine/interface.hpp>

namespace cls
{

    struct ClassifierConfig
    {
        trt::EngineConfig engine{};
        float confidence_threshold = 0.9f;
        std::optional<std::string> class_names_file{};
        std::vector<std::string> class_names{};

        static ClassifierConfig load(const std::string &filename)
        {
            auto config = loadConfig<ClassifierConfig>(filename);
            if (config.class_names_file)
                config.class_names = loadClassNamesFromFile(*config.class_names_file);
            return config;
        }
    };

    class BaseClassifier : public trt::ClassificationProcessor, public trt::SISOProcessor<Detection>
    {
    public:
        BaseClassifier(const ClassifierConfig &t_config) : trt::SISOProcessor<Detection>(t_config.engine), config(t_config) {}
        virtual ~BaseClassifier() = default;

        Detection process(const cv::Mat &frame) override
        {
            return trt::SISOProcessor<Detection>::process(frame);
        }

        std::vector<Detection> process(const std::vector<cv::Mat> &frames) override
        {
            return trt::SISOProcessor<Detection>::process(frames);
        }

        const ClassifierConfig &getConfig() const { return config; }

        const std::string getClassName(int class_id) const
        {
            return (static_cast<size_t>(class_id) < config.class_names.size()) ? config.class_names[class_id] : std::to_string(class_id);
        }

    protected:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) override;
        const ClassifierConfig config;
    };

    class SingleLabelClassifier : public BaseClassifier
    {
    public:
        SingleLabelClassifier(const ClassifierConfig &t_config) : BaseClassifier(t_config) {}

    protected:
        Detection postprocess(const trt::SingleOutput &featureVector) override;
    };

    class MultiLabelClassifier : public BaseClassifier
    {
    public:
        MultiLabelClassifier(const ClassifierConfig &t_config) : BaseClassifier(t_config) {}

    protected:
        Detection postprocess(const trt::SingleOutput &featureVector) override;
    };

} // namespace cls
