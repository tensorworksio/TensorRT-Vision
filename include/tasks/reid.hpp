#pragma once

#include <types/detection.hpp>
#include <utils/config_utils.hpp>
#include <engine/processor.hpp>

namespace reid
{

    struct ReIdConfig
    {
        trt::EngineConfig engine{};
        float confidence_threshold = 0.5f;

        static ReIdConfig load(const std::string &filename, const std::string &task = "")
        {
            return loadConfig<ReIdConfig>(filename, task);
        }
    };

    class ReId : public trt::SISOProcessor<std::vector<float>>
    {
    public:
        ReId(const ReIdConfig &config) : trt::SISOProcessor<std::vector<float>>(config.engine), m_config(config) {}
        const ReIdConfig &getConfig() const { return m_config; };

    protected:
        bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg) override;
        std::vector<float> postprocess(const trt::SingleOutput &featureVector) override;

    private:
        const ReIdConfig m_config;
    };

} // namespace reid
