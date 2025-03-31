#pragma once
#include <fstream>
#include <utils/json_utils.hpp>
#include <engine/processor.hpp>

namespace reid
{

    struct ReIdConfig : public JsonConfig
    {
        trt::EngineConfig engine{};
        float confidenceThreshold = 0.5f;

        std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<ReIdConfig>(*this); }

        void loadFromJson(const nlohmann::json &data) override
        {
            if (data.contains("engine"))
                engine.loadFromJson(data["engine"]);
            if (data.contains("confidence_threshold"))
                confidenceThreshold = data["confidence_threshold"].get<float>();
        }

        static ReIdConfig load(const std::string &filename, const std::string &task = "")
        {
            std::ifstream file(filename);
            auto data = nlohmann::json::parse(file);

            ReIdConfig config;
            if (task.empty())
            {
                config.loadFromJson(data);
            }
            else if (data.contains(task))
            {
                config.loadFromJson(data[task]);
            }
            else
            {
                throw std::runtime_error("Config file does not contain task: " + task);
            }

            return config;
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