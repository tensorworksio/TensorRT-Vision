#pragma once
#include <fstream>
#include <utils/json_utils.hpp>
#include <engine/processor.hpp>

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

    static ReIdConfig load(const std::string &filename)
    {
        std::ifstream file(filename);
        auto data = nlohmann::json::parse(file);

        ReIdConfig config;
        config.loadFromJson(data["reid"]);
        return config;
    }
};

class ReId : public trt::ModelProcessor<std::vector<float>>
{
public:
    ReId(const ReIdConfig &config) : ModelProcessor(config.engine), m_config(config) {}
    const ReIdConfig &getConfig() const { return m_config; };

protected:
    bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
    std::vector<float> postprocess(const std::vector<float> &featureVector) override;

private:
    const ReIdConfig m_config;
};