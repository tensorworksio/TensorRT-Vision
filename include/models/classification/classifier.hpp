#pragma once

#include <fstream>
#include <types/detection.hpp>
#include <utils/json_utils.hpp>
#include <engine/processor.hpp>

struct ClassifierConfig : JsonConfig
{
    trt::EngineConfig engine;
    float confidenceThreshold{0.9f};
    std::vector<std::string> classNames{};

    void loadFromJson(const nlohmann::json &data) override
    {
        if (data.contains("engine"))
            engine.loadFromJson(data["engine"]);
        if (data.contains("confidence_threshold"))
            confidenceThreshold = data["confidence_threshold"].get<float>();
        if (data.contains("class_names"))
            classNames = data["class_names"].get<std::vector<std::string>>();
    }

    static ClassifierConfig load(const std::string &filename, const std::string &task = "")
    {
        std::ifstream file(filename);
        auto data = nlohmann::json::parse(file);

        ClassifierConfig config;
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

    std::shared_ptr<const JsonConfig> clone() const override { return std::make_shared<ClassifierConfig>(*this); }
};

class Classifier : public trt::ModelProcessor<Detection>
{
public:
    Classifier(const ClassifierConfig &t_config) : ModelProcessor(t_config.engine), config(t_config) {}
    const ClassifierConfig &getConfig() const { return config; };
    const std::string getClassName(int class_id) const
    {
        return (static_cast<size_t>(class_id) < config.classNames.size()) ? config.classNames[class_id] : std::to_string(class_id);
    };

protected:
    bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
    Detection postprocess(const std::vector<float> &featureVector) override;

private:
    const ClassifierConfig config;
};