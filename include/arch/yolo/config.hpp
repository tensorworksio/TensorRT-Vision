#pragma once

#include <utils/json_utils.hpp>
#include <engine/engine.hpp>

struct YoloBaseConfig : JsonConfig
{
    trt::EngineConfig engine{};
    float confidenceThreshold = 0.25f;
    float nmsThreshold = 0.45f;
    float nmsEta = 1.f;
    int topK = 100;
    std::vector<std::string> classNames{};

    void loadFromJson(const nlohmann::json &data) override
    {
        if (data.contains("engine"))
            engine.loadFromJson(data["engine"]);
        if (data.contains("confidence_threshold"))
            confidenceThreshold = data["confidence_threshold"].get<float>();
        if (data.contains("nms_threshold"))
            nmsThreshold = data["nms_threshold"].get<float>();
        if (data.contains("nms_eta"))
            nmsEta = data["nms_eta"].get<float>();
        if (data.contains("top_k"))
            topK = data["top_k"].get<int>();
        if (data.contains("class_names_file"))
            classNames = loadClassNamesFromFile(data["class_names_file"].get<std::string>());
        else if (data.contains("class_names"))
            classNames = data["class_names"].get<std::vector<std::string>>();
    }
};
