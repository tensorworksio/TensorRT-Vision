#pragma once

#include <optional>
#include <string>
#include <vector>
#include <rfl/Literal.hpp>
#include <engine/engine.hpp>
#include <utils/config_utils.hpp>

struct YoloBaseConfig
{
    using Tag = rfl::Literal<"yolo">;

    trt::EngineConfig engine{};
    std::string name{};
    float confidence_threshold = 0.25f;
    float nms_threshold = 0.45f;
    float nms_eta = 1.f;
    int top_k = 100;
    std::optional<std::string> class_names_file{};
    std::optional<float> mask_threshold{};
    std::vector<std::string> class_names{};
};
