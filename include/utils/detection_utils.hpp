#pragma once

#include <opencv2/opencv.hpp>

inline cv::Mat letterbox(const cv::Mat &input, cv::Size new_shape, cv::Scalar color, bool auto_size, bool scale_fill, bool scaleup, int stride)
{
    // Resize and pad image while meeting stride-multiple constraints
    cv::Mat out(new_shape, CV_8UC3, color);
    cv::Size shape = input.size(); // current shape [height, width]

    // Scale ratio (new / old)
    float r = std::min(static_cast<float>(new_shape.height) / shape.height, static_cast<float>(new_shape.width) / shape.width);
    if (!scaleup)
    {
        r = std::min(r, 1.0f);
    }

    // Compute padding
    std::pair<float, float> ratio(r, r); // width, height ratios
    cv::Size new_unpad(static_cast<int>(std::round(shape.width * r)), static_cast<int>(std::round(shape.height * r)));
    float dw = static_cast<float>(new_shape.width - new_unpad.width);
    float dh = static_cast<float>(new_shape.height - new_unpad.height); // wh padding

    if (auto_size)
    {
        dw = std::fmod(dw, static_cast<float>(stride));
        dh = std::fmod(dh, static_cast<float>(stride)); // wh padding
    }
    else if (scale_fill)
    {
        dw = 0.0;
        dh = 0.0;
        new_unpad = new_shape;
        ratio = std::make_pair(static_cast<float>(new_shape.width) / shape.width, static_cast<float>(new_shape.height) / shape.height); // width, height ratios
    }

    dw /= 2; // divide padding into 2 sides
    dh /= 2;

    if (shape != new_unpad)
    {
        cv::resize(input, out, new_unpad, 0, 0, cv::INTER_LINEAR);
    }

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::copyMakeBorder(out, out, top, bottom, left, right, cv::BORDER_CONSTANT, color); // add border

    return out;
}

inline std::vector<float> softmax(const std::vector<float> &logits)
{
    std::vector<float> exp_values(logits.size());
    std::vector<float> probabilities(logits.size());

    // Find max logit value for numerical stability
    auto max_logit = std::max_element(logits.begin(), logits.end());

    // Calculate exponentials and their sum
    float sum_exp = 0.f;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        // Substract max_logit for numerical stability
        exp_values[i] = std::exp(logits[i] - *max_logit);
        sum_exp += exp_values[i];
    }

    // Calculate softmax probabilities
    for (size_t i = 0; i < logits.size(); ++i)
    {
        probabilities[i] = exp_values[i] / sum_exp;
    }

    return probabilities;
}