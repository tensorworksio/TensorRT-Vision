#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

struct Detection
{
    int class_id;
    float probability;
    cv::Rect2f bbox;
    std::string class_name;

    std::vector<float> features{};
};