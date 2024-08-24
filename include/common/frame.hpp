#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "detection.hpp"

struct Frame
{
    int idx = 0;
    cv::Mat image;
    std::vector<Detection> detected_objects;
};