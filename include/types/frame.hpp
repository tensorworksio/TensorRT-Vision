#pragma once

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <types/detection.hpp>

using Timestamp = std::chrono::system_clock::time_point;

struct Frame
{
    cv::Mat image;
    std::string source;
    Timestamp timestamp;

    static uint64_t id;
    nlohmann::json metadata;
    std::vector<std::unique_ptr<Detection>> detections;

    static uint64_t getNextId() { return ++id; };

    Frame(const cv::Mat &img, const std::string &src = "")
        : image(img), source(src), timestamp(std::chrono::system_clock::now())
    {
        id = getNextId();
    };
};

inline uint64_t Frame::id = 0;
using FramePtr = std::unique_ptr<Frame>;