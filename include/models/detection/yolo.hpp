#pragma once

#include "detector.hpp"

class Yolo : public Detector
{
public:
    using Detector::Detector;

private:
    bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size) override;
    virtual std::vector<Detection> postprocess(const std::vector<float> &featureVector);

protected:
    float m_ratioWidth = 1.f;
    float m_ratioHeight = 1.f;
    float m_imgWidth = 0.f;
    float m_imgHeight = 0.f;
};

class Yolov7 : public Yolo
{
public:
    using Yolo::Yolo;

private:
    std::vector<Detection> postprocess(const std::vector<float> &featureVector) override;
};

using Yolov8 = Yolo;
using Yolov11 = Yolo;