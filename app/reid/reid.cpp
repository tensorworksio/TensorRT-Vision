#include "reid.hpp"
#include <utils/vector_utils.hpp>

bool ReId::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
{
    // The model expects RGB input
    cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);

    // Resize the model to the expected size and pad with background
    dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);

    // Convert to Float32
    dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);

    return !dstImg.empty();
}

bool ReId::postprocess(std::vector<float> &featureVector, std::vector<Detection> &detections)
{
    Detection det;
    det.features = vector_ops::normalize(featureVector);
    detections.push_back(det);
    return detections.size() == 1;
}

void ReId::process(const cv::Mat &image, std::vector<float> &featureVector)
{
    std::vector<Detection> detections;
    ModelProcessor::process(image, detections);

    if (detections.empty())
    {
        throw std::runtime_error("Detection vector must be non empty");
    }

    featureVector = detections[0].features;
}

void ReId::process(const std::vector<cv::Mat> &images, std::vector<std::vector<float>> &features)
{
    if (images.empty())
    {
        return;
    }
    std::vector<std::vector<Detection>> detections;
    ModelProcessor::process(images, detections);
    if (detections.size() != images.size())
    {
        throw std::runtime_error("Detection vector must be the same size as Image vector");
    }

    for (size_t i = 0; i < detections.size(); ++i)
    {
        features.push_back(detections[i][0].features);
    }
}