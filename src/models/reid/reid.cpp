#include <models/reid/reid.hpp>
#include <utils/vector_utils.hpp>
#include <utils/detection_utils.hpp>

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

std::vector<float> ReId::postprocess(const std::vector<float> &featureVector)
{
    return vector_ops::normalize(featureVector);
}