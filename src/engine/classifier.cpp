#include <engine/classifier.hpp>
#include <utils/detection_utils.hpp>

const std::string Classifier::getClassName(int class_id) const
{
    if (config.classNames.empty())
    {
        return std::to_string(class_id);
    }
    return config.classNames[class_id];
}

bool Classifier::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
{
    // The model expects RGB image
    cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);

    // Resize and pad the image to the expected model size
    dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);

    // Convert image to float
    dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);

    return !dstImg.empty();
}

Detection Classifier::postprocess(const std::vector<float> &featureVector)
{
    Detection det;
    auto maxElement = std::max_element(featureVector.begin(), featureVector.end());
    if (*maxElement >= config.confidenceThreshold)
    {
        det.class_id = std::distance(featureVector.begin(), maxElement);
        det.confidence = *maxElement;
        det.class_name = getClassName(det.class_id);
    }
    return det;
}