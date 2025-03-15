#include <utils/detection_utils.hpp>
#include <models/classification/classifier.hpp>

namespace cls
{
    bool BaseClassifier::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
    {
        cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);
        dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);
        dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);

        return !dstImg.empty();
    }

    Detection SingleLabelClassifier::postprocess(const trt::SingleOutput &featureVector)
    {
        Detection det;
        auto maxElement = std::max_element(featureVector.begin(), featureVector.end());
        if (*maxElement >= config.confidenceThreshold)
        {
            det.class_id = std::distance(featureVector.begin(), maxElement);
            det.confidence = *maxElement;
            det.class_name = getClassName(det.class_id);
            det.labels.push_back(det.class_id);
        }
        return det;
    }

    Detection MultiLabelClassifier::postprocess(const trt::SingleOutput &featureVector)
    {
        Detection det;
        float maxConfidence = 0.0f;

        for (size_t i = 0; i < featureVector.size(); ++i)
        {
            if (featureVector[i] < config.confidenceThreshold)
                continue;

            int class_id = static_cast<int>(i);
            det.labels.push_back(class_id);

            // Keep track of the highest confidence for the main detection fields
            if (featureVector[i] > maxConfidence)
            {
                maxConfidence = featureVector[i];
                det.class_id = class_id;
                det.class_name = getClassName(class_id);
                det.confidence = featureVector[i];
            }
        }

        return det;
    }
} // cls