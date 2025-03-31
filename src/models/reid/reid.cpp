#include <models/reid/reid.hpp>
#include <utils/vector_utils.hpp>
#include <utils/detection_utils.hpp>

namespace reid
{

    bool ReId::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg)
    {
        const auto &inputDims = engine->getInputDims();
        assert(inputDims.size() == 1);

        cv::Size size(inputDims[0].d[2], inputDims[0].d[1]);

        cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);
        dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);
        dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);
        return !dstImg.empty();
    }

    std::vector<float> ReId::postprocess(const trt::SingleOutput &featureVector)
    {
        return vector_ops::normalize(featureVector);
    }

} // namespace reid