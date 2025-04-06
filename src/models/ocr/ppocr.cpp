#include <opencv2/dnn.hpp>
#include <utils/detection_utils.hpp>
#include <models/ocr/ppocr.hpp>

namespace ocr
{
    bool PPOCRV3Detector::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg)
    {
        const auto &inputDims = engine->getInputDims();
        assert(inputDims.size() == 1);

        cv::Size size(inputDims[0].d[2], inputDims[0].d[1]);

        cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);

        // Official PPOCRv3 preprocessing steps
        cv::resize(dstImg, dstImg, size, 0, 0, cv::INTER_LINEAR);
        dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);
        cv::subtract(dstImg, cv::Scalar(0.485, 0.456, 0.406), dstImg);
        cv::divide(dstImg, cv::Scalar(0.229, 0.224, 0.225), dstImg);

        return !dstImg.empty();
    }

    std::vector<Detection> PPOCRV3Detector::postprocess(const trt::SingleOutput &engineOutputs)
    {
        const auto &inputDims = engine->getInputDims();
        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 1);

        cv::Size2f size(inputDims[0].d[2], inputDims[0].d[1]);

        std::vector<Detection> detections;
        detections.reserve(config.topK);

        cv::Mat output = cv::Mat(outputDims[0].d[1], outputDims[0].d[2], CV_32F, const_cast<float *>(engineOutputs.data()));
        output = output.t();

        cv::Mat binaryMask;
        cv::threshold(output, binaryMask, config.maskThreshold, 255, cv::THRESH_BINARY);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto &contour : contours)
        {
            cv::Rect roi = cv::boundingRect(contour);
            cv::Rect2f bbox(static_cast<float>(roi.x) / size.width,
                            static_cast<float>(roi.y) / size.height,
                            static_cast<float>(roi.width) / size.width,
                            static_cast<float>(roi.height) / size.height);

            detections.emplace_back(Detection{-1, 1.0f, bbox, "text", binaryMask(roi)});
        }

        return detections;
    }
} // namespace ocr
