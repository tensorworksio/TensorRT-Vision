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

        cv::resize(dstImg, dstImg, size, 0, 0, cv::INTER_LINEAR);
        dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);
        cv::subtract(dstImg, cv::Scalar(0.485, 0.456, 0.406), dstImg);
        cv::divide(dstImg, cv::Scalar(0.229, 0.224, 0.225), dstImg);

        return !dstImg.empty();
    }

    bool PPOCRV3Recognizer::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg)
    {
        const auto &inputDims = engine->getInputDims();
        assert(inputDims.size() == 1);

        // Check if width needs to be capped at max width
        if (srcImg.cols * 1.0 / srcImg.rows * inputDims[0].d[1] >= inputDims[0].d[2])
        {
            // Direct resize to max dimensions
            cv::resize(srcImg, dstImg, cv::Size(inputDims[0].d[2], inputDims[0].d[1]));
        }
        else
        {
            // Resize maintaining aspect ratio
            int target_width = int(srcImg.cols * 1.0 / srcImg.rows * inputDims[0].d[1] + 1);
            cv::resize(srcImg, dstImg, cv::Size(target_width, inputDims[0].d[1]), 0.f, 0.f, cv::INTER_LINEAR);

            // Pad the right side with zeros
            int padding = inputDims[0].d[2] - target_width;
            cv::copyMakeBorder(dstImg, dstImg, 0, 0, 0, padding, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        }

        // Normalize using their specific parameters
        dstImg.convertTo(dstImg, CV_32FC3, 1.0 / 255.0);
        std::vector<cv::Mat> channels(3);
        cv::split(dstImg, channels);

        // Apply mean and scale to each channel
        for (size_t i = 0; i < channels.size(); i++)
        {
            channels[i].convertTo(channels[i], CV_32FC1, 1.0 * (1 / 0.5), (0.0 - 0.5) * (1 / 0.5));
        }
        cv::merge(channels, dstImg);

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

        cv::Mat output = cv::Mat(outputDims[0].d[2], outputDims[0].d[3], CV_32F, const_cast<float *>(engineOutputs.data()));

        cv::Mat binaryMask;
        cv::threshold(output, binaryMask, config.maskThreshold, 1.0, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8U); // Convert to 8-bit unsigned

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binaryMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto &contour : contours)
        {
            cv::Rect roi = cv::boundingRect(contour);
            if (roi.area() < config.minArea)
                continue;

            cv::Rect2f bbox(static_cast<float>(roi.x) / size.width,
                            static_cast<float>(roi.y) / size.height,
                            static_cast<float>(roi.width) / size.width,
                            static_cast<float>(roi.height) / size.height);

            detections.emplace_back(Detection{-1, 1.0f, bbox, "text", output(roi)});
        }

        return detections;
    }

    std::string PPOCRV3Recognizer::postprocess(const trt::SingleOutput &featureVector)
    {
        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 1);

        auto seqLen = outputDims[0].d[1];
        auto numClasses = outputDims[0].d[2];
        cv::Mat output = cv::Mat(seqLen, numClasses, CV_32F, const_cast<float *>(featureVector.data()));

        std::string result;
        int lastIndex = -1;

        for (auto t = 0; t < seqLen; ++t)
        {
            auto rowPtr = output.row(t).ptr<float>();
            auto maxIndex = std::max_element(rowPtr, rowPtr + numClasses) - rowPtr;

            // CTC decoding
            if (maxIndex != 0 && maxIndex != lastIndex)
                result += config.vocabulary[maxIndex];

            lastIndex = maxIndex;
        }

        return result;
    }

} // namespace ocr