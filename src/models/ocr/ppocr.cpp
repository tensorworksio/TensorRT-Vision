#include <opencv2/dnn.hpp>
#include <utils/detection_utils.hpp>
#include <models/ocr/ppocr.hpp>

namespace ocr
{
    bool PPOCRV3Detector::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg)
    {
        const auto &inputDims = engine->getInputDims();
        assert(inputDims.size() == 1);

        // Store original size for postprocessing
        original_size_ = srcImg.size();
        cv::Size target_size(inputDims[0].d[2], inputDims[0].d[1]);

        // Convert to RGB
        cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);

        // Calculate letterbox parameters
        float scale = std::min(
            static_cast<float>(target_size.height) / srcImg.rows,
            static_cast<float>(target_size.width) / srcImg.cols);

        cv::Size unpadded_size(
            std::round(srcImg.cols * scale),
            std::round(srcImg.rows * scale));

        // Store padding info for postprocessing
        padding_.scale = scale;
        padding_.offset = cv::Point2f(
            (target_size.width - unpadded_size.width) / 2.0f,
            (target_size.height - unpadded_size.height) / 2.0f);

        // Resize maintaining aspect ratio
        cv::resize(dstImg, dstImg, unpadded_size, 0, 0, cv::INTER_LINEAR);

        // Add padding
        cv::Mat padded = cv::Mat(target_size, dstImg.type(), cv::Scalar(114, 114, 114));
        cv::Rect roi(
            padding_.offset.x, padding_.offset.y,
            dstImg.cols, dstImg.rows);
        dstImg.copyTo(padded(roi));
        dstImg = padded;

        // Normalize
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
        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 1);

        std::vector<Detection> detections;
        detections.reserve(config.topK);

        cv::Mat output = cv::Mat(outputDims[0].d[2], outputDims[0].d[3], CV_32F,
                                 const_cast<float *>(engineOutputs.data()));

        // Try different thresholding approaches
        cv::Mat binaryMask;
        cv::threshold(output, binaryMask, config.maskThreshold, 255, cv::THRESH_BINARY);
        binaryMask.convertTo(binaryMask, CV_8U);

        // Optional: Apply morphological operations to connect nearby text regions
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(binaryMask, binaryMask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        // Try RETR_LIST instead of RETR_EXTERNAL to catch all contours
        cv::findContours(binaryMask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        for (const auto &contour : contours)
        {
            // Use minAreaRect to better handle rotated text
            cv::RotatedRect rotatedRect = cv::minAreaRect(contour);
            cv::Rect roi = rotatedRect.boundingRect(); // Get the upright bounding rect

            // Filter small noise but with a smaller threshold
            if (roi.area() < config.minArea)
                continue;

            // Ensure ROI stays within bounds
            roi &= cv::Rect(0, 0, output.cols, output.rows);

            // Remove padding and scale back to original image coordinates
            float x = (roi.x - padding_.offset.x) / padding_.scale;
            float y = (roi.y - padding_.offset.y) / padding_.scale;
            float w = roi.width / padding_.scale;
            float h = roi.height / padding_.scale;

            // Normalize to [0,1] range
            x /= original_size_.width;
            y /= original_size_.height;
            w /= original_size_.width;
            h /= original_size_.height;

            // Ensure normalized coordinates are within [0,1]
            x = std::clamp(x, 0.0f, 1.0f);
            y = std::clamp(y, 0.0f, 1.0f);
            w = std::clamp(w, 0.0f, 1.0f - x);
            h = std::clamp(h, 0.0f, 1.0f - y);

            cv::Rect2f bbox(x, y, w, h);

            // Calculate confidence based on average mask value in ROI
            cv::Mat roiMask = output(roi);
            float confidence = static_cast<float>(cv::mean(roiMask)[0]);

            detections.emplace_back(Detection{-1, confidence, bbox, "text", output(roi)});
        }

        // Sort by confidence if needed
        if (!detections.empty())
        {
            std::sort(detections.begin(), detections.end(),
                      [](const Detection &a, const Detection &b)
                      {
                          return a.confidence > b.confidence;
                      });
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