#include <opencv2/dnn.hpp>
#include <utils/detection_utils.hpp>
#include <models/detection/yolo.hpp>

namespace det
{

    bool Yolo::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg)
    {
        const auto &inputDims = engine->getInputDims();
        assert(inputDims.size() == 1);

        cv::Size size(inputDims[0].d[2], inputDims[0].d[1]);

        cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);
        dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);
        dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);
        return !dstImg.empty();
    }

    std::vector<Detection> Yolo::postprocess(const trt::SingleOutput &featureVector)
    {
        const auto &inputDims = engine->getInputDims();
        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 1);

        cv::Size2f size(inputDims[0].d[2], inputDims[0].d[1]);

        auto numChannels = outputDims[0].d[1];
        auto numAnchors = outputDims[0].d[2];
        auto numClasses = numChannels - 4; // 4 bbox

        std::vector<cv::Rect2d> bboxes;
        bboxes.reserve(numAnchors);

        std::vector<float> scores;
        scores.reserve(numAnchors);

        std::vector<int> class_ids;
        class_ids.reserve(numAnchors);

        cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, const_cast<float *>(featureVector.data()));
        output = output.t();

        for (int i = 0; i < numAnchors; i++)
        {
            auto rowPtr = output.row(i).ptr<float>();
            auto bboxesPtr = rowPtr;
            auto scoresPtr = rowPtr + 4;
            auto maxClsPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
            float score = *maxClsPtr;
            int class_id = maxClsPtr - scoresPtr;

            if (score < config.confidenceThreshold)
            {
                continue;
            }

            float xn = *bboxesPtr++;
            float yn = *bboxesPtr++;
            float wn = *bboxesPtr++;
            float hn = *bboxesPtr++;

            float x = std::clamp((xn - 0.5f * wn) / size.width, 0.f, 1.f);
            float y = std::clamp((yn - 0.5f * hn) / size.height, 0.f, 1.f);
            float w = std::clamp(wn / size.width, 0.f, 1.f);
            float h = std::clamp(hn / size.height, 0.f, 1.f);

            bboxes.emplace_back(x, y, w, h);
            class_ids.emplace_back(class_id);
            scores.emplace_back(score);
        }

        // Non Maximum Suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, scores, config.confidenceThreshold, config.nmsThreshold, indices, config.nmsEta, config.topK);

        // Fill output detections
        std::vector<Detection> detections;
        detections.reserve(indices.size());

        for (auto &idx : indices)
        {
            detections.emplace_back(Detection{
                class_ids[idx],
                scores[idx],
                bboxes[idx],
                getClassName(class_ids[idx])});
        }

        return detections;
    }

    std::vector<Detection> Yolov7::postprocess(const trt::SingleOutput &featureVector)
    {
        const auto &inputDims = engine->getInputDims();
        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 1);

        cv::Size2f size(inputDims[0].d[2], inputDims[0].d[1]);

        auto numAnchors = outputDims[0].d[1];
        auto numChannels = outputDims[0].d[2];
        auto numClasses = numChannels - 5;

        std::vector<cv::Rect2d> bboxes;
        bboxes.reserve(numAnchors);

        std::vector<float> scores;
        scores.reserve(numAnchors);

        std::vector<int> class_ids;
        class_ids.reserve(numAnchors);

        cv::Mat output = cv::Mat(numAnchors, numChannels, CV_32F, const_cast<float *>(featureVector.data()));

        for (int i = 0; i < numAnchors; i++)
        {
            auto rowPtr = output.row(i).ptr<float>();
            auto bboxesPtr = rowPtr;
            auto objScorePtr = rowPtr + 4;
            auto clsScoresPtr = rowPtr + 5;
            auto maxClsPtr = std::max_element(clsScoresPtr, clsScoresPtr + numClasses);
            float score = (*maxClsPtr) * (*objScorePtr);
            int class_id = maxClsPtr - clsScoresPtr;

            if (score < config.confidenceThreshold)
            {
                continue;
            }

            float xn = *bboxesPtr++;
            float yn = *bboxesPtr++;
            float wn = *bboxesPtr++;
            float hn = *bboxesPtr++;

            float x = std::clamp((xn - 0.5f * wn) / size.width, 0.f, 1.f);
            float y = std::clamp((yn - 0.5f * hn) / size.height, 0.f, 1.f);
            float w = std::clamp(wn / size.width, 0.f, 1.f);
            float h = std::clamp(hn / size.height, 0.f, 1.f);

            bboxes.emplace_back(x, y, w, h);
            class_ids.emplace_back(class_id);
            scores.emplace_back(score);
        }

        // Non Maximum Suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, scores, config.confidenceThreshold, config.nmsThreshold, indices, config.nmsEta, config.topK);

        // Fill output detections
        std::vector<Detection> detections;
        detections.reserve(indices.size());

        for (auto &idx : indices)
        {
            detections.emplace_back(Detection{
                class_ids[idx],
                scores[idx],
                bboxes[idx],
                getClassName(class_ids[idx])});
        }

        return detections;
    }
} // det