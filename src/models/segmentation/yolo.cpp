#include <opencv2/dnn.hpp>
#include <utils/detection_utils.hpp>
#include <models/segmentation/yolo.hpp>

namespace seg
{

    bool Yolo::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
    {
        cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);
        cv::resize(dstImg, dstImg, size, 0, 0, cv::INTER_LINEAR);
        dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);
        return !dstImg.empty();
    }

    std::vector<Detection> Yolo::postprocess(const trt::MultiOutput &engineOutputs)
    {
        const auto &inputDims = engine->getInputDims();
        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 2);

        cv::Size2f size(inputDims[0].d[2], inputDims[0].d[1]);

        auto numChannels = outputDims[0].d[1];
        auto numAnchors = outputDims[0].d[2];

        auto numMasks = outputDims[1].d[1];
        auto maskHeight = outputDims[1].d[2];
        auto maskWidth = outputDims[1].d[3];

        auto numClasses = numChannels - numMasks - 4; // 4 bbox

        std::vector<cv::Rect2d> bboxes;
        bboxes.reserve(numAnchors);

        std::vector<float> scores;
        scores.reserve(numAnchors);

        std::vector<int> class_ids;
        class_ids.reserve(numAnchors);

        std::vector<cv::Mat> maskWeights;
        maskWeights.reserve(numAnchors);

        cv::Mat output0 = cv::Mat(numChannels, numAnchors, CV_32F, const_cast<float *>(engineOutputs[0].data())).t();
        cv::Mat output1 = cv::Mat(numMasks, maskHeight * maskWidth, CV_32F, const_cast<float *>(engineOutputs[1].data()));

        for (auto i = 0; i < numAnchors; i++)
        {
            auto rowPtr = output0.row(i).ptr<float>();
            auto bboxesPtr = rowPtr;
            auto scoresPtr = rowPtr + 4;
            auto maskWeightsPtr = scoresPtr + numClasses;

            auto maxClsPtr = std::max_element(scoresPtr, maskWeightsPtr);
            float score = *maxClsPtr;
            int class_id = maxClsPtr - scoresPtr;

            if (score < config.confidenceThreshold)
                continue;

            float xn = *bboxesPtr++;
            float yn = *bboxesPtr++;
            float wn = *bboxesPtr++;
            float hn = *bboxesPtr++;

            float x = std::clamp((xn - 0.5f * wn) / size.width, 0.f, 1.f);
            float y = std::clamp((yn - 0.5f * hn) / size.height, 0.f, 1.f);
            float w = std::clamp(wn / size.width, 0.f, 1.f);
            float h = std::clamp(hn / size.height, 0.f, 1.f);

            cv::Mat maskWeight = cv::Mat(1, numMasks, CV_32F, maskWeightsPtr);

            bboxes.emplace_back(x, y, w, h);
            class_ids.emplace_back(class_id);
            scores.emplace_back(score);
            maskWeights.emplace_back(maskWeight);
        }

        // Non Maximum Suppression
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, scores, config.confidenceThreshold, config.nmsThreshold, indices, config.nmsEta, config.topK);

        // Fill output detections
        std::vector<Detection> detections;
        detections.reserve(indices.size());

        std::vector<cv::Mat> maskWeightsToKeep;
        maskWeightsToKeep.reserve(indices.size());

        for (auto &idx : indices)
        {
            maskWeightsToKeep.push_back(maskWeights[idx]);
            detections.emplace_back(Detection{class_ids[idx], scores[idx], bboxes[idx], getClassName(class_ids[idx])});
        }

        // Process masks
        if (!maskWeightsToKeep.empty())
        {
            cv::Mat masks;
            cv::vconcat(maskWeightsToKeep, masks);
            cv::Mat maskWeightMap = (masks * output1).t();

            cv::Mat maskScoreMap;
            cv::exp(-maskWeightMap, maskScoreMap);
            maskScoreMap = 1.0 / (1.0 + maskScoreMap);
            maskScoreMap = maskScoreMap.reshape(indices.size(), {static_cast<int>(maskWidth), static_cast<int>(maskHeight)});

            std::vector<cv::Mat> maskChannels;
            cv::split(maskScoreMap, maskChannels);
            float maskScaleX = static_cast<float>(maskWidth);
            float maskScaleY = static_cast<float>(maskHeight);

            // Process each mask
            for (size_t i = 0; i < indices.size(); i++)
            {
                cv::Rect roi(
                    static_cast<int>(detections[i].bbox.x * maskScaleX),
                    static_cast<int>(detections[i].bbox.y * maskScaleY),
                    static_cast<int>(detections[i].bbox.width * maskScaleX),
                    static_cast<int>(detections[i].bbox.height * maskScaleY));

                // Ensure ROI stays within mask bounds
                roi &= cv::Rect(0, 0, maskWidth, maskHeight);
                detections[i].mask = maskChannels[i](roi);
            }
        }

        return detections;
    }

}