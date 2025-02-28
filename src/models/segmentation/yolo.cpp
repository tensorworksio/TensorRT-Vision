#include <opencv2/dnn.hpp>
#include <utils/detection_utils.hpp>
#include <models/segmentation/yolo.hpp>

namespace seg
{

    bool Yolo::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
    {
        // These params will be used in the post-processing stage
        // FIXME: This 4 values assume that images in a batch have the same size
        // original_size must be stored in a datastructure
        // Plan:
        // preprocess return a Frame with original size stored + the processed image
        // postprocess consumes network output + Frame
        // postprocess returns a Frame

        m_imgHeight = static_cast<float>(srcImg.rows);
        m_imgWidth = static_cast<float>(srcImg.cols);
        m_ratioHeight = m_imgHeight / static_cast<float>(size.height);
        m_ratioWidth = m_imgWidth / static_cast<float>(size.width);

        // The model expects RGB input
        cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);

        // Resize the model to the expected size and pad with background
        cv::resize(dstImg, dstImg, size, 0, 0, cv::INTER_LINEAR);

        // Convert to Float32
        dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);

        return !dstImg.empty();
    }

    std::vector<Detection> Yolo::postprocess(const trt::MultiOutput &engineOutputs)
    {
        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 2);

        auto numChannels = outputDims[0].d[1];
        auto numAnchors = outputDims[0].d[2];

        auto numMasks = outputDims[1].d[1];
        auto maskHeight = outputDims[1].d[2];
        auto maskWidth = outputDims[1].d[3];

        auto numClasses = numChannels - numMasks - 4; // 4 bbox

        std::vector<cv::Rect> bboxes;
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

            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr++;

            float x0 = std::clamp((x - 0.5f * w) * m_ratioWidth, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratioHeight, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratioWidth, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratioHeight, 0.f, m_imgHeight);

            cv::Mat maskWeight = cv::Mat(1, numMasks, CV_32F, maskWeightsPtr);

            bboxes.emplace_back(cv::Rect2f(x0, y0, x1 - x0, y1 - y0));
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

        for (auto &idx : indices)
        {
            detections.emplace_back(Detection{class_ids[idx], scores[idx], bboxes[idx], getClassName(class_ids[idx])});
        }

        cv::Mat masks;
        std::vector<cv::Mat> selectedWeights;
        selectedWeights.reserve(indices.size());

        for (auto &idx : indices)
        {
            selectedWeights.push_back(maskWeights[idx]);
        }
        cv::vconcat(selectedWeights, masks);

        if (!masks.empty())
        {
            cv::Mat maskMap1d = (masks * output1).t();
            cv::Mat maskMap2d = maskMap1d.reshape(indices.size(), {static_cast<int>(maskWidth), static_cast<int>(maskHeight)});

            std::vector<cv::Mat> maskChannels;
            cv::split(maskMap2d, maskChannels);

            cv::Rect roi;
            roi = cv::Rect(0, 0, maskWidth, maskHeight);

            for (size_t i = 0; i < indices.size(); i++)
            {
                cv::Mat dest, mask;
                cv::exp(-maskChannels[i], dest);
                dest = 1.0 / (1.0 + dest);
                dest = dest(roi);
                cv::resize(dest, mask, cv::Size(static_cast<int>(m_imgWidth), static_cast<int>(m_imgHeight)), cv::INTER_LINEAR);

                cv::Rect rect = cv::Rect(static_cast<int>(detections[i].bbox.x), static_cast<int>(detections[i].bbox.y), static_cast<int>(detections[i].bbox.width), static_cast<int>(detections[i].bbox.height));
                detections[i].mask = mask(rect) > config.maskThreshold;
            }
        }

        return detections;
    }

}