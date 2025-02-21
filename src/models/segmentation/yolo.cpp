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
        dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);

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

        cv::Mat output0 = cv::Mat(numChannels, numAnchors, CV_32F, const_cast<float *>(engineOutputs[0].data())).t();
        cv::Mat output1 = cv::Mat(numMasks, maskHeight * maskWidth, CV_32F, const_cast<float *>(engineOutputs[1].data()));

        for (auto i = 0; i < numAnchors; i++)
        {
            auto rowPtr = output0.row(i).ptr<float>();
            auto bboxesPtr = rowPtr;
            auto scoresPtr = rowPtr + 4;
            auto masksPtr = scoresPtr + numClasses;

            auto maxClsPtr = std::max_element(scoresPtr, masksPtr);
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

            bboxes.emplace_back(cv::Rect2f(x0, y0, x1 - x0, y1 - y0));
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
            int x0 = static_cast<int>(bboxes[idx].x);
            int y0 = static_cast<int>(bboxes[idx].y);
            int x1 = x0 + static_cast<int>(bboxes[idx].width);
            int y1 = y0 + static_cast<int>(bboxes[idx].height);

            int x0_scaled = static_cast<int>((bboxes[idx].x / m_imgWidth) * maskWidth);
            int y0_scaled = static_cast<int>((bboxes[idx].y / m_imgHeight) * maskHeight);
            int x1_scaled = static_cast<int>(((bboxes[idx].x + bboxes[idx].width) / m_imgWidth) * maskWidth);
            int y1_scaled = static_cast<int>(((bboxes[idx].y + bboxes[idx].height) / m_imgHeight) * maskHeight);

            cv::Mat selectedWeights = output0(cv::Range(idx, idx + 1), cv::Range(output0.cols - numMasks, output0.cols)); // 1 × numMasks
            std::cout << "selectedWeights size: " << selectedWeights.size() << std::endl;
            std::cout << "output1 size: " << output1.size() << std::endl;
            cv::Mat maskWeights = cv::Mat(selectedWeights * output1).reshape(1, maskHeight); // maskHeight × maskWidth
            std::cout << "maskWeights size: " << maskWeights.size() << std::endl;

            cv::Mat maskScoreScaled;
            cv::exp(-maskWeights, maskScoreScaled);
            maskScoreScaled = 1.0 / (1.0 + maskScoreScaled);
            std::cout << "maskScoreScaled size: " << maskScoreScaled.size() << std::endl;

            cv::Mat maskScoreScaledCrop = maskScoreScaled(cv::Range(y0_scaled, y1_scaled), cv::Range(x0_scaled, x1_scaled));
            std::cout << "maskScoreScaledCrop size: " << maskScoreScaledCrop.size() << std::endl;

            cv::Mat maskScoreCrop;
            cv::resize(maskScoreScaledCrop, maskScoreCrop, cv::Size(x1 - x0, y1 - y0), cv::INTER_CUBIC);

            cv::Size blurSize = cv::Size(static_cast<int>(m_imgWidth) / maskWidth, static_cast<int>(m_imgHeight) / maskHeight);
            cv::blur(maskScoreCrop, maskScoreCrop, blurSize);

            cv::Mat maskCrop;
            cv::threshold(maskScoreCrop, maskCrop, 0.05f, 1.0f, cv::THRESH_BINARY_INV);

            cv::Mat mask;
            maskCrop.convertTo(mask, CV_8UC1, 255);
            std::cout << "mask size: " << mask.size() << std::endl;

            // Debug prints
            cv::Scalar mean1 = cv::mean(output1);
            cv::Scalar mean2 = cv::mean(selectedWeights);
            std::cout << "output1 mean: " << mean1[0] << std::endl;
            std::cout << "selectedWeights mean: " << mean2[0] << std::endl;

            double minVal, maxVal;
            cv::minMaxLoc(maskScoreScaledCrop, &minVal, &maxVal);
            std::cout << "maskScoreScaledCrop range: " << minVal << " to " << maxVal << std::endl;

            detections.emplace_back(Detection{class_ids[idx], scores[idx], bboxes[idx], getClassName(class_ids[idx]), mask});
        }

        return detections;
    }

}