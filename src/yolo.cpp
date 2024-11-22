#include <engine/yolo.hpp>

namespace trt
{

    const std::string Yolo::getClassName(int class_id) const
    {
        if (config.classNames.empty())
        {
            return std::to_string(class_id);
        }
        return config.classNames[class_id];
    }

    bool Yolo::preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
    {
        // These params will be used in the post-processing stage
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

    bool Yolov7::postprocess(std::vector<float> &featureVector, std::vector<Detection> &detections)
    {
        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<int> indices;

        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 1);

        auto numAnchors = outputDims[0].d[1];
        auto numChannels = outputDims[0].d[2];
        auto numClasses = numChannels - 5;

        cv::Mat output = cv::Mat(numAnchors, numChannels, CV_32F, featureVector.data());

        for (int i = 0; i < numAnchors; i++)
        {
            auto rowPtr = output.row(i).ptr<float>();
            auto bboxesPtr = rowPtr;
            auto objScorePtr = rowPtr + 4;
            auto clsScoresPtr = rowPtr + 5;
            auto maxClsPtr = std::max_element(clsScoresPtr, clsScoresPtr + numClasses);
            float score = (*maxClsPtr) * (*objScorePtr);

            if (score < config.probabilityThreshold)
            {
                continue;
            }

            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratioWidth, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratioHeight, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratioWidth, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratioHeight, 0.f, m_imgHeight);

            int class_id = maxClsPtr - clsScoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            class_ids.push_back(class_id);
            scores.push_back(score);
        }

        // Non Maximum Suppression
        cv::dnn::NMSBoxes(bboxes, scores, config.probabilityThreshold, config.nmsThreshold, indices, config.nmsEta, config.topK);

        for (auto &idx : indices)
        {
            Detection det{};
            det.probability = scores[idx];
            det.class_id = class_ids[idx];
            det.class_name = getClassName(det.class_id);
            det.bbox = bboxes[idx];
            detections.push_back(det);
        }

        return true;
    }

    bool Yolov8::postprocess(std::vector<float> &featureVector, std::vector<Detection> &detections)
    {
        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<int> indices;

        const auto &outputDims = engine->getOutputDims();
        assert(outputDims.size() == 1);

        auto numChannels = outputDims[0].d[1];
        auto numAnchors = outputDims[0].d[2];
        auto numClasses = numChannels - 4;

        cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
        output = output.t();

        for (int i = 0; i < numAnchors; i++)
        {
            auto rowPtr = output.row(i).ptr<float>();
            auto bboxesPtr = rowPtr;
            auto scoresPtr = rowPtr + 4;
            auto maxClsPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
            float score = *maxClsPtr;

            if (score < config.probabilityThreshold)
            {
                continue;
            }

            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = std::clamp((x - 0.5f * w) * m_ratioWidth, 0.f, m_imgWidth);
            float y0 = std::clamp((y - 0.5f * h) * m_ratioHeight, 0.f, m_imgHeight);
            float x1 = std::clamp((x + 0.5f * w) * m_ratioWidth, 0.f, m_imgWidth);
            float y1 = std::clamp((y + 0.5f * h) * m_ratioHeight, 0.f, m_imgHeight);

            int class_id = maxClsPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            class_ids.push_back(class_id);
            scores.push_back(score);
        }

        // Non Maximum Suppression
        cv::dnn::NMSBoxes(bboxes, scores, config.probabilityThreshold, config.nmsThreshold, indices, config.nmsEta, config.topK);

        for (auto &idx : indices)
        {
            Detection det{};
            det.probability = scores[idx];
            det.class_id = class_ids[idx];
            det.class_name = getClassName(det.class_id);
            det.bbox = bboxes[idx];
            detections.push_back(det);
        }

        return true;
    }

} // namespace trt