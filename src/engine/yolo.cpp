#include <engine/yolo.hpp>
#include <opencv2/dnn.hpp>
#include <utils/detection_utils.hpp>

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

std::vector<Detection> Yolo::postprocess(const std::vector<float> &featureVector)
{
    const auto &outputDims = engine->getOutputDims();
    assert(outputDims.size() == 1);

    auto numChannels = outputDims[0].d[1];
    auto numAnchors = outputDims[0].d[2];
    auto numClasses = numChannels - 4;

    std::vector<cv::Rect> bboxes;
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

        float x = *bboxesPtr++;
        float y = *bboxesPtr++;
        float w = *bboxesPtr++;
        float h = *bboxesPtr;

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
        detections.emplace_back(Detection{
            class_ids[idx],
            scores[idx],
            bboxes[idx],
            getClassName(class_ids[idx])});
    }

    return detections;
}

std::vector<Detection> Yolov7::postprocess(const std::vector<float> &featureVector)
{
    const auto &outputDims = engine->getOutputDims();
    assert(outputDims.size() == 1);

    auto numAnchors = outputDims[0].d[1];
    auto numChannels = outputDims[0].d[2];
    auto numClasses = numChannels - 5;

    std::vector<cv::Rect> bboxes;
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

        float x = *bboxesPtr++;
        float y = *bboxesPtr++;
        float w = *bboxesPtr++;
        float h = *bboxesPtr;

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
        detections.emplace_back(Detection{
            class_ids[idx],
            scores[idx],
            bboxes[idx],
            getClassName(class_ids[idx])});
    }

    return detections;
}
