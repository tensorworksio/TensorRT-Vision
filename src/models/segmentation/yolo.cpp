#include <opencv2/dnn.hpp>
#include <utils/detection_utils.hpp>
#include <models/segmentation/yolo.hpp>

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

    std::vector<cv::Mat> masks;
    masks.reserve(numAnchors);

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
        auto float score = *maxClsPtr;

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

        int x0_scaled = static_cast<int>((x0 / m_imgWidth) * maskWidth);
        int y0_scaled = static_cast<int>((y0 / m_imgHeight) * maskHeight);
        int x1_scaled = static_cast<int>((x1 / m_imgWidth) * maskWidth);
        int y1_scaled = static_cast<int>((y1 / m_imgHeight) * maskHeight);

        cv::Mat mask = cv::Mat::zeros(m_imgHeight, m_imgWidth, CV_8UC1);

        std::vector<float> maskWeights(maskHeight * maskWidth, 0);
        for (auto j = 0; j < maskHeight * maskWidth; j++)
        {
            for (auto k = 0; k < numMasks; k++)
            {
                maskWeights[j] += masksPtr[k] * output1.at<float>(k, j);
            }
        }
        cv::Mat maskScoreScaled = cv::Mat(maskHeight, maskWidth, CV_32F, sigmoid(maskWeights).data());
        cv::Mat maskScoreScaledCrop = maskScore(cv::Range(y0_mask, y1_mask), cv::Range(x0_mask, x1_mask));
        cv::Mat maskScoreCrop;
        cv::resize(maskScoreScaledCrop, maskScoreCrop, cv::Size(x1 - x0, y1 - y0), cv::INTER_CUBIC);

        cv::Size blurSize = cv::Size(static_cast<int>(m_imgWidth) / maskWidth, static_cast<int>(m_imgHeight) / maskHeight);
        cv::blur(maskScoreCrop, maskScoreCrop, blurSize);

        cv::Mat maskCrop;
        cv::threshold(maskScoreCrop, maskCrop, 0.5f, 1, cv::THRESH_BINARY);
        mask(cv::Range(y0, y1), cv::Range(x0, x1)) = maskCrop;

        bboxes.emplace_back(cv::Rect2f(x0, y0, x1 - x0, y1 - y0));
        class_ids.emplace_back(class_id);
        scores.emplace_back(score);
        masks.emplace_back(mask);
    }
}