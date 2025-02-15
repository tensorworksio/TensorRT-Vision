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

    std::vector<std::array<float, numMasks>> maskWeights;
    maskWeights.reserve(numAnchors);

    std::vector<float> scores;
    scores.reserve(numAnchors);

    std::vector<int> class_ids;
    class_ids.reserve(numAnchors);

    cv::Mat output0 = cv::Mat(numChannels, numAnchors, CV_32F, const_cast<float *>(engineOutputs[0].data())).t();

    for (int i = 0; i < numAnchors; i++)
    {
        auto rowPtr = output0.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto masksPtr = scoresPtr + numClasses;

        auto maxClsPtr = std::max_element(scoresPtr, masksPtr);
        auto float score = *maxClsPtr;
    }
}