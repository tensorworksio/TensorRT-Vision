#include <string>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>

#include <engine/engine.hpp>
#include <common/detection_utils.hpp>

namespace po = boost::program_options;

bool preprocess(const cv::Mat &srcImg, cv::Mat &dstImg, cv::Size size)
{
    // The model expects RGB input
    cv::cvtColor(srcImg, dstImg, cv::COLOR_BGR2RGB);

    // Resize the model to the expected size and pad with background
    dstImg = letterbox(dstImg, size, cv::Scalar(114, 114, 114), false, true, false, 32);

    // Convert to Float32
    dstImg.convertTo(dstImg, CV_32FC3, 1.f / 255.f);

    return !dstImg.empty();
}

bool postprocess(std::vector<float> &featureVector, Detection &det)
{
    // Find the iterator to the maximum element in the feature vector
    auto maxElementIter = std::max_element(featureVector.begin(), featureVector.end());

    // Calculate the index of the maximum element
    int maxElementIndex = std::distance(featureVector.begin(), maxElementIter);

    // Modify the Detection object accordingly
    det.class_id = maxElementIndex;
    det.probability = *maxElementIter;

    return true;
}

int main(int argc, char *argv[])
{

    std::string imagePath;
    std::string engineModelPath;

    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")
                      ("image", po::value<std::string>(&imagePath)->required(), "path to the image file")
                      ("model", po::value<std::string>(&engineModelPath)->required(), "path to the model file");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help"))
        {
            std::cout << desc << "\n";
            return 1;
        }

        po::notify(vm);
    }
    catch (po::error &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << desc << "\n";
        return 1;
    }

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image " << imagePath << std::endl;
        return 1;
    }

    trt::NvLogger logger;
    trt::EngineOptions options;
    trt::Engine engine = trt::Engine(options, logger);

    bool success;

    success = engine.loadNetwork(engineModelPath);
    if (!success)
    {
        throw std::runtime_error("Unable to load TRT engine.");
    }

    cv::Mat processedImage;
    const auto inputDims = engine.getInputDims();
    cv::Size imageSize(inputDims[0].d[1], inputDims[0].d[2]);
    success = preprocess(image, processedImage, imageSize);
    if (!success)
    {
        throw std::runtime_error("Preprocessing failed.");
    }

    std::vector<float> featureVector;
    success = engine.runInference(processedImage, featureVector);
    if (!success)
    {
        throw std::runtime_error("Inference failed.");
    }

    Detection detection;
    success = postprocess(featureVector, detection);
    if (!success)
    {
        throw std::runtime_error("Postprocessing failed.");
    }

    std::cout << "Image class: " << detection.class_id << std::endl;

    return 0;
}