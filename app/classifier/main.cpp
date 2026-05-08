#include <string>
#include <fstream>
#include <print>
#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>
#include <types/detection.hpp>
#include <models/classification/classifier.hpp>

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("classify");
    program.add_argument("-i", "--input").required().help("Input image");
    program.add_argument("-c", "--config").required().help("Path to model config.json");
    program.add_argument("-d", "--display").flag().help("Display image with results");
    program.add_argument("-o", "--output").help("Output text file for results");

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::println(stderr, "{}", e.what());
        std::println(stderr, "{}", program.help().str());
        return 1;
    }

    // Input
    auto imagePath = program.get<std::string>("--input");
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::println(stderr, "Error: Could not load image {}", imagePath);
        return 1;
    }

    // Config
    auto config = cls::ClassifierConfig::load(program.get<std::string>("--config"));

    // Process image
    cls::SingleLabelClassifier classifier(config);
    Detection det = classifier.process(image);

    // Output
    nlohmann::json output = {
        {"status", "success"},
        {"data", {{"class_id", det.class_id}, {"class_name", det.class_name}, {"confidence", det.confidence}}}};

    if (auto outputPath = program.present<std::string>("--output"))
    {
        std::ofstream outFile(*outputPath);
        if (outFile.is_open())
        {
            outFile << output.dump() << "\n";
        }
        else
        {
            nlohmann::json error = {{"status", "error"}, {"message", "Could not create output file"}};
            std::println(stderr, "{}", error.dump());
            return 1;
        }
    }
    else
    {
        std::println("{}", output.dump());
    }

    if (program.get<bool>("--display"))
    {
        cv::putText(image,
                    det.class_name + " (" + std::to_string(det.confidence) + ")",
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    1.0,
                    cv::Scalar(0, 255, 0),
                    2);

        cv::namedWindow("Classification Result", cv::WINDOW_AUTOSIZE);
        cv::imshow("Classification Result", image);
        cv::waitKey(0);
    }

    return 0;
}
