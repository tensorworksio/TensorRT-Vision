#include <string>
#include <fstream>
#include <print>
#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>
#include <rfl/json.hpp>
#include <utils/geometry_utils.hpp>
#include <tasks/reid.hpp>

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("reid");
    program.add_argument("-q", "--query").required().help("Query image to compare");
    program.add_argument("-k", "--key").required().help("Key image to compare against");
    program.add_argument("-c", "--config").required().help("Path to model config TOML");
    program.add_argument("-o", "--output").help("Output file");
    program.add_argument("-d", "--display").flag().help("Display images");

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

    // Input query
    auto queryPath = program.get<std::string>("--query");
    cv::Mat queryImage = cv::imread(queryPath, cv::IMREAD_COLOR);
    if (queryImage.empty())
    {
        std::println(stderr, "Error: Could not load query image {}", queryPath);
        return 1;
    }

    // Input key
    auto keyPath = program.get<std::string>("--key");
    cv::Mat keyImage = cv::imread(keyPath, cv::IMREAD_COLOR);
    if (keyImage.empty())
    {
        std::println(stderr, "Error: Could not load key image {}", keyPath);
        return 1;
    }

    // Config
    auto config = reid::ReIdConfig::load(program.get<std::string>("--config"));

    // Process images
    reid::ReId reid(config);
    auto featureVector1 = reid.process(queryImage);
    auto featureVector2 = reid.process(keyImage);

    // Output
    float similarity = cosineSimilarity(featureVector1, featureVector2);
    bool match = similarity > reid.getConfig().confidence_threshold;

    struct OutputData { bool match; float similarity; };
    struct Output { std::string status; OutputData data; };
    auto output = rfl::json::write(Output{"success", {match, similarity}});

    if (auto outputPath = program.present<std::string>("--output"))
    {
        std::ofstream outFile(*outputPath);
        if (outFile.is_open())
        {
            outFile << output << "\n";
        }
        else
        {
            struct Error { std::string status; std::string message; };
            std::println(stderr, "{}", rfl::json::write(Error{"error", "Could not create output file"}));
            return 1;
        }
    }
    else
    {
        std::println("{}", output);
    }

    if (program.get<bool>("--display"))
    {
        int maxHeight = std::max(queryImage.rows, keyImage.rows);
        int totalWidth = queryImage.cols + keyImage.cols;

        cv::Mat canvas(maxHeight, totalWidth, CV_8UC3);
        cv::Mat leftROI(canvas, cv::Rect(0, 0, queryImage.cols, queryImage.rows));
        cv::Mat rightROI(canvas, cv::Rect(queryImage.cols, 0, keyImage.cols, keyImage.rows));

        queryImage.copyTo(leftROI);
        keyImage.copyTo(rightROI);

        cv::putText(canvas, "Query", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
        cv::putText(canvas, "Key", cv::Point(queryImage.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::namedWindow("ReID Comparison", cv::WINDOW_AUTOSIZE);
        cv::imshow("ReID Comparison", canvas);
        cv::waitKey(0);
    }

    return 0;
}
