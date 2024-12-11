#include <string>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <types/detection.hpp>
#include "yolo.hpp"

namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description desc("Allowed options");
    desc.add_options()("help,h", "produce help message")("input,i", po::value<std::string>()->required(), "Input video file or camera index (0,1,...)")("config,c", po::value<std::string>(), "Path to model config")("output,o", po::value<std::string>(), "Output video file (optional)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 1;
    }

    po::notify(vm);

    // Input setup
    std::string inputPath = vm["input"].as<std::string>();
    cv::VideoCapture cap;
    if (inputPath.size() == 1 && std::isdigit(inputPath[0]))
    {
        cap.open(std::stoi(inputPath));
    }
    else
    {
        cap.open(inputPath);
    }
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video source " << inputPath << std::endl;
        return 1;
    }

    // Config
    YoloConfig config;
    if (vm.count("config"))
    {
        std::string configPath = vm["config"].as<std::string>();
        config = YoloConfig::load(configPath);
    }

    // Load model
    auto detector = YoloFactory::create(config);

    // Output setup
    cv::VideoWriter writer;
    if (vm.count("output"))
    {
        std::string outputPath = vm["output"].as<std::string>();
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH),
                           cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(outputPath, fourcc, fps, frameSize);
        if (!writer.isOpened())
        {
            std::cerr << "Error: Could not create output video " << outputPath << std::endl;
            return 1;
        }
    }

    cv::namedWindow("Detections", cv::WINDOW_AUTOSIZE);
    cv::Mat frame;

    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        auto detections = detector->process(frame);

        // Draw detections
        for (const auto &det : detections)
        {
            cv::rectangle(frame, det.bbox, cv::Scalar(0, 255, 0), 2);
            cv::putText(frame, std::to_string(det.class_id),
                        cv::Point(det.bbox.x, det.bbox.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow("Detections", frame);

        if (writer.isOpened())
        {
            writer.write(frame);
        }

        // Break on 'q' press
        if (cv::waitKey(1) == 'q')
            break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    return 0;
}