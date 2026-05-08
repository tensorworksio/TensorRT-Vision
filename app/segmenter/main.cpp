#include <string>
#include <signal.h>
#include <atomic>
#include <print>
#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>
#include <types/frame.hpp>
#include <models/segmentation/factory.hpp>

std::atomic<bool> running{true};

void signalHandler([[maybe_unused]] int signum)
{
    running = false;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("segment");
    program.add_argument("-i", "--input").required().help("Input video file or camera index (0,1,...)");
    program.add_argument("-c", "--config").required().help("Path to model config.json");
    program.add_argument("-o", "--output").help("Output video file");
    program.add_argument("-d", "--display").flag().help("Display video frames");

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
    auto inputPath = program.get<std::string>("--input");
    cv::VideoCapture cap;
    if (inputPath.size() == 1 && std::isdigit(inputPath[0]))
        cap.open(std::stoi(inputPath));
    else
        cap.open(inputPath);

    if (!cap.isOpened())
    {
        std::println(stderr, "Error: Could not open video source {}", inputPath);
        return 1;
    }

    // Load model
    auto model = seg::SegmenterFactory::create(program.get<std::string>("--config"));

    // Output
    cv::VideoWriter writer;
    if (auto outputPath = program.present<std::string>("--output"))
    {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(cap.get(cv::CAP_PROP_FRAME_WIDTH), cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        writer.open(*outputPath, fourcc, fps, frameSize);
        if (!writer.isOpened())
        {
            std::println(stderr, "Error: Could not create output video {}", *outputPath);
            return 1;
        }
    }

    bool display = program.get<bool>("--display") || !program.is_used("--output");
    if (display)
        cv::namedWindow("Segmentations", cv::WINDOW_AUTOSIZE);

    Frame frame;
    signal(SIGINT, signalHandler);

    while (running)
    {
        cap >> frame;
        if (frame.empty())
            break;

        auto detections = model->process(frame.image);
        cv::Mat output = frame.draw(detections);

        if (display)
            cv::imshow("Segmentations", output);

        if (writer.isOpened())
            writer.write(output);

        if (cv::waitKey(1) == 27)
            running = false;
    }

    if (cap.isOpened())
        cap.release();

    if (writer.isOpened())
        writer.release();

    if (display)
        cv::destroyAllWindows();

    return 0;
}
