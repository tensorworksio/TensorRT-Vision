#include <string>
#include <signal.h>
#include <atomic>
#include <print>
#include <argparse/argparse.hpp>
#include <opencv2/opencv.hpp>
#include <types/frame.hpp>
#include <utils/draw_utils.hpp>
#include <tracking/factory.hpp>
#include <tasks/reid.hpp>
#include <tasks/detection.hpp>
#include <tasks/segmentation.hpp>

std::atomic<bool> running{true};

void signalHandler([[maybe_unused]] int signum)
{
    running = false;
}

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("mot");
    program.add_argument("-i", "--input").required().help("Input video file or camera index (0,1,...)");
    program.add_argument("--tracker").required().help("Path to tracker config TOML");
    program.add_argument("--detector").help("Path to detector config TOML");
    program.add_argument("--segmenter").help("Path to segmenter config TOML");
    program.add_argument("--reid").help("Path to ReID config TOML");
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

    bool hasDetector = program.is_used("--detector");
    bool hasSegmenter = program.is_used("--segmenter");
    if (!hasDetector && !hasSegmenter)
    {
        std::println(stderr, "Error: either --detector or --segmenter must be provided");
        return 1;
    }
    if (hasDetector && hasSegmenter)
    {
        std::println(stderr, "Error: --detector and --segmenter are mutually exclusive");
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

    // Load tracker
    auto tracker = TrackerFactory::create(program.get<std::string>("--tracker"));

    // Load reid (optional)
    std::unique_ptr<reid::ReId> reidModel = nullptr;
    if (auto reidPath = program.present<std::string>("--reid"))
    {
        auto reidConfig = reid::ReIdConfig::load(*reidPath);
        reidModel = std::make_unique<reid::ReId>(reidConfig);
    }

    // Load detector or segmenter
    std::unique_ptr<trt::DetectionProcessor> detector = nullptr;
    if (hasSegmenter)
        detector = seg::SegmenterFactory::create(program.get<std::string>("--segmenter"));
    else
        detector = det::DetectorFactory::create(program.get<std::string>("--detector"));

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
        cv::namedWindow("Multi Object Tracking", cv::WINDOW_AUTOSIZE);

    Frame frame;
    signal(SIGINT, signalHandler);

    while (running)
    {
        cap >> frame;
        if (frame.empty())
            break;

        auto detections = detector->process(frame.image);

        if (reidModel)
        {
            for (auto &det : detections)
            {
                cv::Mat roi = frame(det.bbox);
                det.features = reidModel->process(roi);
            }
        }

        tracker->update(detections);
        cv::Mat output = drawDetections(frame, detections, true, true);

        if (display)
            cv::imshow("Multi Object Tracking", output);

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
