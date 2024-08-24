#pragma once

#include <NvInfer.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace trt
{

    class NvLogger : public nvinfer1::ILogger
    {
    public:
        NvLogger(Severity log_level = Severity::kWARNING) : NvLogger(spdlog::stdout_color_mt("NvLogger"), log_level) {}

        NvLogger(std::shared_ptr<spdlog::logger> logger, Severity log_level = Severity::kWARNING)
            : level(log_level), m_logger(std::move(logger))
        {
            spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
        }

        void log(Severity severity, const char *msg) noexcept override
        {
            if (severity > level)
            {
                return;
            }
            switch (severity)
            {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                m_logger->error(msg);
                break;
            case Severity::kWARNING:
                m_logger->warn(msg);
                break;
            case Severity::kINFO:
                m_logger->info(msg);
                break;
            case Severity::kVERBOSE:
                m_logger->debug(msg);
                break;
            default:
                m_logger->trace(msg);
                break;
            }
        }

        template <typename... Args>
        void log(Severity severity, const char *fmt, const Args &...args) noexcept
        {
            if (severity > level)
            {
                return;
            }
            switch (severity)
            {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                m_logger->error(fmt, args...);
                break;
            case Severity::kWARNING:
                m_logger->warn(fmt, args...);
                break;
            case Severity::kINFO:
                m_logger->info(fmt, args...);
                break;
            case Severity::kVERBOSE:
                m_logger->debug(fmt, args...);
                break;
            default:
                m_logger->trace(fmt, args...);
                break;
            }
        }

    private:
        Severity level;
        std::shared_ptr<spdlog::logger> m_logger;
    };
} // namespace trt