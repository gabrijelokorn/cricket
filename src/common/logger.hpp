#pragma once

#include <iostream>
#include <mutex>

enum LogPriority
{
    TracePriority,
    DebugPriority,
    InfoPriority,
    WarnPriority,
    ErrorPriority,
    CriticalPriority
};

class Logger
{
private:
    static LogPriority priority;
    static std::mutex log_mutex;

public:
    static void set_priority(LogPriority new_priority)
    {
        priority = new_priority;
    }

    template <typename... Args>
    static void Trace(const char *message, Args... args)
    {
        if (priority <= TracePriority)
        {
            std::scoped_lock(log_mutex);
            printf("[Trace]\t");
            if constexpr (sizeof...(Args) == 0)
                printf("%s", message);
            else
                printf(message, args...);
            printf("\n");
        }
    }

    template <typename... Args>
    static void Debug(const char *message, Args... args)
    {
        if (priority <= DebugPriority)
        {
            std::scoped_lock(log_mutex);
            printf("[Debug]\t");
            if constexpr (sizeof...(Args) == 0)
                printf("%s", message);
            else
                printf(message, args...);
            printf("\n");
        }
    }

    template <typename... Args>
    static void Info(const char *message, Args... args)
    {
        if (priority <= InfoPriority)
        {
            std::scoped_lock(log_mutex);
            printf("[Info]\t");
            if constexpr (sizeof...(Args) == 0)
                printf("%s", message);
            else
                printf(message, args...);
            printf("\n");
        }
    }

    template <typename... Args>
    static void Warn(const char *message, Args... args)
    {
        if (priority <= WarnPriority)
        {
            std::scoped_lock(log_mutex);
            printf("[Warn]\t");
            if constexpr (sizeof...(Args) == 0)
                printf("%s", message);
            else
                printf(message, args...);
            printf("\n");
        }
    }

    template <typename... Args>
    static void Error(const char *message, Args... args)
    {
        if (priority <= ErrorPriority)
        {
            std::scoped_lock(log_mutex);
            printf("[Error]\t");
            if constexpr (sizeof...(Args) == 0)
                printf("%s", message);
            else
                printf(message, args...);
            printf("\n");
        }
    }

    template <typename... Args>
    static void Critical(const char *message, Args... args)
    {
        if (priority <= CriticalPriority)
        {
            std::scoped_lock(log_mutex);
            printf("[Critical]\t");
            if constexpr (sizeof...(Args) == 0)
                printf("%s", message);
            else
                printf(message, args...);
            printf("\n");
        }
    }
};

inline LogPriority Logger::priority = InfoPriority;
