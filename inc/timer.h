#ifndef TIMER_H
#define TIMER_H

#include <chrono>

template <typename Period = std::milli, typename T = f64>
class Timer {
private:
    std::chrono::time_point<std::chrono::system_clock> startTime;
    T totalTime;
    b8 running;

public:
    Timer() {
        clear();
    }

    void start() {
        if (running) { throw std::runtime_error("Timer is already running!"); }
        startTime = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void pause() {
        if (!running) { throw std::runtime_error("Timer is not yet running!"); }
        const std::chrono::time_point<std::chrono::system_clock> pauseTime = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<T, Period> duration = pauseTime - startTime;
        totalTime += duration.count();
        running = false;
    }

    T getTotal() {
        if (running) { throw std::runtime_error("Timer is still running!"); }
        return totalTime;
    }

    void clear() {
        totalTime = 0;
        running = false;
    }
};

#endif //TIMER_H
