#ifndef TIMER_H
#define TIMER_H

#include <chrono>

#include "util.h"

template <typename Period = std::milli, typename T = f64>
class Timer {
protected:
    std::chrono::time_point<std::chrono::system_clock> startTime;
    T totalTime;
    b8 running;

public:
    Timer(T totalTime) {
        this->totalTime = totalTime;
        running = false;
    }

    Timer() : Timer(0) {}

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

    Timer<> operator+(const Timer &other) const {
        if (running || other.running) { throw std::runtime_error("Timer is still running!"); }
        return Timer<>(totalTime + other.totalTime);
    }
};

template <typename Period = std::milli, typename T = f64>
void printTimerMinMaxAvg(Timer<Period, T> timers[], usize n) {
    T runtimes[n];
    for (usize i = 0; i < n; ++i)
        runtimes[i] = timers[i].getTotal();
    // TODO: Select the right unit based on the template variable!
    printArrayMinMaxAvg(runtimes, n, "ms");
}

#endif //TIMER_H
