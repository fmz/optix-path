// Repurposed from ggml.c

#pragma once

#include <time.h>
#include <string>

#if defined(_MSC_VER) || defined(__MINGW32__)
static int64_t timer_freq, timer_start;
static void timer_init(void) {
    LARGE_INTEGER t;
    QueryPerformanceFrequency(&t);
    timer_freq = t.QuadPart;

    // The multiplication by 1000 or 1000000 below can cause an overflow if timer_freq
    // and the uptime is high enough.
    // We subtract the program start time to reduce the likelihood of that happening.
    QueryPerformanceCounter(&t);
    timer_start = t.QuadPart;
}
static int64_t time_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000) / timer_freq;
}
static int64_t time_us(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return ((t.QuadPart-timer_start) * 1000000) / timer_freq;
}
#else
static void time_init(void) {}
static int64_t time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000 + (int64_t)ts.tv_nsec/1000000;
}

static int64_t time_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec*1000000 + (int64_t)ts.tv_nsec/1000;
}
#endif

static int64_t cycles(void) {
    return clock();
}

static int64_t cycles_per_ms(void) {
    return CLOCKS_PER_SEC/1000;
}

static bool timer_initialized = false;

struct mytimer {
    int64_t t0;
    int64_t t1;
    int64_t last;

    mytimer() {
    if (!timer_initialized) {
        time_init();
        } 
        t0 =  time_us();
    }
    
    void checkpoint(void) {
        t1 = time_us();
        last = t1 - t0;
        t0 = t1;
    }

    std::string to_string(void) {
        double t = double(last) / 1000.0;
        std::string t_str = " msec";
        if (t > 1000.0) {
            t /= 1000.0;
            t_str = " sec";
        }
        return std::to_string(t) + t_str;
    }
};
