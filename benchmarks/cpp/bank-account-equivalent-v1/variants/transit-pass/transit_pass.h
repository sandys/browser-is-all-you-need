#pragma once

#include <mutex>

namespace transit_pass {

class fare_pass {
public:
    fare_pass();
    void activate();
    void top_up(int amount);
    void charge(int amount);
    void suspend();
    int funds();

private:
    enum class status { inactive, active };

    int amount_{0};
    status status_{status::inactive};
    std::mutex mutex_{};
};

}  // namespace transit_pass
