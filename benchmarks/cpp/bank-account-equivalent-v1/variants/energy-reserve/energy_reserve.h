#pragma once

#include <mutex>

namespace energy_reserve {

class reserve_meter {
public:
    reserve_meter();
    void enable();
    void add_units(int amount);
    void consume_units(int amount);
    void disable();
    int remaining_units();

private:
    int amount_;
    bool enabled_;
    std::mutex mutex_;
};

}  // namespace energy_reserve
