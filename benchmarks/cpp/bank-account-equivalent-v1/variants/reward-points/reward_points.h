#pragma once

#include <mutex>

namespace reward_points {

class reward_account {
public:
    reward_account();
    void open();
    void earn(int amount);
    void redeem(int amount);
    void close();
    int points();

private:
    struct state {
        int value{0};
        bool active{false};
    };

    state state_{};
    std::mutex mutex_{};
};

}  // namespace reward_points
