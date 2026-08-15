#pragma once

#include <mutex>

namespace arcade_card {

class player_card {
public:
    player_card();
    void issue();
    void load(int amount);
    void spend(int amount);
    void revoke();
    int credits();

private:
    int value_{0};
    bool active_{false};
    std::mutex mutex_{};

    void require_active() const;
    static void require_positive(int amount);
};

}  // namespace arcade_card
