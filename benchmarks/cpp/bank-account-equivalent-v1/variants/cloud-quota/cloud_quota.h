#pragma once

#include <mutex>

namespace cloud_quota {

class quota_bucket {
public:
    quota_bucket();
    void provision();
    void grant(int amount);
    void consume(int amount);
    void retire();
    int available();

private:
    int value_{0};
    bool active_{false};
    std::mutex mutex_{};
};

}  // namespace cloud_quota
