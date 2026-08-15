#include "cloud_quota.h"

#include <chrono>
#include <cstddef>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace {

class check_suite {
public:
    void expect(bool condition, std::string_view description) {
        ++count_;
        if (!condition) {
            failed_ = true;
            std::cerr << "FAILED: " << description << '\n';
        }
    }

    template <typename Function>
    void expect_runtime_error(Function&& function, std::string_view description) {
        ++count_;
        try {
            std::forward<Function>(function)();
        } catch (std::runtime_error const&) {
            return;
        } catch (std::exception const& error) {
            failed_ = true;
            std::cerr << "FAILED: " << description
                      << " threw a different exception: " << error.what() << '\n';
            return;
        }
        failed_ = true;
        std::cerr << "FAILED: " << description
                  << " did not throw std::runtime_error\n";
    }

    std::size_t count() const { return count_; }
    bool failed() const { return failed_; }

private:
    std::size_t count_{0};
    bool failed_{false};
};

}  // namespace

int main() {
    check_suite checks;

    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        checks.expect(subject.available() == 0, "newly_started_zero");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.grant(100);
        checks.expect(subject.available() == 100, "single_credit");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.grant(100);
        subject.grant(50);
        checks.expect(subject.available() == 150, "multiple_credits");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.grant(100);
        subject.consume(75);
        checks.expect(subject.available() == 25, "single_debit");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.grant(100);
        subject.consume(80);
        subject.consume(20);
        checks.expect(subject.available() == 0, "multiple_debits");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.grant(100);
        subject.grant(110);
        subject.consume(200);
        subject.grant(60);
        subject.consume(50);
        checks.expect(subject.available() == 20, "sequential_operations");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.retire();
        checks.expect_runtime_error(
            [&]() { (void)subject.available(); },
            "value_after_stop_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.retire();
        checks.expect_runtime_error(
            [&]() { subject.grant(50); },
            "credit_after_stop_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        checks.expect_runtime_error(
            [&]() { subject.grant(50); },
            "credit_before_start_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.retire();
        checks.expect_runtime_error(
            [&]() { subject.consume(50); },
            "debit_after_stop_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        checks.expect_runtime_error(
            [&]() { subject.retire(); },
            "stop_before_start_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        checks.expect_runtime_error(
            [&]() { subject.provision(); },
            "start_twice_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.grant(50);
        subject.retire();
        subject.provision();
        checks.expect(subject.available() == 0, "restart_resets_zero");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.grant(25);
        checks.expect_runtime_error(
            [&]() { subject.consume(50); },
            "overdraft_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        subject.grant(100);
        checks.expect_runtime_error(
            [&]() { subject.consume(-50); },
            "negative_debit_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        checks.expect_runtime_error(
            [&]() { subject.grant(-50); },
            "negative_credit_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.grant(1);
                std::this_thread::sleep_for(5ms);
                subject.consume(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.available() == 0, "concurrent_transactions");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        checks.expect_runtime_error(
            [&]() { subject.consume(0); },
            "zero_debit_throws");
    }
    {
        cloud_quota::quota_bucket subject{};
        subject.provision();
        checks.expect_runtime_error(
            [&]() { subject.grant(0); },
            "zero_credit_throws");
    }

    if (checks.failed() || checks.count() != 19U) {
        std::cerr << "Verification failed after " << checks.count()
                  << " assertions\n";
        return 1;
    }

    std::cout << "All tests passed (19 assertions in 19 test cases)\n";
    return 0;
}
