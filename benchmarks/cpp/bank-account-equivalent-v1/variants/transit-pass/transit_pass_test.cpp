#include "transit_pass.h"

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
        transit_pass::fare_pass subject{};
        subject.activate();
        checks.expect(subject.funds() == 0, "newly_started_zero");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.top_up(100);
        checks.expect(subject.funds() == 100, "single_credit");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.top_up(100);
        subject.top_up(50);
        checks.expect(subject.funds() == 150, "multiple_credits");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.top_up(100);
        subject.charge(75);
        checks.expect(subject.funds() == 25, "single_debit");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.top_up(100);
        subject.charge(80);
        subject.charge(20);
        checks.expect(subject.funds() == 0, "multiple_debits");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.top_up(100);
        subject.top_up(110);
        subject.charge(200);
        subject.top_up(60);
        subject.charge(50);
        checks.expect(subject.funds() == 20, "sequential_operations");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.suspend();
        checks.expect_runtime_error(
            [&]() { (void)subject.funds(); },
            "value_after_stop_throws");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.suspend();
        checks.expect_runtime_error(
            [&]() { subject.top_up(50); },
            "credit_after_stop_throws");
    }
    {
        transit_pass::fare_pass subject{};
        checks.expect_runtime_error(
            [&]() { subject.top_up(50); },
            "credit_before_start_throws");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.suspend();
        checks.expect_runtime_error(
            [&]() { subject.charge(50); },
            "debit_after_stop_throws");
    }
    {
        transit_pass::fare_pass subject{};
        checks.expect_runtime_error(
            [&]() { subject.suspend(); },
            "stop_before_start_throws");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        checks.expect_runtime_error(
            [&]() { subject.activate(); },
            "start_twice_throws");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.top_up(50);
        subject.suspend();
        subject.activate();
        checks.expect(subject.funds() == 0, "restart_resets_zero");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.top_up(25);
        checks.expect_runtime_error(
            [&]() { subject.charge(50); },
            "overdraft_throws");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        subject.top_up(100);
        checks.expect_runtime_error(
            [&]() { subject.charge(-50); },
            "negative_debit_throws");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        checks.expect_runtime_error(
            [&]() { subject.top_up(-50); },
            "negative_credit_throws");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.top_up(1);
                std::this_thread::sleep_for(5ms);
                subject.charge(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.funds() == 0, "concurrent_transactions");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        checks.expect_runtime_error(
            [&]() { subject.charge(0); },
            "zero_debit_throws");
    }
    {
        transit_pass::fare_pass subject{};
        subject.activate();
        checks.expect_runtime_error(
            [&]() { subject.top_up(0); },
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
