#include "workshop_tokens.h"

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
        workshop_tokens::token_box subject{};
        subject.unlock();
        checks.expect(subject.token_count() == 0, "newly_started_zero");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.add_tokens(100);
        checks.expect(subject.token_count() == 100, "single_credit");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.add_tokens(100);
        subject.add_tokens(50);
        checks.expect(subject.token_count() == 150, "multiple_credits");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.add_tokens(100);
        subject.take_tokens(75);
        checks.expect(subject.token_count() == 25, "single_debit");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.add_tokens(100);
        subject.take_tokens(80);
        subject.take_tokens(20);
        checks.expect(subject.token_count() == 0, "multiple_debits");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.add_tokens(100);
        subject.add_tokens(110);
        subject.take_tokens(200);
        subject.add_tokens(60);
        subject.take_tokens(50);
        checks.expect(subject.token_count() == 20, "sequential_operations");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.lock();
        checks.expect_runtime_error(
            [&]() { (void)subject.token_count(); },
            "value_after_stop_throws");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.lock();
        checks.expect_runtime_error(
            [&]() { subject.add_tokens(50); },
            "credit_after_stop_throws");
    }
    {
        workshop_tokens::token_box subject{};
        checks.expect_runtime_error(
            [&]() { subject.add_tokens(50); },
            "credit_before_start_throws");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.lock();
        checks.expect_runtime_error(
            [&]() { subject.take_tokens(50); },
            "debit_after_stop_throws");
    }
    {
        workshop_tokens::token_box subject{};
        checks.expect_runtime_error(
            [&]() { subject.lock(); },
            "stop_before_start_throws");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        checks.expect_runtime_error(
            [&]() { subject.unlock(); },
            "start_twice_throws");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.add_tokens(50);
        subject.lock();
        subject.unlock();
        checks.expect(subject.token_count() == 0, "restart_resets_zero");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.add_tokens(25);
        checks.expect_runtime_error(
            [&]() { subject.take_tokens(50); },
            "overdraft_throws");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        subject.add_tokens(100);
        checks.expect_runtime_error(
            [&]() { subject.take_tokens(-50); },
            "negative_debit_throws");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        checks.expect_runtime_error(
            [&]() { subject.add_tokens(-50); },
            "negative_credit_throws");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        std::vector<std::thread> threads;
        threads.reserve(1000);
        for (int index = 0; index < 1000; ++index) {
            threads.emplace_back([&]() {
                using namespace std::chrono_literals;
                subject.add_tokens(1);
                std::this_thread::sleep_for(5ms);
                subject.take_tokens(1);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
        checks.expect(subject.token_count() == 0, "concurrent_transactions");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        checks.expect_runtime_error(
            [&]() { subject.take_tokens(0); },
            "zero_debit_throws");
    }
    {
        workshop_tokens::token_box subject{};
        subject.unlock();
        checks.expect_runtime_error(
            [&]() { subject.add_tokens(0); },
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
