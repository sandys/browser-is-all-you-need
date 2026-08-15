#include "telemetry_ring.h"

#include "catch.hpp"

TEST_CASE("a fresh ring is drained") {
    telemetry::SampleRing ring(3);
    REQUIRE(ring.drained());
    REQUIRE_FALSE(ring.saturated());
    REQUIRE(ring.pending() == 0);
    REQUIRE(ring.depth() == 3);
}

TEST_CASE("zero depth is rejected") {
    REQUIRE_THROWS_AS(telemetry::SampleRing(0), std::invalid_argument);
}

TEST_CASE("popping a drained ring throws") {
    telemetry::SampleRing ring(2);
    REQUIRE_THROWS_AS(ring.pop(), telemetry::RingEmpty);
}

TEST_CASE("pushing a saturated ring throws") {
    telemetry::SampleRing ring(2);
    ring.push(1);
    ring.push(2);
    REQUIRE(ring.saturated());
    REQUIRE_THROWS_AS(ring.push(3), telemetry::RingFull);
}

TEST_CASE("samples come back oldest first") {
    telemetry::SampleRing ring(3);
    ring.push(10);
    ring.push(20);
    ring.push(30);
    REQUIRE(ring.pop() == 10);
    REQUIRE(ring.pop() == 20);
    REQUIRE(ring.pop() == 30);
    REQUIRE(ring.drained());
}

TEST_CASE("a full ring is neither drained nor pushable but is poppable") {
    telemetry::SampleRing ring(1);
    ring.push(7);
    REQUIRE(ring.saturated());
    REQUIRE_FALSE(ring.drained());
    REQUIRE(ring.pop() == 7);
    REQUIRE(ring.drained());
    REQUIRE_FALSE(ring.saturated());
}

TEST_CASE("force evicts the oldest sample when saturated") {
    telemetry::SampleRing ring(2);
    ring.push(1);
    ring.push(2);
    ring.force(3);
    REQUIRE(ring.pending() == 2);
    REQUIRE(ring.pop() == 2);
    REQUIRE(ring.pop() == 3);
}

TEST_CASE("force behaves like push when there is room") {
    telemetry::SampleRing ring(3);
    ring.push(1);
    ring.force(2);
    REQUIRE(ring.pending() == 2);
    REQUIRE(ring.pop() == 1);
    REQUIRE(ring.pop() == 2);
}

TEST_CASE("indices wrap without corruption") {
    telemetry::SampleRing ring(3);
    for (int cycle = 0; cycle < 40; ++cycle) {
        ring.push(cycle);
        REQUIRE(ring.pop() == cycle);
    }
    REQUIRE(ring.drained());
}

TEST_CASE("occupancy tracks pushes and pops") {
    telemetry::SampleRing ring(4);
    ring.push(1);
    ring.push(2);
    REQUIRE(ring.pending() == 2);
    ring.pop();
    REQUIRE(ring.pending() == 1);
}

TEST_CASE("purge empties a partially filled ring") {
    telemetry::SampleRing ring(3);
    ring.push(1);
    ring.push(2);
    ring.purge();
    REQUIRE(ring.drained());
    REQUIRE(ring.pending() == 0);
    REQUIRE_THROWS_AS(ring.pop(), telemetry::RingEmpty);
}
