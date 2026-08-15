"""Episode ladder for the circular-buffer pilot (issue #110, step 2).

Three tiers of initial state over lineage-disjoint ring domains:

  tier 1  atomic repair    complete implementation with exactly one broken
                           invariant: empty/full ambiguity, overwrite movement,
                           wraparound, capacity, or size
  tier 2  partial repo     occupancy and eviction transitions removed
  tier 3  full solve       every body stubbed; the whole invariant set is open

Starters are expressed as fail-closed edits from the reference: an anchor that
does not match exactly once raises, so an episode cannot silently ship without
the defect it claims to carry.

`control` is the "distinct compiling semantic error" the step-3 contract
requires. It is deliberately a different mechanism from every tier-1 mutation,
so passing the negative-control check is evidence the oracle discriminates
rather than merely rejecting anything unusual.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class InjectionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Edit:
    filename: str
    old: str
    new: str


@dataclass(frozen=True)
class Replace:
    filename: str
    body: str


@dataclass(frozen=True)
class Episode:
    episode_id: str
    task: str
    tier: int
    invariant: str
    intent: str
    edits: tuple = field(default_factory=tuple)

    def apply(self, files: dict[str, str]) -> dict[str, str]:
        out = dict(files)
        for edit in self.edits:
            if isinstance(edit, Replace):
                if edit.filename not in out:
                    raise InjectionError(f"{self.episode_id}: missing {edit.filename}")
                out[edit.filename] = edit.body
                continue
            body = out.get(edit.filename)
            if body is None:
                raise InjectionError(f"{self.episode_id}: missing {edit.filename}")
            hits = body.count(edit.old)
            if hits != 1:
                raise InjectionError(f"{self.episode_id}: anchor matched {hits}x in {edit.filename}")
            out[edit.filename] = body.replace(edit.old, edit.new)
        return out


EPISODES: list[Episode] = []
CONTROLS: dict[str, Episode] = {}


def add(episode_id, task, tier, invariant, intent, *edits):
    EPISODES.append(Episode(episode_id, task, tier, invariant, intent, tuple(edits)))


def control(task, episode):
    CONTROLS[task] = episode


# =============================================================== sensor-ring
SR = "telemetry_ring.cpp"

add("sensor-ring/t1-empty-full-ambiguity", "sensor-ring", 1, "empty-full-ambiguity",
    "drained() is derived from index equality, so it reports true when saturated too",
    Edit(SR, "bool SampleRing::drained() const { return pending_ == 0; }",
             "bool SampleRing::drained() const {\n"
             "    return head_ == (head_ + pending_) % slots_.size();\n}"))

add("sensor-ring/t1-overwrite-movement", "sensor-ring", 1, "overwrite-movement",
    "force() drops a sample without advancing the read cursor, so eviction takes the wrong end",
    Edit(SR, "        head_ = (head_ + 1) % slots_.size();\n        --pending_;\n", "        --pending_;\n"))

add("sensor-ring/t1-wraparound", "sensor-ring", 1, "wraparound",
    "push() indexes without the modulus, so the write cursor leaves the ring",
    Edit(SR, "slots_.at((head_ + pending_) % slots_.size()) = sample;",
             "slots_.at(head_ + pending_) = sample;"))

add("sensor-ring/t1-capacity", "sensor-ring", 1, "capacity",
    "the ring allocates one slot more than the configured depth",
    Edit(SR, "SampleRing::SampleRing(std::size_t depth) : slots_(depth) {",
             "SampleRing::SampleRing(std::size_t depth) : slots_(depth + 1) {"))

add("sensor-ring/t1-size-invariant", "sensor-ring", 1, "size-invariant",
    "pop() removes a sample without decrementing the occupancy count",
    Edit(SR, "    --pending_;\n    return sample;", "    return sample;"))

add("sensor-ring/t2-partial", "sensor-ring", 2, "occupancy-and-eviction",
    "eviction and the saturation predicate are removed; the transitions must be rebuilt",
    Edit(SR, "void SampleRing::force(int sample) {\n"
             "    if (saturated()) {\n"
             "        head_ = (head_ + 1) % slots_.size();\n"
             "        --pending_;\n"
             "    }\n"
             "    push(sample);\n}",
             "void SampleRing::force(int) {}"),
    Edit(SR, "bool SampleRing::saturated() const { return pending_ == slots_.size(); }",
             "bool SampleRing::saturated() const { return false; }"))

add("sensor-ring/t3-full-solve", "sensor-ring", 3, "complete-invariant-set",
    "every body is stubbed; the whole ring contract is open",
    Replace(SR, """#include "telemetry_ring.h"

namespace telemetry {

SampleRing::SampleRing(std::size_t depth) : slots_(depth) {}

void SampleRing::push(int) {}

int SampleRing::pop() { return static_cast<int>(head_); }

void SampleRing::force(int) {}

void SampleRing::purge() {}

std::size_t SampleRing::pending() const { return pending_; }

std::size_t SampleRing::depth() const { return slots_.size(); }

bool SampleRing::saturated() const { return false; }

bool SampleRing::drained() const { return true; }

}  // namespace telemetry
"""))

control("sensor-ring", Episode(
    "sensor-ring/control-lifo", "sensor-ring", 0, "read-order",
    "pop() returns the newest sample instead of the oldest: compiles, wrong order, "
    "and distinct from every tier-1 mutation",
    (Edit(SR, "const int sample = slots_.at(head_);",
              "const int sample = slots_.at((head_ + pending_ - 1) % slots_.size());"),)))


# =============================================================== print-spool
PS = "print_spool.cpp"

add("print-spool/t1-empty-full-ambiguity", "print-spool", 1, "empty-full-ambiguity",
    "idle() is derived from index equality, so it reports true when full too",
    Edit(PS, "bool PrintSpool::idle() const { return backlog_ == 0; }",
             "bool PrintSpool::idle() const {\n"
             "    return front_ == (front_ + backlog_) % ring_.size();\n}"))

add("print-spool/t1-overwrite-movement", "print-spool", 1, "overwrite-movement",
    "displace() drops a job without advancing the read cursor, so the wrong job is evicted",
    Edit(PS, "        front_ = (front_ + 1) % ring_.size();\n        --backlog_;\n", "        --backlog_;\n"))

add("print-spool/t1-wraparound", "print-spool", 1, "wraparound",
    "queue() indexes without the modulus, so the write cursor leaves the ring",
    Edit(PS, "ring_.at((front_ + backlog_) % ring_.size()) = job;",
             "ring_.at(front_ + backlog_) = job;"))

add("print-spool/t1-capacity", "print-spool", 1, "capacity",
    "the spool allocates one slot more than configured",
    Edit(PS, "PrintSpool::PrintSpool(std::size_t slots) : ring_(slots) {",
             "PrintSpool::PrintSpool(std::size_t slots) : ring_(slots + 1) {"))

add("print-spool/t1-size-invariant", "print-spool", 1, "size-invariant",
    "release() removes a job without decrementing the backlog",
    Edit(PS, "    --backlog_;\n    return job;", "    return job;"))

add("print-spool/t2-partial", "print-spool", 2, "occupancy-and-eviction",
    "eviction and the fullness predicate are removed; the transitions must be rebuilt",
    Edit(PS, "void PrintSpool::displace(const std::string& job) {\n"
             "    if (full()) {\n"
             "        front_ = (front_ + 1) % ring_.size();\n"
             "        --backlog_;\n"
             "    }\n"
             "    queue(job);\n}",
             "void PrintSpool::displace(const std::string&) {}"),
    Edit(PS, "bool PrintSpool::full() const { return backlog_ == ring_.size(); }",
             "bool PrintSpool::full() const { return false; }"))

add("print-spool/t3-full-solve", "print-spool", 3, "complete-invariant-set",
    "every body is stubbed; the whole spool contract is open",
    Replace(PS, """#include "print_spool.h"

namespace spool {

PrintSpool::PrintSpool(std::size_t slots) : ring_(slots) {}

void PrintSpool::queue(const std::string&) {}

std::string PrintSpool::release() { return ring_.at(front_); }

void PrintSpool::displace(const std::string&) {}

void PrintSpool::abandon() {}

std::size_t PrintSpool::backlog() const { return backlog_; }

std::size_t PrintSpool::slots() const { return ring_.size(); }

bool PrintSpool::full() const { return false; }

bool PrintSpool::idle() const { return true; }

}  // namespace spool
"""))

control("print-spool", Episode(
    "print-spool/control-lifo", "print-spool", 0, "read-order",
    "release() returns the newest job instead of the longest waiting: compiles, wrong order, "
    "and distinct from every tier-1 mutation",
    (Edit(PS, "const std::string job = ring_.at(front_);",
              "const std::string job = ring_.at((front_ + backlog_ - 1) % ring_.size());"),)))
