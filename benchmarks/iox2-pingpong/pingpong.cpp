// iceoryx2 ping-pong RTT microbenchmark (from the Task 8 spike):
// publish/subscribe of a 64-byte message plus an event per direction, like the
// executor <-> gpuless path.
//   pingpong pong            echo side; exits when ping sends its final message
//   pingpong ping [n]        n timed round trips (after 1000 warm-up), prints
//   min/p50/p90/p99/max in us pingpong multi <rA> <rB> checks that two nodes
//   with different root-paths in one process are isolated
// Env: IOX2_ROOT = iceoryx2 root-path (unset = config default), IOX2_PREFIX =
// file name prefix (unset = config default "iox2_"), BUSY=1 = spin on receive
// instead of waiting.
#include "iox2/iceoryx2.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using namespace iox2;
using Clock = std::chrono::steady_clock;

struct Msg {
  uint64_t seq;
  char pad[56];
};

static Config cfg_for(const char *root) {
  auto cfg = Config::global_config().to_owned();
  if (root) {
    auto s = bb::StaticString<bb::platform::IOX2_MAX_PATH_LENGTH>::
        from_utf8_null_terminated_unchecked(root);
    cfg.global().set_root_path(bb::Path::create(s.value()).value());
  }
  if (const char *prefix = getenv(
          "IOX2_PREFIX")) { // private names for the POSIX shm files in /dev/shm
    auto s = bb::StaticString<bb::platform::IOX2_MAX_FILENAME_LENGTH>::
        from_utf8_null_terminated_unchecked(prefix);
    cfg.global().set_prefix(bb::FileName::create(s.value()).value());
  }
  return cfg;
}

static Node<ServiceType::Ipc> node_for(const char *root) {
  return NodeBuilder().config(cfg_for(root)).create<ServiceType::Ipc>().value();
}

static int count_services(const char *root) {
  auto cfg = cfg_for(root);
  int n = 0;
  Service<ServiceType::Ipc>::list(cfg.view(), [&](auto) {
    ++n;
    return CallbackProgression::Continue;
  }).value();
  return n;
}

// Checks the assumption behind per-client iceoryx2 directories: one process
// (the orchestrator) can hold nodes with different root-paths, and they are
// fully separate even when they use the same service names. Roots a and b stand
// for two clients' directories.
static int multi(const char *a, const char *b) {
  // One node per root, each with an event service of the same name ("svc") and
  // a listener on it. If the roots weren't isolated, both nodes would open the
  // same service.
  auto na = node_for(a);
  auto nb = node_for(b);
  auto ea = na.service_builder(ServiceName::create("svc").value())
                .event()
                .open_or_create()
                .value();
  auto eb = nb.service_builder(ServiceName::create("svc").value())
                .event()
                .open_or_create()
                .value();
  auto la = ea.listener_builder().create().value();
  auto lb = eb.listener_builder().create().value();

  // A service that exists only under root b (kept alive by pb).
  auto pb = nb.service_builder(ServiceName::create("onlyB").value())
                .publish_subscribe<Msg>()
                .open_or_create()
                .value();

  // One waitset (default config) over listeners of both nodes, like the
  // orchestrator.
  auto ws = WaitSetBuilder().create<ServiceType::Ipc>().value();
  auto ga = ws.attach_notification(la).value();
  auto gb = ws.attach_notification(lb).value();

  // 1. A notification on a's "svc" must wake only a's listener, not b's.
  auto notifier = ea.notifier_builder().create().value();
  notifier.notify().value();
  bool got_a = false, got_b = false;
  ws.wait_and_process_once_with_timeout(
        [&](WaitSetAttachmentId<ServiceType::Ipc> id) {
          if (id.has_event_from(ga))
            got_a = true;
          if (id.has_event_from(gb))
            got_b = true;
          return CallbackProgression::Continue;
        },
        bb::Duration::from_millis(500))
      .value();
  printf("waitset: event from A=%d B=%d (expect 1 0)\n", got_a, got_b);

  // 2. A service created under root b can't be opened from node a, only from
  // node b.
  auto open_b_from_a = na.service_builder(ServiceName::create("onlyB").value())
                           .publish_subscribe<Msg>()
                           .open();
  printf("open onlyB from node A: %s (expect failure)\n",
         open_b_from_a.has_value() ? "OPENED" : "failed");
  auto open_b_from_b = nb.service_builder(ServiceName::create("onlyB").value())
                           .publish_subscribe<Msg>()
                           .open();
  printf("open onlyB from node B: %s (expect opened)\n",
         open_b_from_b.has_value() ? "OPENED" : "failed");

  // 3. Listing services per root sees only that root's services: "svc" under a;
  // "svc" and "onlyB" under b.
  printf("services under A: %d, under B: %d (expect 1, 2)\n", count_services(a),
         count_services(b));
  bool ok = got_a && !got_b && !open_b_from_a.has_value() &&
            open_b_from_b.has_value() && count_services(a) == 1 &&
            count_services(b) == 2;
  printf("multi: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}

// ping: publishes on ping/, waits for pong/; pong echoes. Root from IOX2_ROOT
// (unset = config default).
static int pingpong(bool is_ping, int iters) {
  const char *root = getenv("IOX2_ROOT");
  auto node = node_for(root);
  auto mk_ps = [&](const char *n) {
    return node.service_builder(ServiceName::create(n).value())
        .publish_subscribe<Msg>()
        .max_publishers(1)
        .max_subscribers(1)
        .open_or_create()
        .value();
  };
  auto mk_ev = [&](const char *n) {
    return node.service_builder(ServiceName::create(n).value())
        .event()
        .open_or_create()
        .value();
  };
  // Each side publishes on its own topic and subscribes to the other's; a
  // separate event service per direction wakes the receiver, as on the
  // executor <-> gpuless path (data sample + notification).
  auto ps_out = mk_ps(is_ping ? "pp/ping" : "pp/pong");
  auto ps_in = mk_ps(is_ping ? "pp/pong" : "pp/ping");
  auto ev_out = mk_ev(is_ping ? "pp/ping.ev" : "pp/pong.ev");
  auto ev_in = mk_ev(is_ping ? "pp/pong.ev" : "pp/ping.ev");
  auto pub = ps_out.publisher_builder().create().value();
  auto sub = ps_in.subscriber_builder().create().value();
  auto notifier = ev_out.notifier_builder().create().value();
  auto listener = ev_in.listener_builder().create().value();
  const char *bz = getenv("BUSY");
  bool busy = bz && *bz;

  // Receives the next message and returns its sequence number. Wait mode
  // blocks on the event, then reads the sample; a wakeup without a sample
  // (e.g. a leftover notification) just waits again. BUSY=1 spins on
  // receive() instead and drains the notifications it didn't wait for, so
  // they don't pile up in the event socket.
  auto recv_one = [&]() -> uint64_t {
    for (;;) {
      if (!busy)
        listener.blocking_wait_one().value();
      auto s = sub.receive().value();
      if (s.has_value()) {
        if (busy)
          while (listener.try_wait_one().value().has_value()) {
          }
        return s->payload().seq;
      }
    }
  };
  // Publishes the sequence number, then notifies the other side.
  auto send_one = [&](uint64_t seq) {
    auto smp = pub.loan_uninit().value();
    send(smp.write_payload(Msg{seq, {}})).value();
    notifier.notify().value();
  };

  // pong: echoes every message back unchanged. UINT64_MAX is ping's final
  // message: pong echoes it like the others (nobody waits for that one), then
  // exits.
  if (!is_ping) {
    for (;;) {
      uint64_t s = recv_one();
      send_one(s);
      if (s == UINT64_MAX)
        return 0;
    }
  }
  // ping: wait until pong subscribes (a message sent earlier would be lost),
  // plus 100 ms for its listener and publisher to come up.
  while (ps_out.dynamic_config().number_of_subscribers() == 0)
    usleep(1000);
  usleep(100000);
  // Round trip = send + pong's echo received. The first 1000 are warm-up and
  // not recorded.
  std::vector<double> lat;
  for (int i = 0; i < iters + 1000; ++i) {
    auto t0 = Clock::now();
    send_one(i);
    recv_one();
    if (i >= 1000)
      lat.push_back(
          std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
  }
  send_one(UINT64_MAX); // tells pong to exit (its echo is not awaited)
  std::sort(lat.begin(), lat.end());
  auto p = [&](double q) { return lat[size_t(q * (lat.size() - 1))]; };
  printf("root=%s busy=%d n=%zu RTT us: min %.2f p50 %.2f p90 %.2f p99 %.2f "
         "max %.2f\n",
         root ? root : "(default)", busy, lat.size(), lat.front(), p(0.5),
         p(0.9), p(0.99), lat.back());
  return 0;
}

int main(int argc, char **argv) {
  set_log_level_from_env_or(LogLevel::Warn);
  if (argc >= 4 && !strcmp(argv[1], "multi"))
    return multi(argv[2], argv[3]);
  if (argc >= 2 && !strcmp(argv[1], "ping"))
    return pingpong(true, argc >= 3 ? atoi(argv[2]) : 20000);
  if (argc >= 2 && !strcmp(argv[1], "pong"))
    return pingpong(false, 0);
  fprintf(stderr, "usage: %s multi <rootA> <rootB> | ping [n] | pong\n",
          argv[0]);
  return 2;
}
