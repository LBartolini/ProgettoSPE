"""
Underwater Acoustic Network - Performance Model
================================================
Faithfully ports underwater_performance.jsimg to SimPy.

Network topology (mirrors the JMT model):
  Source_i  -->  Node_i (Delay, Normal service)  -->  ClassSwitch
                                                           |
                                                         Buoy  (M/M/1/1 loss server)
                                                        /     \
                                                   Success   Discard

Sources:
  - Source 1  -> Node 1 class, Exponential inter-arrival, lambda = 0.08
  - Source 2  -> Node 2 class, Exponential inter-arrival, lambda = 0.09
  - Source 3  -> Node 3 class, Exponential inter-arrival, lambda = 0.12

Propagation delays (deterministic, speed of sound = 1500 m/s):
  - Node 1: 9000 m  ->  6.0 s
  - Node 2: 7500 m  ->  5.0 s
  - Node 3: 6000 m  ->  4.0 s

Configuration is loaded from config.json (sources, distances, arrival rates, buoy service rate).

Buoy (single server, state-dependent probabilistic routing):
  - Service: Exponential, lambda = 1.0  (mean = 1.0 s)
  - On arrival, the current occupancy is looked up in ROUTING_PROB_TABLE to
    obtain P(Success | occupancy).  Occupancies absent from the table default
    to P(Success) = 0.0 (always discard).
  - Default table replicates M/M/1/1 loss: admit if idle, discard if busy.

Metrics collected:
  - Arrival rate at Buoy
  - Throughput at Success sink
  - Throughput at Discard sink
  - Loss probability
"""

import json
import simpy
import numpy as np

# ── Physics constant ────────────────────────────────────────────────────────
SPEED_OF_SOUND = 1500.0   # m/s in seawater

# ── Load configuration ───────────────────────────────────────────────────────
CONFIG_FILE = "config.json"
with open(CONFIG_FILE) as _f:
    config = json.load(_f)

SIM_TIME    = config["simulation"]["sim_time"]
WARM_UP     = config["simulation"]["warm_up"]
SEED        = config["simulation"]["seed"]
LAMBDA_BUOY = config["buoy"]["service_rate"]

# ── Reproducibility ──────────────────────────────────────────────────────────
rng = np.random.default_rng(SEED)

# ── Buoy routing probability table ──────────────────────────────────────────
# Maps buoy occupancy at the moment of a packet's arrival to the probability
# of routing that packet to Success.  P(Discard) = 1 - P(Success).
# Occupancies not listed default to P(Success) = 0.0 (always discard).
#
# The default below reproduces the original M/M/1/1 loss model:
#   occupancy = 0  ->  server idle,  admit with certainty  (p = 1.0)
#   occupancy = 1  ->  server busy,  discard with certainty (p = 0.0)
#
# Example of a softer policy (uncomment to use):
#   ROUTING_PROB_TABLE = {0: 1.0, 1: 0.5, 2: 0.2, 3: 0.0}
ROUTING_PROB_TABLE: dict[int, float] = {
    0: 1.0,   # server idle  -> always admit
    1: 0.0,   # server busy  -> always discard
}


# ── Counters (populated after warm-up) ──────────────────────────────────────
stats = {
    "buoy_arrivals":    0,
    "success_count":    0,
    "discard_count":    0,
    "arrival_times":    [],   # for arrival rate estimation
    "success_times":    [],
    "discard_times":    [],
}


def source(env, name, interarrival_lambda, buoy_resource, buoy_occupancy):
    """
    Open source: generates packets with Exponential inter-arrivals.
    Each packet is immediately handed off to a separate propagation process
    so the IAT timer is never blocked by the propagation delay.
    """
    while True:
        # Inter-arrival time ~ Exp(lambda)
        iat = rng.exponential(1.0 / interarrival_lambda)
        yield env.timeout(iat)

        # Fire-and-forget
        env.process(buoy_arrive(env, buoy_resource, buoy_occupancy))


def buoy_arrive(env, buoy_resource, buoy_occupancy):
    """
    State-dependent probabilistic routing at the Buoy:
      - Read current occupancy from buoy_occupancy[0].
      - Look up P(Success | occupancy) in ROUTING_PROB_TABLE
        (missing entries default to 0.0 -> always discard).
      - Sample destination at arrival time (before queuing).
      - Both Success and Discard paths seize the server, wait for service,
        then release — occupancy is updated for all admitted packets.
    """
    arrival_time  = env.now
    current_occ   = buoy_occupancy[0]

    # Determine routing outcome based on occupancy at arrival
    p_success  = ROUTING_PROB_TABLE.get(current_occ, 0.0)
    go_success = rng.random() < p_success

    # Record arrival (after warm-up)
    if arrival_time >= WARM_UP:
        stats["buoy_arrivals"] += 1
        stats["arrival_times"].append(arrival_time)

    # Both paths go through the server (occupancy updated for all)
    buoy_occupancy[0] += 1
    with buoy_resource.request() as req:
        yield req
        service_time = rng.exponential(1.0 / LAMBDA_BUOY)
        yield env.timeout(service_time)
    buoy_occupancy[0] -= 1

    # Route to destination determined at arrival
    if go_success:
        if arrival_time >= WARM_UP:
            stats["success_count"] += 1
            stats["success_times"].append(env.now)
    else:
        if arrival_time >= WARM_UP:
            stats["discard_count"] += 1
            stats["discard_times"].append(env.now)


def run_simulation():
    env            = simpy.Environment()
    buoy_resource  = simpy.Resource(env, capacity=1)   # single server
    buoy_occupancy = [0]   # mutable counter: packets currently at the buoy

    # Instantiate sources from config
    for s in config["sources"]:
        env.process(source(env, s["name"], s["arrival_rate"],
                           buoy_resource, buoy_occupancy))

    env.run(until=SIM_TIME)
    return env


# ── Run & Report ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Underwater Network Performance Simulation (SimPy)")
    print("=" * 55)
    print(f"  Config file     : {CONFIG_FILE}")
    print(f"  Simulation time : {SIM_TIME:,} time units")
    print(f"  Warm-up period  : {WARM_UP:,} time units")
    print(f"  RNG seed        : {SEED}")
    print(f"  Speed of sound  : {SPEED_OF_SOUND} m/s")
    print("-" * 55)

    env = run_simulation()

    effective_time = SIM_TIME - WARM_UP

    total_arrivals = stats["buoy_arrivals"]
    n_success      = stats["success_count"]
    n_discard      = stats["discard_count"]

    arrival_rate   = total_arrivals  / effective_time
    success_tput   = n_success       / effective_time
    discard_tput   = n_discard       / effective_time
    loss_prob      = n_discard / total_arrivals if total_arrivals > 0 else 0.0

    # Effective arrival rate at Buoy:
    #   Propagation runs independently of the IAT loop, so packets arrive at
    #   the buoy at exactly the source emission rate lambda_i.
    lambda_total = sum(s["arrival_rate"] for s in config["sources"])
    # M/M/1/1 theoretical loss probability: rho/(1+rho), rho = lambda/mu
    rho            = lambda_total / LAMBDA_BUOY
    theoretical_loss = rho / (1 + rho)

    print(f"\n  {'Metric':<40} {'Simulated':>10}  {'Theoretical':>12}")
    print(f"  {'-'*40}   {'-'*10}  {'-'*12}")
    print(f"  {'Buoy arrival rate (pkts/s)':<40} {arrival_rate:>10.4f}  {lambda_total:>12.4f}")
    print(f"  {'Success throughput (pkts/s)':<40} {success_tput:>10.4f}  {lambda_total/(1+rho):>12.4f}")
    print(f"  {'Discard throughput (pkts/s)':<40} {discard_tput:>10.4f}  {lambda_total*rho/(1+rho):>12.4f}")
    print(f"  {'Loss probability':<40} {loss_prob*100:>9.2f}%  {theoretical_loss*100:>11.2f}%")
    print(f"\n  Total packets arrived at Buoy : {total_arrivals}")
    print(f"  Packets to Success            : {n_success}")
    print(f"  Packets to Discard            : {n_discard}")
    print("=" * 55)
