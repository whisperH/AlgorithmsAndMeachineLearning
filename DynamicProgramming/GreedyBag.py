states_needed = set(["mt", "wa", "or", "id", "nv", "ut", "ca", "az"])
stations = {}
stations["kone"] = set(["id", "nv", "ut"])
stations["ktwo"] = set(["wa", "id", "mt"])
stations["kthree"] = set(["or", "nv", "ca"])
stations["kfour"] = set(["nv", "ut"])
stations["kfive"] = set(["ca", "az"])

final_station = set()

while states_needed:
    best_station = None
    current_state = None
    max_len = 0
    for station, state in stations.items():
        current_state = state & states_needed
        if len(current_state) > max_len:
            best_station = station
            max_len = len(current_state)

    states_needed -= stations[best_station]
    final_station.add(best_station)

print(final_station)