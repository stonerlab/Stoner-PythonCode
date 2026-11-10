"""Apply the shared migration benchmark acceptance policy to reported measurements."""


def migration_limits(legacy, workload):
    """Return agreed timing and traced-allocation ceilings for one legacy measurement."""
    dimensions = legacy["dimensions"]
    if "rows" in dimensions:
        payload = (dimensions["rows"] + dimensions.get("appends", 0)) * dimensions["columns"] * 9
    elif "frames" in dimensions:
        frames = dimensions["frames"] + (workload == "stack_insert")
        payload = frames * dimensions["height"] * dimensions["width"] * 3
    else:
        payload = 0
    return (max(2 * legacy["median_seconds"], legacy["median_seconds"] + 0.005),
            max(2 * legacy["traced_peak_bytes"], 3 * payload + 1024**2))
