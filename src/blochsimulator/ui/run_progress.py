"""Elapsed time, percentage and remaining time of background simulation runs.

Native simulation chunks report progress every few seconds. Between reports
the completed work is extrapolated at the measured rate, but never beyond the
next report: if a report is overdue, the run is slower than measured and the
remaining time grows. The first chunk is a deliberately small leading chunk,
so the size of the next report is only known from the second chunk on. The
rate comes from the latest part of the run, so it follows a laptop that slows
down under sustained load instead of the fast first seconds.
"""

import math

# Share of the measured run time, counted back from the latest report, over
# which the work rate is measured.
RATE_WINDOW_FRACTION = 0.5
# Running bars stay below 100% until the run reports completion.
MAX_RUNNING_FRACTION = 0.99


def format_duration(seconds):
    """Format a short, stable duration for progress and completion messages."""
    seconds = max(0.0, float(seconds))
    if seconds < 1.0:
        return "<1s"
    total_seconds = int(math.ceil(seconds))
    if total_seconds < 60:
        return f"{total_seconds}s"
    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:02d}s"
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes:02d}m"


def remaining_text(remaining_s):
    """Describe an estimated remaining time; None means not yet known."""
    if remaining_s is None:
        return "estimating time left…"
    if remaining_s <= 0.0:
        return "almost done…"
    if remaining_s < 1.0:
        return "<1s left"
    return f"~{format_duration(remaining_s)} left"


class RunProgressEstimator:
    """Track the reported progress of one run and estimate its remaining time.

    The first report, sent after preparation and compilation, is the
    baseline, so preparation time does not count toward the work rate. The
    displayed fraction never moves back.
    """

    def __init__(self, started_at, total=1):
        self.started_at = float(started_at)
        self.total = max(1, int(total))
        self.done = 0
        self._samples = []
        self._steps = 0
        self._largest_step = 0
        self._displayed_fraction = 0.0

    def report(self, done, total, now):
        """Record that ``done`` of ``total`` work units are complete at ``now``."""
        self.total = max(1, int(total))
        done = min(self.total, max(0, int(done)))
        if self._samples and done < self._samples[-1][1]:
            # A restarted count cannot share a rate with the earlier one.
            self._samples = []
            self._steps = 0
            self._largest_step = 0
        if not self._samples or done > self._samples[-1][1]:
            if self._samples:
                self._steps += 1
                self._largest_step = max(
                    self._largest_step, done - self._samples[-1][1]
                )
            self._samples.append((float(now), done))
        self.done = done

    @property
    def complete(self):
        return self.done >= self.total

    @property
    def rate(self):
        """Work units per second over the latest part of the run, or None."""
        if len(self._samples) < 2:
            return None
        first_time = self._samples[0][0]
        last_time, last_done = self._samples[-1]
        window_start = last_time - RATE_WINDOW_FRACTION * (last_time - first_time)
        reference_time, reference_done = self._samples[0]
        for sample_time, sample_done in self._samples[1:-1]:
            if sample_time > window_start:
                break
            reference_time, reference_done = sample_time, sample_done
        if last_time <= reference_time:
            return None
        return (last_done - reference_done) / (last_time - reference_time)

    def _estimate(self, now):
        """Return the estimated done units and remaining seconds at ``now``."""
        if self.complete:
            return float(self.total), 0.0
        rate = self.rate
        if rate is None or rate <= 0.0:
            return float(self.done), None
        last_time, last_done = self._samples[-1]
        waited_s = max(0.0, float(now) - last_time)
        # The next report usually arrives after one more chunk. After only the
        # small leading chunk, the next one may be much larger.
        next_report = self.total
        if self._steps >= 2:
            next_report = min(self.total, last_done + self._largest_step)
        done = last_done + rate * waited_s
        if done > next_report:
            # That report is overdue: assume it arrives now, at a lower rate.
            done = float(next_report)
            rate = (next_report - last_done) / waited_s
        return done, (self.total - done) / rate

    def remaining_s(self, now):
        """Estimated seconds left, 0.0 when complete, or None while unknown."""
        return self._estimate(now)[1]

    def status(self, now):
        """Return the displayed fraction and the progress text at ``now``."""
        done, remaining_s = self._estimate(now)
        if self.complete:
            fraction = 1.0
            remaining = "finishing…"
        else:
            fraction = min(MAX_RUNNING_FRACTION, done / self.total)
            remaining = remaining_text(remaining_s)
        fraction = max(self._displayed_fraction, fraction)
        self._displayed_fraction = fraction
        elapsed = format_duration(float(now) - self.started_at)
        return fraction, f"{int(fraction * 100)}% · {elapsed} elapsed · {remaining}"


class LoopProgressEstimator:
    """Estimate the remaining time of a loop of runs from its finished runs.

    Until a run has finished, the current run's own estimate stands in for
    the duration of every run.
    """

    def __init__(self, started_at, count):
        self.started_at = float(started_at)
        self.count = max(1, int(count))
        self.completed = 0
        self._run_started_at = float(started_at)
        self._run_durations = []
        self._displayed_fraction = 0.0

    def run_finished(self, now):
        """Record that the current run, including saving, ended at ``now``."""
        self._run_durations.append(max(0.0, float(now) - self._run_started_at))
        self._run_started_at = float(now)
        self.completed += 1

    def remaining_s(self, now, run_remaining_s=None):
        """Estimated seconds left for the whole loop, or None while unknown."""
        later_runs = max(0, self.count - self.completed - 1)
        if self.completed >= self.count:
            return 0.0
        run_elapsed_s = max(0.0, float(now) - self._run_started_at)
        if self._run_durations:
            run_s = sum(self._run_durations) / len(self._run_durations)
            if run_remaining_s is None:
                run_remaining_s = max(0.0, run_s - run_elapsed_s)
        elif run_remaining_s is not None:
            run_s = run_elapsed_s + run_remaining_s
        else:
            return None
        return run_remaining_s + later_runs * run_s

    def status(self, now, run_fraction=0.0, run_remaining_s=None):
        """Return the loop fraction and the loop progress text at ``now``."""
        run_fraction = min(1.0, max(0.0, float(run_fraction)))
        fraction = min(1.0, (self.completed + run_fraction) / self.count)
        # Between runs no run fraction is known; the bar keeps its position.
        fraction = max(self._displayed_fraction, fraction)
        self._displayed_fraction = fraction
        position = min(self.count, self.completed + 1)
        elapsed = format_duration(float(now) - self.started_at)
        remaining = remaining_text(self.remaining_s(now, run_remaining_s))
        return fraction, (
            f"Loop {position}/{self.count} · {elapsed} elapsed · {remaining}"
        )
