"""Independent checks of disk risk, using only the Python standard library.

Distances, Gaussian standard deviations and disk radii must share a unit. Gaussian densities have inverse-area units.

Both shared-space Monte Carlo estimators are provided:
  * box: area(box) / N_total * sum(density * disk_indicator);
  * conditional: area(disk) / N_disk * sum(density_inside_disk).
The conditional estimate is unavailable when N_disk == 0. An unconditional
box estimate of zero after zero hits is not evidence that the risk is zero.
The error of the nonlinear multi-obstacle product can be correlated and biased
at finite sample counts because all integrals share the same spatial points.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

Point = tuple[float, float]


def _check_point(point: Point) -> None:
    if len(point) != 2 or not all(math.isfinite(value) for value in point):
        raise ValueError("A point must contain two finite coordinates.")


@dataclass(frozen=True)
class Gaussian2D:
    """An isotropic 2D Gaussian; covariance is sigma**2 times identity."""

    mean: Point
    sigma: float

    def __post_init__(self) -> None:
        _check_point(self.mean)
        if not math.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be finite and strictly positive.")

    def density(self, point: Point) -> float:
        dx = (point[0] - self.mean[0]) / self.sigma
        dy = (point[1] - self.mean[1]) / self.sigma
        return math.exp(-0.5 * (dx * dx + dy * dy)) / (
            2.0 * math.pi * self.sigma * self.sigma
        )


@dataclass(frozen=True)
class GaussianMixture2D:
    """A normalized mixture of (weight, Gaussian2D) modes.

    Modes represent mutually exclusive alternatives of ONE physical obstacle;
    they must not be combined as independent obstacles.
    """

    modes: tuple[tuple[float, Gaussian2D], ...]

    def __post_init__(self) -> None:
        if not self.modes:
            raise ValueError("A Gaussian mixture needs at least one mode.")
        for weight, component in self.modes:
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Mixture weights must be finite and nonnegative.")
            if not isinstance(component, Gaussian2D):
                raise TypeError("Every mixture component must be Gaussian2D.")
        if not math.isclose(
            math.fsum(weight for weight, _ in self.modes),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("Mixture weights must sum to one.")

    def density(self, point: Point) -> float:
        return math.fsum(
            weight * component.density(point)
            for weight, component in self.modes
        )


Distribution2D = Gaussian2D | GaussianMixture2D


@dataclass(frozen=True)
class Disk:
    center: Point
    radius: float

    def __post_init__(self) -> None:
        _check_point(self.center)
        if not math.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Disk radius must be finite and strictly positive.")

    @property
    def area(self) -> float:
        return math.pi * self.radius * self.radius

    def contains(self, point: Point) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        return dx * dx + dy * dy <= self.radius * self.radius


@dataclass(frozen=True)
class Rectangle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.xmin, self.xmax, self.ymin, self.ymax)
        ):
            raise ValueError("Rectangle bounds must be finite.")
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("Rectangle must have positive width and height.")

    @property
    def area(self) -> float:
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def covers(self, disk: Disk) -> bool:
        x, y = disk.center
        r = disk.radius
        return (
            self.xmin <= x - r
            and self.xmax >= x + r
            and self.ymin <= y - r
            and self.ymax >= y + r
        )


def covering_rectangle(disks: Sequence[Disk]) -> Rectangle:
    """Construct the smallest axis-aligned box covering all complete disks."""
    if not disks:
        raise ValueError("At least one disk is required.")
    return Rectangle(
        min(disk.center[0] - disk.radius for disk in disks),
        max(disk.center[0] + disk.radius for disk in disks),
        min(disk.center[1] - disk.radius for disk in disks),
        max(disk.center[1] + disk.radius for disk in disks),
    )


def _normal_interval_probability(lower: float, upper: float) -> float:
    """Standard-normal interval mass with stable same-tail subtraction."""
    root_two = math.sqrt(2.0)
    if lower >= 0:
        return 0.5 * (
            math.erfc(lower / root_two) - math.erfc(upper / root_two)
        )
    if upper <= 0:
        return 0.5 * (
            math.erfc(-upper / root_two) - math.erfc(-lower / root_two)
        )
    return 0.5 * (math.erf(upper / root_two) - math.erf(lower / root_two))


def disk_reference(
    distribution: Distribution2D, disk: Disk, *, panels: int = 1024
) -> float:
    """Deterministic disk integral for isotropic Gaussians or their mixtures.

    Use x = center_x + radius*sin(theta). This removes the square-root slice
    endpoint singularity. The y slice is integrated by the Gaussian CDF and
    theta by composite Simpson quadrature. This is a numerical reference,
    not a certified bound: callers should compare panels and 2*panels for
    demanding parameter regimes, especially radius/sigma >> 1.
    """
    if isinstance(panels, bool) or not isinstance(panels, int):
        raise ValueError("panels must be an even positive integer.")
    if panels < 2 or panels % 2:
        raise ValueError("panels must be an even integer >= 2.")
    if isinstance(distribution, GaussianMixture2D):
        return math.fsum(
            weight * disk_reference(component, disk, panels=panels)
            for weight, component in distribution.modes
        )
    if not isinstance(distribution, Gaussian2D):
        raise TypeError("Reference integration supports Gaussian2D and mixtures.")

    sigma = distribution.sigma
    qx, qy = disk.center
    mux, muy = distribution.mean
    step = math.pi / panels

    def integrand(index: int) -> float:
        # Both mathematical endpoints are zero; set them exactly to zero.
        if index == 0 or index == panels:
            return 0.0
        theta = -math.pi / 2.0 + index * step
        cosine = math.cos(theta)
        x = qx + disk.radius * math.sin(theta)
        half_slice = disk.radius * cosine
        y_mass = _normal_interval_probability(
            (qy - half_slice - muy) / sigma,
            (qy + half_slice - muy) / sigma,
        )
        x_density = math.exp(-0.5 * ((x - mux) / sigma) ** 2) / (
            sigma * math.sqrt(2.0 * math.pi)
        )
        return disk.radius * cosine * x_density * y_mass

    weighted = math.fsum(
        (4.0 if index % 2 else 2.0) * integrand(index)
        for index in range(1, panels)
    )
    return weighted * step / 3.0


def independent_union_probability(probabilities: Sequence[float]) -> float:
    """Probability of any event, assuming PHYSICAL event independence.

    Invalid input probabilities are rejected, never silently clamped. Shared
    MC points do not justify or invalidate physical independence by themselves.
    """
    if any(
        not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in probabilities
    ):
        raise ValueError("Each event probability must lie in [0, 1].")
    if not probabilities:
        return 0.0
    if any(probability == 1.0 for probability in probabilities):
        return 1.0
    return -math.expm1(
        math.fsum(math.log1p(-probability) for probability in probabilities)
    )




@dataclass(frozen=True)
class DiskMCResult:
    disk: Disk
    hit_count: int
    rectangle_probabilities: tuple[float, ...]
    conditional_probabilities: tuple[float, ...] | None
    rectangle_joint: float | None
    conditional_joint: float | None


@dataclass(frozen=True)
class SharedMCResult:
    bounds: Rectangle
    n_samples: int
    seed: int
    disks: tuple[DiskMCResult, ...]


def _joint_if_valid(probabilities: Sequence[float]) -> float | None:
    # Finite-sample spatial integral estimates can exceed one. Keep individual
    # estimates visible, and mark their probability combination unavailable.
    if any(
        not math.isfinite(probability) or not 0.0 <= probability <= 1.0
        for probability in probabilities
    ):
        return None
    return independent_union_probability(probabilities)


def shared_space_mc(
    distributions: Sequence[Distribution2D],
    disks: Sequence[Disk],
    *,
    n_samples: int,
    seed: int,
    bounds: Rectangle | None = None,
) -> SharedMCResult:
    """Evaluate multiple disks using one shared uniform spatial sample batch.

    Each distribution belongs to one physical obstacle. Compute a separate
    disk integral per obstacle, then combine probabilities under the explicit
    physical independence assumption.
    A local Random instance makes calls reproducible without global RNG changes.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ValueError("n_samples must be a positive integer.")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if not disks:
        raise ValueError("At least one disk is required.")
    if not distributions:
        raise ValueError("At least one obstacle distribution is required.")
    if not all(
        isinstance(distribution, (Gaussian2D, GaussianMixture2D))
        for distribution in distributions
    ):
        raise TypeError("Unsupported obstacle distribution.")
    if bounds is None:
        bounds = covering_rectangle(disks)
    if not all(bounds.covers(disk) for disk in disks):
        raise ValueError("The shared rectangle must cover every complete disk.")

    rng = random.Random(seed)
    hits = [0 for _ in disks]
    sums = [[0.0 for _ in distributions] for _ in disks]
    for _ in range(n_samples):
        point = (
            bounds.xmin + (bounds.xmax - bounds.xmin) * rng.random(),
            bounds.ymin + (bounds.ymax - bounds.ymin) * rng.random(),
        )
        densities = tuple(distribution.density(point) for distribution in distributions)
        for index, disk in enumerate(disks):
            if disk.contains(point):
                hits[index] += 1
                for obstacle_index, density in enumerate(densities):
                    sums[index][obstacle_index] += density

    results = []
    rectangle_factor = bounds.area / n_samples
    for index, disk in enumerate(disks):
        rectangle_probs = tuple(rectangle_factor * value for value in sums[index])
        conditional_probs = None
        conditional_joint = None
        if hits[index] > 0:
            conditional_factor = disk.area / hits[index]
            conditional_probs = tuple(
                conditional_factor * value for value in sums[index]
            )
            conditional_joint = _joint_if_valid(conditional_probs)
        results.append(
            DiskMCResult(
                disk=disk,
                hit_count=hits[index],
                rectangle_probabilities=rectangle_probs,
                conditional_probabilities=conditional_probs,
                rectangle_joint=_joint_if_valid(rectangle_probs),
                conditional_joint=conditional_joint,
            )
        )
    return SharedMCResult(bounds, n_samples, seed, tuple(results))
