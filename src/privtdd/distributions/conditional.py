"""Direct degree-conditioned integer laws shared by the distribute version."""
from dataclasses import dataclass, field
import math
import numpy as np


def normalize_row(keys, mass, degree):
    """Normalize an actual generated row, without inferring a new coupling."""
    keys, mass = np.asarray(keys), np.asarray(mass, dtype=float)
    capacity = int(degree)*(int(degree)-1)//2
    if (keys.ndim != 1 or mass.shape != keys.shape or not len(keys)
        or keys.dtype.kind not in 'iu' or np.any(keys < 0) or np.any(keys > capacity)
        or not np.isfinite(mass).all() or np.any(mass < 0)):
        raise ValueError('Invalid generated conditional integer law')
    unique, inverse = np.unique(keys.astype(np.int64), return_inverse=True)
    probability = np.bincount(inverse, weights=mass, minlength=len(unique))
    active = probability > 0
    unique, probability = unique[active], probability[active]
    total = float(probability.sum())
    if not total > 0 or not np.isfinite(total):
        raise ValueError('A conditional law has no finite positive mass')
    probability /= total
    unique.setflags(write=False)
    probability.setflags(write=False)
    return unique, probability


def validate_degree_mass(mass, support):
    mass = np.asarray(mass, dtype=float)
    if (mass.shape != np.asarray(support).shape or not np.isfinite(mass).all()
        or np.any(mass < 0) or abs(float(mass.sum())-1) > 2e-10):
        raise ValueError('Invalid population degree marginal')
    return mass/mass.sum()


@dataclass(frozen=True)
class ConditionalDistribution:
    """R(t|d), before its population degree marginal is integrated out.

    native_degree_mass records the source's own degree marginal for diagnosis.
    Population output always receives its degree marginal explicitly.
    """
    name: str
    support: np.ndarray
    keys_by_degree: tuple
    probability_by_degree: tuple
    native_degree_mass: np.ndarray
    audit: dict = field(default_factory=dict)

    def __post_init__(self):
        support = np.asarray(self.support)
        if (support.ndim != 1 or not len(support) or support.dtype.kind not in 'iu'
            or np.any(support < 0) or np.any(np.diff(support.astype(np.int64)) <= 0)
            or len(self.keys_by_degree) != len(support)
            or len(self.probability_by_degree) != len(support)):
            raise ValueError('Invalid degree support or conditional row count')
        support = support.astype(np.int64, copy=True)
        rows = [normalize_row(k, p, d) for k, p, d in zip(
            self.keys_by_degree, self.probability_by_degree, support)]
        native_mass = validate_degree_mass(self.native_degree_mass, support).copy()
        support.setflags(write=False)
        native_mass.setflags(write=False)
        object.__setattr__(self, 'support', support)
        object.__setattr__(self, 'keys_by_degree', tuple(row[0] for row in rows))
        object.__setattr__(self, 'probability_by_degree', tuple(row[1] for row in rows))
        object.__setattr__(self, 'native_degree_mass', native_mass)

    def row(self, degree_index):
        return self.keys_by_degree[degree_index], self.probability_by_degree[degree_index]

    def fixed_degree_joint(self, degree_mass):
        """J(d,t)=pi(d)R(t|d); retain the generated conditional relationship."""
        degree_mass = validate_degree_mass(degree_mass, self.support)
        return tuple((keys, pi*probability) for keys, probability, pi in zip(
            self.keys_by_degree, self.probability_by_degree, degree_mass))

    def population_histogram(self, degree_mass, node_num):
        """Final degree integration; no lift or reconstruction from a GDD."""
        degree_mass = validate_degree_mass(degree_mass, self.support)
        if not math.isfinite(node_num) or node_num <= 0:
            raise ValueError('Population size must be finite and positive')
        histogram = {}
        for keys, probability, pi in zip(
            self.keys_by_degree, self.probability_by_degree, degree_mass):
            if pi == 0:
                continue
            for key, value in zip(keys, node_num*pi*probability):
                histogram[int(key)] = histogram.get(int(key), 0.)+float(value)
        if abs(math.fsum(histogram.values())-node_num) > 2e-10*max(1., node_num):
            raise ArithmeticError('Final degree integration lost population mass')
        return histogram
