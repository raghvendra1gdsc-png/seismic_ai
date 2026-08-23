"""Newmark-beta direct time-history integration solver for MDOF systems.

Solves the dynamic equations of motion:
    M * u_ddot(t) + C * u_dot(t) + K * u(t) = P(t)
    where P(t) = -M * r * a_g(t) for earthquake ground excitation.
"""

from typing import Optional, Union, Literal, Tuple
import numpy as np

from src.structural.building import ShearBuilding
from src.dynamics.damping import RayleighDamping
from src.dynamics.response import DynamicResponse


class NewmarkSolver:
    """Step-by-step Newmark-beta time-history integration solver.

    Supports:
    - Average Acceleration Method (gamma=0.5, beta=0.25): unconditionally stable.
    - Linear Acceleration Method (gamma=0.5, beta=1/6): conditionally stable.
    - Custom (gamma, beta) integration parameters.

    Parameters
    ----------
    building : ShearBuilding
        The structural building model providing M, K, and r.
    damping : RayleighDamping | np.ndarray
        The damping model providing the C matrix (N x N), or a direct damping matrix.
    gamma : float, default=0.5
        Newmark velocity parameter gamma (gamma >= 0.5 for stability/no numerical damping).
    beta : float, default=0.25
        Newmark displacement parameter beta (beta=0.25 for average acceleration).
    """

    def __init__(
        self,
        building: ShearBuilding,
        damping: Union[RayleighDamping, np.ndarray],
        gamma: float = 0.5,
        beta: float = 0.25,
    ) -> None:
        if not isinstance(building, ShearBuilding):
            raise TypeError(f"Expected ShearBuilding instance, got {type(building).__name__}")

        self.building = building
        self.num_storeys: int = building.num_storeys

        if isinstance(damping, RayleighDamping):
            self.damping_matrix = damping.damping_matrix
        elif isinstance(damping, np.ndarray):
            if damping.shape != (self.num_storeys, self.num_storeys):
                raise ValueError(
                    f"Damping matrix shape mismatch: expected ({self.num_storeys}, {self.num_storeys}), "
                    f"got {damping.shape}"
                )
            self.damping_matrix = np.asarray(damping, dtype=np.float64)
        else:
            raise TypeError(f"Expected RayleighDamping or np.ndarray, got {type(damping).__name__}")

        if gamma < 0.5:
            raise ValueError(f"Newmark gamma must be >= 0.5 for numerical stability, got {gamma}")
        if beta <= 0.0:
            raise ValueError(f"Newmark beta must be strictly positive, got {beta}")

        self.gamma: float = float(gamma)
        self.beta: float = float(beta)

        self._m_mat = self.building.mass_matrix
        self._k_mat = self.building.stiffness_matrix
        self._r_vec = self.building.influence_vector
        self._m_diag = np.diag(self._m_mat)

    @classmethod
    def average_acceleration(
        cls,
        building: ShearBuilding,
        damping: Union[RayleighDamping, np.ndarray],
    ) -> "NewmarkSolver":
        """Factory for the unconditionally stable Average Acceleration Method (gamma=0.5, beta=0.25)."""
        return cls(building=building, damping=damping, gamma=0.5, beta=0.25)

    @classmethod
    def linear_acceleration(
        cls,
        building: ShearBuilding,
        damping: Union[RayleighDamping, np.ndarray],
    ) -> "NewmarkSolver":
        """Factory for the Linear Acceleration Method (gamma=0.5, beta=1/6)."""
        return cls(building=building, damping=damping, gamma=0.5, beta=1.0 / 6.0)

    def solve(
        self,
        ground_acceleration: Optional[np.ndarray] = None,
        external_force: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        time: Optional[np.ndarray] = None,
        initial_displacement: Optional[np.ndarray] = None,
        initial_velocity: Optional[np.ndarray] = None,
    ) -> DynamicResponse:
        """Run step-by-step dynamic integration over the time domain.

        Parameters
        ----------
        ground_acceleration : Optional[np.ndarray]
            1D array of ground accelerations a_g(t) in m/s^2.
        external_force : Optional[np.ndarray]
            Array of external dynamic forces P(t) of shape (N, N_t) in Newtons.
        dt : Optional[float]
            Time step in seconds. Required if time array is not supplied.
        time : Optional[np.ndarray]
            Explicit 1D time vector. If provided, dt = time[1] - time[0].
        initial_displacement : Optional[np.ndarray]
            Initial relative displacement vector u_0 of length N in meters (default 0).
        initial_velocity : Optional[np.ndarray]
            Initial relative velocity vector u_dot_0 of length N in m/s (default 0).

        Returns
        -------
        DynamicResponse
            Complete time-history response container.
        """
        # Determine time vector and step count
        if time is not None:
            t_vec = np.asarray(time, dtype=np.float64).flatten()
            num_steps = t_vec.size
            if num_steps < 2:
                raise ValueError("Time vector must have at least 2 points.")
            dt_val = float(t_vec[1] - t_vec[0])
        elif dt is not None:
            dt_val = float(dt)
            if dt_val <= 0.0:
                raise ValueError(f"Time step dt must be positive, got {dt_val}")

            if ground_acceleration is not None:
                num_steps = np.asarray(ground_acceleration).size
            elif external_force is not None:
                num_steps = np.asarray(external_force).shape[1]
            else:
                raise ValueError("Must provide either ground_acceleration or external_force when time is not given.")
            t_vec = np.arange(num_steps, dtype=np.float64) * dt_val
        else:
            raise ValueError("Either dt or time vector must be provided.")

        n = self.num_storeys

        # Construct total load history P(t) of shape (N, num_steps)
        p_mat = np.zeros((n, num_steps), dtype=np.float64)

        if ground_acceleration is not None:
            ag_arr = np.asarray(ground_acceleration, dtype=np.float64).flatten()
            if ag_arr.size != num_steps:
                raise ValueError(
                    f"Ground acceleration size ({ag_arr.size}) does not match time steps ({num_steps})."
                )
            # P_ground(t) = -M * r * a_g(t) = - diag(m) * [1,...,1]^T * a_g(t) = - m_i * a_g(t)
            m_r = self._m_diag[:, np.newaxis]  # shape (N, 1)
            p_mat += -m_r * ag_arr[np.newaxis, :]
        else:
            ag_arr = np.zeros(num_steps, dtype=np.float64)

        if external_force is not None:
            f_arr = np.asarray(external_force, dtype=np.float64)
            if f_arr.shape != (n, num_steps):
                raise ValueError(
                    f"External force shape mismatch: expected ({n}, {num_steps}), got {f_arr.shape}"
                )
            p_mat += f_arr

        # Initial state
        if initial_displacement is not None:
            u0 = np.asarray(initial_displacement, dtype=np.float64).flatten()
            if u0.size != n:
                raise ValueError(f"initial_displacement size mismatch: expected {n}, got {u0.size}")
        else:
            u0 = np.zeros(n, dtype=np.float64)

        if initial_velocity is not None:
            v0 = np.asarray(initial_velocity, dtype=np.float64).flatten()
            if v0.size != n:
                raise ValueError(f"initial_velocity size mismatch: expected {n}, got {v0.size}")
        else:
            v0 = np.zeros(n, dtype=np.float64)

        # Compute initial acceleration from dynamic equilibrium:
        # a0 = M^{-1} * (P_0 - C * v0 - K * u0)
        p0 = p_mat[:, 0]
        f_damping_0 = self.damping_matrix @ v0
        f_elastic_0 = self._k_mat @ u0
        a0 = (p0 - f_damping_0 - f_elastic_0) / self._m_diag

        # Newmark integration constants
        gamma = self.gamma
        beta = self.beta

        a_0 = 1.0 / (beta * dt_val**2)
        a_1 = gamma / (beta * dt_val)
        a_2 = 1.0 / (beta * dt_val)
        a_3 = 1.0 / (2.0 * beta) - 1.0
        a_4 = gamma / beta - 1.0
        a_5 = dt_val * (gamma / (2.0 * beta) - 1.0)
        a_6 = dt_val * (1.0 - gamma)
        a_7 = gamma * dt_val

        # Effective dynamic stiffness: K_hat = K + a_0 * M + a_1 * C
        k_hat = self._k_mat + a_0 * self._m_mat + a_1 * self.damping_matrix

        # Allocate response matrices (N, num_steps)
        u_all = np.zeros((n, num_steps), dtype=np.float64)
        v_all = np.zeros((n, num_steps), dtype=np.float64)
        a_all = np.zeros((n, num_steps), dtype=np.float64)

        u_all[:, 0] = u0
        v_all[:, 0] = v0
        a_all[:, 0] = a0

        # Step-by-step time integration
        u_curr = u0.copy()
        v_curr = v0.copy()
        a_curr = a0.copy()

        for i in range(num_steps - 1):
            p_next = p_mat[:, i + 1]

            # Effective dynamic load:
            # P_hat_{i+1} = P_{i+1} + M*(a0*u + a2*v + a3*a) + C*(a1*u + a4*v + a5*a)
            m_term = self._m_diag * (a_0 * u_curr + a_2 * v_curr + a_3 * a_curr)
            c_term = self.damping_matrix @ (a_1 * u_curr + a_4 * v_curr + a_5 * a_curr)
            p_hat = p_next + m_term + c_term

            # Solve linear system K_hat * u_{i+1} = P_hat
            u_next = np.linalg.solve(k_hat, p_hat)

            # Acceleration and velocity update
            a_next = a_0 * (u_next - u_curr) - a_2 * v_curr - a_3 * a_curr
            v_next = v_curr + a_6 * a_curr + a_7 * a_next

            # Store in output matrices
            u_all[:, i + 1] = u_next
            v_all[:, i + 1] = v_next
            a_all[:, i + 1] = a_next

            # Advance state
            u_curr = u_next
            v_curr = v_next
            a_curr = a_next

        return DynamicResponse(
            building=self.building,
            time=t_vec,
            displacement=u_all,
            velocity=v_all,
            acceleration=a_all,
            ground_acceleration=ag_arr if ground_acceleration is not None else None,
        )
