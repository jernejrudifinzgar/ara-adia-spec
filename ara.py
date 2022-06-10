import qutip
from qutip import tensor, identity, basis, qzero, sesolve, sigmax, sigmay, ket, ket2dm, bell_state
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt


def self_consistency(m, lamb, s, c, gamma, p=3):
    """Self consistency equation for the magnetization m."""
    first = c * (s * p * m ** (p - 1) + (1 - s) * (1 - lamb))
    first /= np.sqrt(
        (s * p * m ** (p - 1) + (1 - s) * (1 - lamb)) ** 2
        +
        (gamma * (1 - s) * lamb) ** 2
    )
    second = (1 - c) * (s * p * m ** (p - 1) - (1 - s) * (1 - lamb))
    second /= np.sqrt(
        (s * p * m ** (p - 1) - (1 - s) * (1 - lamb)) ** 2
        +
        (gamma * (1 - s) * lamb) ** 2
    )

    return m - first - second


def free_energy(m, lamb, s, c, gamma, p=3):
    """Free energy function for the p-spin model."""
    tot = s * (p - 1) * m ** (p - 1)
    tot -= c * np.sqrt((s * p * m ** (p - 1) + (1 - s) * (1 - lamb)) ** 2 + (gamma * (1 - s) * lamb) ** 2)
    tot -= (1 - c) * np.sqrt((s * p * m ** (p - 1) - (1 - s) * (1 - lamb)) ** 2 + (gamma * (1 - s) * lamb) ** 2)

    return tot


def get_magnetization(lamb, s, c, gamma=1, x0s=np.linspace(-1, 1, 3), p=3):
    """Solves the self-consistency equation numerically and chooses the magnetization with the minimal
    free energy."""
    roots = fsolve(self_consistency, x0=x0s, args=(lamb, s, c, gamma, p))
    pos_min_root = np.argmin([free_energy(root, lamb, s, c, gamma, p) for root in roots])

    return roots[int(pos_min_root)]


def sigz(j):
    """Sigma-z operator in the reduced Hilbert space of conserved total angular momentum."""
    return qutip.Qobj(np.diag(np.arange(-j, j + 1))) / 2


def sigx(j):
    """Sigma-x operator in the reduced Hilbert space of conserved total angular momentum."""
    d = 2 * j + 1
    arr = np.zeros((d, d))

    for x, m in enumerate(np.arange(-j, j + 1)):
        if x + 1 < d:
            arr[x, x + 1] += np.sqrt(j * (j + 1) - m * (m + 1))
        if x - 1 >= 0:
            arr[x, x - 1] += np.sqrt(j * (j + 1) - m * (m - 1))

    return qutip.Qobj(arr) / 4


def pspin(n, c=0, p=3):
    """p-spin hamiltonian in the J=max subspace."""
    dim1, dim2 = round(n * c), n - round(n * c)
    return - n * (2 / n * (tensor(sigz(dim1), identity(2 * dim2 + 1)) +
                           tensor(identity(2 * dim1 + 1), sigz(dim2)))
                  ) ** p


def h_init(n, c):
    """Initial solution hamiltonian in the J=max subspace."""
    dim1, dim2 = round(n * c), n - round(n * c)
    return -2 * (tensor(sigz(dim1), identity(2 * dim2 + 1)) - tensor(identity(2 * dim1 + 1), sigz(dim2)))


def v_tf(n, c=0):
    """Transverse field hamiltonian in the J=max subspace"""
    dim1, dim2 = round(n * c), n - round(n * c)
    return -2 * (tensor(sigx(dim1), identity(2 * dim2 + 1)) + tensor(identity(2 * dim1 + 1), sigx(dim2)))


def h_ara(lamb, s, n, c, gamma=1, p=3):
    """Total ARA hamiltonian written in the J=max subspace."""
    return s * pspin(n, c, p) + (1 - s) * (1 - lamb) * h_init(n, c) + gamma * (1 - s) * lamb * v_tf(n, c)


def magnetization_op(j1, j2):
    """Magnetization operator for for the tensor product of two subspaces of total angular momenta."""
    mag_op = []
    m1, m2 = np.arange(-j1, j1 + 1), np.arange(-j2, j2 + 1)
    dim1, dim2 = 2 * j1 + 1, 2 * j2 + 1

    for i in range(dim1):
        for k in range(dim2):
            mag_op.append((m1[i] + m2[k]) / (j1 + j2))

    return np.array(mag_op)


def perform_ara(n, c, T, s=lambda t, _: t, lamb=lambda t, _: t, gamma=1, p=3, nsteps=100):
    """Performs ARA and returns the time dependence of the magnetization."""
    # total angular momenta and corresponding dimensions
    j1, j2 = round(n * c), n - round(n * c)
    dim1, dim2 = 2 * j1 + 1, 2 * j2 + 1
    # initial state
    psi0 = tensor(basis(dim1, dim1 - 1), basis(dim2, 0))
    # ARA hamiltonian coefficients
    h0_c = lambda t, _: s(t / T, _)
    hinit_c = lambda t, _: (1 - s(t / T, _)) * (1 - lamb(t / T, _))
    vtf_c = lambda t, _: gamma * (1 - s(t / T, _)) * lamb(t / T, _)

    # hamiltonian
    ham = [qzero([dim1, dim2]), [pspin(n, c, p), h0_c], [h_init(n, c), hinit_c], [v_tf(n, c), vtf_c]]

    # getting the magnetization operator
    mag_op = magnetization_op(j1, j2)

    # solve the Schrödinger eq. and reshape the result
    states = sesolve(ham, psi0, np.linspace(0, T, nsteps), progress_bar=True).states
    states = np.reshape(states, (nsteps, dim1 * dim2))
    # return the expectation value of the magnetization as a function of time (since mag_op is diagonal)
    return np.abs(states) ** 2 @ mag_op, states


def compute_eig(n, c, gamma=1, p=3, nsteps=100):
    """Computes the energy gap along a linear path."""
    times = np.linspace(0, 1, nsteps)
    gaps = []
    gss = []

    for t in times:
        evals, evecs = np.linalg.eigh(h_ara(t, t, n, c, gamma, p))
        gss.append(evecs[:, 0])
        gaps.append(np.diff(evals)[0])

    return np.array(gss), np.array(gaps)


def compute_overlaps(states1, states2):
    return np.abs(np.diag(states1 @ np.conj(states2.T)))


def get_adiaspec_coef(coef, probe, reverse_scale, t):
    forw = lambda x: coef(x, 0)
    back = lambda x: coef((probe * (1 + reverse_scale) - x) / reverse_scale, 0)
    theta = lambda x: int(x > probe)

    return (1 - theta(t)) * forw(t) + theta(t) * back(t)


def ara_spectroscopy(n, c, T_f, probe, T_r=1000, s=lambda t, _: t, lamb=lambda t, _: t, gamma=1, p=3, nsteps=100):
    """Estimate the overlap by slowly evolving backwards."""

    # total angular momenta and corresponding dimensions
    j1, j2 = round(n * c), n - round(n * c)
    dim1, dim2 = 2 * j1 + 1, 2 * j2 + 1
    # initial state
    psi0 = tensor(basis(dim1, dim1 - 1), basis(dim2, 0))
    # ARA hamiltonian coefficients
    rev_scale = T_r / T_f
    t_probe = probe * T_f
    T_final = (1 + rev_scale) * t_probe

    h0_c = lambda t, _: get_adiaspec_coef(s, probe, rev_scale, t / T_f)
    hinit_c = lambda t, _: get_adiaspec_coef(lambda x, _: (1 - s(x, _)) * (1 - lamb(x, _)),
                                             probe, rev_scale, t / T_f)
    vtf_c = lambda t, _: get_adiaspec_coef(lambda x, _: gamma * (1 - s(x, _)) * lamb(x, _),
                                           probe, rev_scale, t / T_f)
    # hamiltonian
    ham = [qzero([dim1, dim2]), [pspin(n, c, p), h0_c], [h_init(n, c), hinit_c], [v_tf(n, c), vtf_c]]
    times = np.hstack((
        np.linspace(0, t_probe, nsteps // 2, endpoint=False), np.linspace(t_probe, T_final, nsteps // 2, endpoint=True)
    ))
    # solve the schrödinger eq.
    fin_state = np.ravel(sesolve(ham, psi0, times, progress_bar=None).states[-1])
    return np.abs(np.ravel(psi0) @ np.conj(fin_state))


class AdiabaticEvolution:
    """Class for handling all things related to adiabatic spectroscopy."""
    def __init__(self, problem=None, mixer=None, n_chunks=1, T_init=10, interpolation='gervey'):
        # problem hamiltonian
        self.problem = problem
        # mixer hamiltonian H0
        self.mixer = mixer
        # chunks time spent in each equally long interval of s
        self.chunks = np.array([T_init / n_chunks] * n_chunks)
        # initial state
        self.psi0 = None
        # interpolation function specification (either gervey or linear)
        self.interpolation = interpolation

        # hard-coded parameters
        self.step_density = 10
        self.bisection_tolerance = 0.01

    @property
    def n_chunks(self):
        """Number of chunks of the current instance."""
        return len(self.chunks)

    @property
    def schedule(self):
        """Translated from chunks into actual times at which a certain chunk is active."""
        return np.pad(np.cumsum(self.chunks), (1, 0))

    def set_problem(self, problem):
        self.problem = problem

    def set_mixer(self, mixer):
        self.mixer = mixer
        self.psi0 = self.mixer.groundstate()[1]

    def t_to_s(self, t):
        """Translates the time into the s coordinate."""
        if t > self.schedule[-1]:
            return 1
        elif t == 0:
            # ugly code :-(
            return 0
        else:
            where = np.max(np.where(t > self.schedule))
            s = ((t - self.schedule[where]) / (self.schedule[where + 1] - self.schedule[where]) + where) / self.n_chunks
            return s

    def s_to_t(self, s):
        """Returns the time of a given s value."""
        if s == 0:
            return 0
        else:
            where = np.max(np.where(s > np.arange(self.n_chunks + 1) / self.n_chunks))
            tm = self.schedule[where]
            tp = self.schedule[where + 1]
            return tm + (tp - tm) * (s * self.n_chunks - where)

    def get_coefficients(self):
        """Helper function that provides the qutip format coefficients for the ODE solver with time-dependent
        Hamiltonians."""
        def _gervey(s, lm, lp):
            return lm + (lp - lm) * np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * s) ** 2) ** 2

        def _linear(s, lm, lp):
            return lm + (lp - lm) * s

        def _coefficient(t, schedule, chunks, interpolation):
            if t > schedule[-1]:
                return 1
            elif t == 0:
                # ugly code :-(
                return 0
            else:
                where = np.max(np.where(t > schedule))
                s = (t - schedule[where]) / chunks[where]
                # print(chunks[where])
                lm = where / len(chunks)
                lp = (where + 1) / len(chunks)

                if interpolation == 'linear':
                    return _linear(s, lm, lp)
                elif interpolation == 'gervey':
                    return _gervey(s, lm, lp)
                else:
                    raise Exception('Interpolation function not defined.')

        a = lambda t, _: _coefficient(t, self.schedule, self.chunks, self.interpolation)
        b = lambda t, _: 1 - _coefficient(t, self.schedule, self.chunks, self.interpolation)

        return a, b

    def perform_evolution(self, s):
        """Performs the evolution according to the time-dependent Schrödinger equation, using an ODE solver."""
        a, b = self.get_coefficients()
        ham = [qzero(self.problem.dims[0]), [self.problem, a], [self.mixer, b]]

        if self.psi0 is None:
            self.psi0 = self.mixer.groundstate()[1]

        t_final = self.s_to_t(s)
        fin_state = sesolve(ham, self.psi0,
                            np.linspace(0, t_final, 1 + int(t_final / self.step_density))
                            ).states[-1]
        return fin_state

    def instant_ham(self, s):
        """Helper function that returns the instantaneous Hamiltonian at s."""
        t = self.s_to_t(s)
        a, b = self.get_coefficients()

        return a(t, 0) * self.problem + b(t, 0) * self.mixer

    def get_overlap(self, s):
        """Computes the overlap of the evolved state with the instantaneous ground state."""
        if s == 0:
            return 1
        else:
            # compute evolved state
            fin_state = self.perform_evolution(s)

            # compute instantaneous ground state
            instant_ham = self.instant_ham(s)
            instant_gs = instant_ham.groundstate()[1]

            # return overlap
            return np.abs(instant_gs.overlap(fin_state))

    def update_chunk(self, chunk, new_length):
        """Helper function to update the chunk lengths in an adaptive algorithm."""
        tmp = self.chunks.copy()
        tmp[chunk] = new_length
        self.chunks = tmp

    def target_overlap(self, target, interpolate_overlaps=True, verbose=True):
        """This method takes a target overlap and attempts to find the minimal time that the final ground state
        overlap is larger than the target. Choose interpolate_overlaps if you want the requirement to be interpolated
        between 1 and the final value along the adiabatic evolution."""
        if interpolate_overlaps:
            targets = np.linspace(1, target, self.n_chunks + 1)[1:]
        else:
            targets = [target] * self.n_chunks

        for which in range(self.n_chunks):
            s = (which + 1) / self.n_chunks
            chunk = self.chunks[which]
            ch = chunk
            while self.get_overlap(s) < targets[which]:
                # find the starting chunk-length
                ch *= 2
                self.update_chunk(which, ch)
                # print(ch)
            print(f'Satisfactory overlap found for {which + 1} (Target: {targets[which]:.5f}). Begin optimization:')
            # optimize via bisection:
            if ch > chunk:
                ch_m = ch / 2
                ch_p = ch
            else:
                ch_m = 0
                ch_p = ch
            step = 0
            while abs(ch_m - ch_p) > self.bisection_tolerance * chunk / 2:
                step += 1
                ch = np.mean([ch_m, ch_p])
                self.update_chunk(which, ch)
                overlap = self.get_overlap(s)
                if verbose:
                    print(f"Bisection step: {step}. "
                          f"Overlap: {overlap:.2f} "
                          f"Chunk: {ch}")
                # print(self.chunks)
                if overlap < targets[which]:
                    ch_m = ch
                else:
                    ch_p = ch
            self.update_chunk(which, ch_p)

            print(f"Chunk {which + 1} tuned.")

    def get_gaps(self, resolution=100):
        """Compute the spectral gap by exact (numerical) diagonalization."""
        ss = np.linspace(0, 1, resolution)
        gaps = []

        for s in ss:
            ham = self.instant_ham(s)
            # print(ham.gro)
            gap = np.diff(np.sort(ham.eigenenergies()))[0]
            gaps.append(gap)
        return gaps

    def visualize_chunks(self, gaps=False):
        """Function to visualize the current distribution of total time into chunks."""
        fig, ax = plt.subplots()

        if gaps:
            gs = self.get_gaps()
            ts = [self.s_to_t(s) for s in np.linspace(0, 1, len(gs))]
            ax.plot(ts, gs, 'k--')
            ax.set_ylabel('Gap')
        else:
            ax.set_yticks([])

        for i in range(self.n_chunks):
            ax.axvspan(self.schedule[i], self.schedule[i + 1],
                       color=plt.rcParams['axes.prop_cycle'].by_key()['color'][i])
        ax.set_xlim(min(self.schedule), max(self.schedule))
        ax.set_title(f'Number of chunks: {self.n_chunks}')
        ax.set_xlabel('Time')

        # top axis:
        ax_top = ax.twiny()

        ax_top.set_xticks(self.schedule)
        ax_top.set_xticklabels((np.arange(self.n_chunks + 1) / self.n_chunks).round(2))
        ax_top.set_xlabel('$s$')

        return fig, ax

    def get_overlap_reverse(self, s, reverse_time):
        """Implementation of the forwards-backwards protocol for ground-state overlap estimation."""
        a, b = self.get_coefficients()

        turnaround_time = self.s_to_t(s)

        def _reverse_coefficient(coef, t_turn):
            theta = lambda t: np.heaviside(t - t_turn, 0)
            back_slope = (coef(0, 0) - coef(t_turn, 0)) / (s * reverse_time)

            return lambda t, _: (1 - theta(t)) * coef(t, _) - theta(t) * (back_slope * (t - t_turn) + coef(t_turn, 0))

        ar, br = _reverse_coefficient(a, turnaround_time), _reverse_coefficient(b, turnaround_time)

        t_final = turnaround_time + s * reverse_time
        ham = [qzero(self.problem.dims[0]), [self.problem, ar], [self.mixer, br]]
        fin_state = sesolve(ham, self.psi0,
                            np.linspace(0, t_final, 1 + int(t_final / self.step_density))
                            ).states[-1]

        return np.abs(fin_state.overlap(self.psi0))

    def get_overlap_ancilla(self, s, n_shots=100, tau_interval=(30, 200), use_esq=False):
        """Implementation of the single-ancilla protocol for ground-state overlap estimation."""
        evolved_state = self.perform_evolution(s)

        state = tensor((ket('0') + ket('1')) / np.sqrt(2), evolved_state)

        instant_ham = self.instant_ham(s)

        hamiltonian = tensor(ket2dm(ket('1')), instant_ham)
        overlap_op = tensor(sigmax() - 1j * sigmay(), identity(self.problem.dims[0]))

        alphas = []

        for shot in range(n_shots):
            # draw a random guess in the tau_interval
            tau = tau_interval[0] + np.random.rand() * (tau_interval[1] - tau_interval[0])
            alpha = sesolve(hamiltonian, state,
                            np.linspace(0, tau, 1 + int(tau / self.step_density)),
                            e_ops=[overlap_op]).expect[0][-1]
            alphas.append(alpha)

        e_squared = np.mean(np.abs(alphas) ** 2)

        if e_squared > 0.5 and use_esq:
            # print('e_sq')
            return np.sqrt(0.5 + 0.5 * np.sqrt(2 * e_squared - 1))
        else:
            return np.mean(np.abs(alphas))

    def get_overlap_entangled(self, s, n_shots=100, tau_interval=(30, 200), use_esq=False):
        """Implementation of the entangled-ancillae protocol for ground-state overlap estimation."""

        evolved_state = self.perform_evolution(s)

        state = tensor((ket('0') + ket('1')) / np.sqrt(2), evolved_state)

        instant_ham = self.instant_ham(s)

        # backwards and forwards controlled hamiltonian
        ham1 = tensor(ket2dm(ket('1')), instant_ham)
        ham2 = tensor(ket2dm(ket('1')), -instant_ham)

        alphas = []

        for shot in range(n_shots):
            # draw a random guess in the tau_interval
            tau = tau_interval[0] + np.random.rand() * (tau_interval[1] - tau_interval[0])
            ts = np.linspace(0, tau, 1 + int(tau / self.step_density))
            s1 = sesolve(ham1, state, ts).states[-1]
            s2 = sesolve(ham2, state, ts).states[-1]
            final_state = tensor(s1, s2)
            # final_state = tensor([sesolve(ham, state, ts).states[-1] for ham in (ham1, ham2)])
        # return final_state
            rho_ancilla = final_state.ptrace([0, 3])
            alpha_sq = 1 - 4 * rho_ancilla.overlap(bell_state('01'))
            alphas.append(alpha_sq)
        # print(alphas)
        e_squared = np.mean(alphas)

        if e_squared > 0.5 and use_esq:
            # print('e_sq')
            return np.sqrt(0.5 + 0.5 * np.sqrt(2 * e_squared - 1))
        else:
            return np.mean(np.sqrt(np.abs(alphas)))
