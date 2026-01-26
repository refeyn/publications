import csv
import dataclasses
import matplotlib.pyplot as plt
import pathlib
from typing import Final, NamedTuple, TypeAlias, Sequence
import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.integrate import solve_ivp

FloatArray: TypeAlias = np.ndarray[tuple[int], np.dtype[np.float64]]

PD1_KDA: Final[float] = 30.0
IVONESICMAB_KDA: Final[float] = 210.0


def gaussian(
    x: FloatArray,
    amplitude: float,
    position: float,
    sigma: float,
) -> FloatArray:
    res = abs(amplitude) * np.exp((-((x - position) ** 2)) / (2.0 * sigma**2))
    return res


def gaussian_integral(amplitude: float, sigma: float) -> float:
    return amplitude * sigma * np.sqrt(2 * np.pi)


def multi_gaussians(
    x: FloatArray,
    amplitudes: Sequence[float],
    positions: Sequence[float],
    sigma: float,
) -> FloatArray:
    res = np.sum(
        [
            gaussian(x=x, amplitude=_a0, position=_x0, sigma=sigma)
            for _a0, _x0 in zip(amplitudes, positions)
        ],
        axis=0,
    )
    return res


class IvonescimabComplexFits(NamedTuple):
    ivonescimab_amplitude: float
    ivonescimab_pd1_amplitude: float
    ivonescimab_2pd1_amplitude: float
    shared_sigma: float
    mass_multiplier: float


def fit_ivonescimab_complexes(
    masses: FloatArray, counts: FloatArray
) -> tuple[IvonescimabComplexFits, FloatArray]:
    known_masses = np.array(
        [IVONESICMAB_KDA, IVONESICMAB_KDA + PD1_KDA, IVONESICMAB_KDA + (2 * PD1_KDA)]
    )

    def f(x: FloatArray, parameters: IvonescimabComplexFits) -> FloatArray:
        return multi_gaussians(
            x,
            amplitudes=[
                parameters.ivonescimab_amplitude,
                parameters.ivonescimab_pd1_amplitude,
                parameters.ivonescimab_2pd1_amplitude,
            ],
            positions=list(known_masses * parameters.mass_multiplier),
            sigma=parameters.shared_sigma,
        )

    def sum_of_squared_residuals(parameters: IvonescimabComplexFits) -> float:
        return float(((counts - f(x=masses, parameters=parameters)) ** 2).sum())

    amplitude_guesses: list[float] = [
        counts[np.argmin(abs(masses - mass))] for mass in known_masses
    ]
    amplitude_bounds = [(None, None) for _ in known_masses]

    shared_sigma_guess = [8.0]
    shared_sigma_bounds = [(7, 14)]

    mass_multiplier_guess = [1.0]
    mass_multiplier_bounds = [(0.95, 1.05)]

    initial_guesses = amplitude_guesses + shared_sigma_guess + mass_multiplier_guess
    bounds = amplitude_bounds + shared_sigma_bounds + mass_multiplier_bounds

    fit_result = minimize(
        lambda p: sum_of_squared_residuals(IvonescimabComplexFits(*p)),
        initial_guesses,
        bounds=bounds,
    )
    fitted_params = IvonescimabComplexFits(*fit_result.x)

    fitted_summed_gaussian = f(x=masses, parameters=fitted_params)

    return fitted_params, fitted_summed_gaussian


def determine_complex_abundance_from_csv(
    path: pathlib.Path, show_fits: bool = False
) -> tuple[float, float, float]:
    with path.open() as file:
        reader = csv.DictReader(file)
        masses = np.array([float(row["calibrated_values"]) for row in reader])

    bins = np.linspace(180, 300, 60)
    counts, bin_edges = np.histogram(masses, bins=bins)
    bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2
    fit, yest = fit_ivonescimab_complexes(
        bin_mid,
        counts,
    )

    x = np.linspace(bins[0], bins[-1], 1000)
    if show_fits:
        plt.hist(masses, bins=bins)
        plt.plot(bin_mid, yest)

        plt.plot(
            x,
            gaussian(
                x,
                amplitude=fit.ivonescimab_amplitude,
                position=IVONESICMAB_KDA * fit.mass_multiplier,
                sigma=fit.shared_sigma,
            ),
            color="black",
        )
        plt.plot(
            x,
            gaussian(
                x,
                amplitude=fit.ivonescimab_pd1_amplitude,
                position=(IVONESICMAB_KDA + PD1_KDA) * fit.mass_multiplier,
                sigma=fit.shared_sigma,
            ),
            color="black",
        )
        plt.plot(
            x,
            gaussian(
                x,
                amplitude=fit.ivonescimab_2pd1_amplitude,
                position=(IVONESICMAB_KDA + (2 * PD1_KDA)) * fit.mass_multiplier,
                sigma=fit.shared_sigma,
            ),
            color="black",
        )
        plt.show()

    return (
        gaussian_integral(
            amplitude=fit.ivonescimab_amplitude,
            sigma=fit.shared_sigma,
        ),
        gaussian_integral(
            amplitude=fit.ivonescimab_pd1_amplitude,
            sigma=fit.shared_sigma,
        ),
        gaussian_integral(
            amplitude=fit.ivonescimab_2pd1_amplitude,
            sigma=fit.shared_sigma,
        ),
    )


@dataclasses.dataclass(frozen=True)
class KineticFit:
    kon1: float
    koff1: float
    kon2: float
    koff2: float
    kd1: float
    kd2: float
    kon1_err: float
    koff1_err: float
    kon2_err: float
    koff2_err: float
    kd1_err: float
    kd2_err: float
    result: OptimizeResult


if __name__ == "__main__":
    IVONESCIMAB_CONCENTRATION_NM = 5.0
    PD1_CONCENTRATION_NM = 20.0

    raw_data_location = pathlib.Path("put_data_folder_here")

    data_paths = list(raw_data_location.glob("*.csv"))

    times = np.zeros(len(data_paths))
    iv_counts = np.zeros(len(data_paths))
    iv_pd1_counts = np.zeros(len(data_paths))
    iv_2pd1_counts = np.zeros(len(data_paths))
    repeat_index = np.zeros(len(data_paths))

    for index, data_path in enumerate(data_paths):
        _, repeat_num, time, _ = data_path.name.split("_")
        times[index] = float(time)
        repeat_index[index] = int(repeat_num)
        iv_counts[index], iv_pd1_counts[index], iv_2pd1_counts[index] = (
            determine_complex_abundance_from_csv(data_path)
        )
    # 60s measurement, use halfway through as time.
    times += 30

    time_sort = np.argsort(times)

    times = times[time_sort]
    repeat_index = repeat_index[time_sort]
    iv_counts = iv_counts[time_sort]
    iv_pd1_counts = iv_pd1_counts[time_sort]
    iv_2pd1_counts = iv_2pd1_counts[time_sort]

    unique_times = np.sort(np.unique(times))
    time_lookup = np.array([np.argwhere(unique_times == time)[0][0] for time in times])

    iv_total_counts = iv_counts + iv_pd1_counts + iv_2pd1_counts
    conc_scale = IVONESCIMAB_CONCENTRATION_NM / iv_total_counts

    iv_conc = iv_counts * conc_scale
    iv_pd1_conc = iv_pd1_counts * conc_scale
    iv_2pd1_conc = iv_2pd1_counts * conc_scale

    def fit_pd1_monomer_addition(initial_pd1_conc: float) -> KineticFit:
        data = np.vstack(
            [
                initial_pd1_conc - iv_pd1_conc - (2 * iv_2pd1_conc),
                iv_conc,
                iv_pd1_conc,
                iv_2pd1_conc,
            ]
        )

        def odes(
            _t: float,
            y: FloatArray,
            kon1: float,
            koff1: float,
            kon2: float,
            koff2: float,
        ) -> FloatArray:
            pd1, iv, iv_pd1, iv_2pd2 = y
            r1_on = kon1 * iv * pd1
            r1_off = koff1 * iv_pd1
            r2_on = kon2 * iv_pd1 * pd1
            r2_off = koff2 * iv_2pd2

            dA = -r1_on + r1_off
            dP = -r1_on + r1_off - r2_on + r2_off
            dAP = r1_on - r1_off - r2_on + r2_off
            dAP2 = r2_on - r2_off

            return np.array([dP, dA, dAP, dAP2], dtype=np.float64)

        def simulate(
            kon1: float, koff1: float, kon2: float, koff2: float, t_eval=unique_times
        ) -> FloatArray:
            starting_conc = np.array(
                [PD1_CONCENTRATION_NM, IVONESCIMAB_CONCENTRATION_NM, 0.0, 0.0]
            )
            solution = solve_ivp(
                odes,
                (0, t_eval[-1] + 1),
                starting_conc,
                t_eval=t_eval,
                args=(kon1, koff1, kon2, koff2),
                method="RK45",
            )

            return solution.y

        def raw_residuals(params: FloatArray) -> np.ndarray:
            model = np.vstack(simulate(*params))
            true_model = model[:, time_lookup]
            return (true_model - data).ravel()

        def squared_sum_residual(params: FloatArray) -> float:
            r = raw_residuals(params)
            return np.sum(r**2)

        x0 = np.array(
            [
                0.0006748528217521699,
                0.00019865004914931816,
                0.00038596920201050544,
                0.00019865004914931816,
            ],
            dtype=np.float64,
        )
        bounds = [(1e-8, 0.1)] * 4

        result = minimize(squared_sum_residual, x0=x0, bounds=bounds)
        params = result.x

        def jacobian(
            params: FloatArray, epsilon: float = 1e-8
        ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
            residuals = raw_residuals(params)
            n_params = len(params)
            n_res = residuals.size
            J = np.zeros((n_res, n_params), dtype=float)
            for i in range(n_params):
                delta = np.zeros_like(params)
                delta[i] = epsilon
                J[:, i] = (raw_residuals(params + delta) - residuals) / epsilon
            return J

        dof = data.size - len(params)
        J = jacobian(params)
        H_approx = J.T @ J
        cov = np.linalg.inv(H_approx)
        sigma2 = np.sum(raw_residuals(params) ** 2) / dof
        cov *= sigma2
        errors = np.sqrt(np.diag(cov))

        kon1, koff1, kon2, koff2 = params
        kon1_err, koff1_err, kon2_err, koff2_err = errors
        kd1 = koff1 / kon1
        kd2 = koff2 / kon2

        def kd_error(koff, kon, cov, idx_off, idx_on):
            df_doff = 1 / kon
            df_don = -koff / kon**2
            return np.sqrt(
                df_doff**2 * cov[idx_off, idx_off]
                + df_don**2 * cov[idx_on, idx_on]
                + 2 * df_doff * df_don * cov[idx_off, idx_on]
            )

        kd1_err = kd_error(koff1, kon1, cov, 1, 0)
        kd2_err = kd_error(koff2, kon2, cov, 3, 2)

        def plot_fit(fit: KineticFit):
            t_eval = np.linspace(0, max(unique_times), 1000)
            P, A, AP, AP2 = simulate(
                fit.kon1, fit.koff1, fit.kon2, fit.koff2, t_eval=t_eval
            )

            plt.figure(figsize=(7, 7), dpi=100)
            markers = np.array(["o", "s", "^"])
            for index in (0, 1, 2):
                sel = repeat_index == index
                plt.scatter(
                    times[sel],
                    (initial_pd1_conc - iv_pd1_conc - (2 * iv_2pd1_conc))[sel],
                    label="PD-1 (mass-balanced)",
                    color="tab:blue",
                    marker=markers[index],
                )
                plt.scatter(
                    times[sel],
                    iv_conc[sel],
                    label="Ivonescimab",
                    color="tab:orange",
                    marker=markers[index],
                )
                plt.scatter(
                    times[sel],
                    iv_pd1_conc[sel],
                    label="Ivonescimab-PD-1",
                    color="tab:green",
                    marker=markers[index],
                )
                plt.scatter(
                    times[sel],
                    iv_2pd1_conc[sel],
                    label="Ivonescimab-2PD-1",
                    color="tab:purple",
                    marker=markers[index],
                )
            plt.plot(t_eval, P, color="tab:blue")
            plt.plot(t_eval, A, color="tab:orange")
            plt.plot(t_eval, AP, color="tab:green")
            plt.plot(t_eval, AP2, color="tab:purple")
            plt.title(
                f"kon1 = {fit.kon1:.2e} ± {fit.kon1_err:.2e}\n"
                f"koff1 = {fit.koff1:.2e} ± {fit.koff1_err:.2e}\n"
                f"kon2 = {fit.kon2:.2e} ± {fit.kon2_err:.2e}\n"
                f"koff2 = {fit.koff2:.2e} ± {fit.koff2_err:.2e}\n"
                f"kd1 = {fit.kd1:.2e} ± {fit.kd1_err:.2e}\n"
                f"kd2 = {fit.kd2:.2e} ± {fit.kd2_err:.2e}"
            )
            plt.xlabel("Time")
            plt.ylabel("Concentration")
            plt.legend()
            plt.tight_layout()
            plt.show()

        fit = KineticFit(
            kon1=kon1,
            koff1=koff1,
            kon2=kon2,
            koff2=koff2,
            kd1=kd1,
            kd2=kd2,
            kon1_err=kon1_err,
            koff1_err=koff1_err,
            kon2_err=kon2_err,
            koff2_err=koff2_err,
            kd1_err=kd1_err,
            kd2_err=kd2_err,
            result=result,
        )

        plot_fit(fit)
        return fit

    fit = fit_pd1_monomer_addition(PD1_CONCENTRATION_NM)
    print(fit)
