import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags, eye
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


N=128
x=np.linspace(0,22,N,endpoint=False)
t= np.linspace(0,300,60*300)
def u0_unpert(L, points=N):
    local_x = np.linspace(0, L, points, endpoint=False)
    return (0.5*np.cos(2*np.pi*local_x/L)
            + 0.2*np.sin(2*np.pi*local_x/L)
            + 0.1*np.cos(3*2*np.pi*local_x/L+0.4))

def u0_pert(L, points=N):
    local_x = np.linspace(0, L, points, endpoint=False)
    return u0_unpert(L, points) + 1e-4*np.cos(5*2*np.pi*local_x/L)

#u_t+uu_x+u_xx+u_xxxx=0
#u(x,0)=u0(x)
#u(x+L,t)=u(x,t)

U_22_u= u0_unpert(22)
U_6_u= u0_unpert(6)
U_22_p= u0_pert(22)


def method_of_lines(u0, t_final, dt, L=22):
    dx = L / len(u0)
    times = np.arange(0, t_final + dt / 2, dt)

    def rhs(current_time, u):
        first = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        second = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2
        fourth = (
            np.roll(u, 2) - 4 * np.roll(u, 1) + 6 * u
            - 4 * np.roll(u, -1) + np.roll(u, -2)
        ) / dx**4
        return -u * first - second - fourth

    solution = solve_ivp(
        rhs,
        (0, t_final),
        u0,
        method="BDF",
        t_eval=times,
        rtol=1e-6,
        atol=1e-8,
    )
    assert solution.success, solution.message
    return solution.y, solution.t


def scipy_differential_solution(u0, times, L=22, method="BDF"):
    """Solve the periodic KSE with SciPy and sparse differentiation matrices."""
    points = len(u0)
    dx = L / points
    shift = diags(
        [np.ones(points - 1), np.ones(1)], [1, 1 - points],
        shape=(points, points), format="csc"
    )
    first_derivative = (shift - shift.T) / (2 * dx)
    second_derivative = (
        shift + shift.T - 2 * eye(points, format="csc")
    ) / dx**2
    linear_part = -second_derivative - second_derivative @ second_derivative

    def rhs(current_time, values):
        return -(values * (first_derivative @ values)
                 + first_derivative @ (values**2)) / 3 + linear_part @ values

    def jacobian(current_time, values):
        return (
            -(diags(first_derivative @ values)
              + diags(values) @ first_derivative
              + 2 * first_derivative @ diags(values)) / 3
            + linear_part
        )

    solution = solve_ivp(
        rhs,
        (times[0], times[-1]),
        u0,
        method=method,
        jac=jacobian,
        t_eval=times,
        rtol=1e-7,
        atol=1e-9,
    )
    assert solution.success, solution.message
    return solution.y, solution.t


def compare_solutions(personal_solution, personal_times, scipy_solution,
                      scipy_times, L=22, show_plot=True):
    """Compare two solutions and return error metrics."""
    if personal_solution.shape[0] != scipy_solution.shape[0]:
        raise ValueError("Solutions must have the same spatial resolution")

    common_times = np.asarray(personal_times)
    scipy_on_common_grid = np.vstack([
        np.interp(common_times, scipy_times, row) for row in scipy_solution
    ])
    difference = personal_solution - scipy_on_common_grid
    dx = L / personal_solution.shape[0]
    personal_mass = dx * np.sum(personal_solution, axis=0)
    scipy_mass = dx * np.sum(scipy_on_common_grid, axis=0)
    personal_energy = dx * np.sum(personal_solution**2, axis=0) / 2
    scipy_energy = dx * np.sum(scipy_on_common_grid**2, axis=0) / 2
    metrics = {
        "max_solution_error": float(np.max(np.abs(difference))),
        "rms_solution_error": float(np.sqrt(np.mean(difference**2))),
        "max_mass_difference": float(np.max(np.abs(personal_mass - scipy_mass))),
        "max_energy_difference": float(
            np.max(np.abs(personal_energy - scipy_energy))
        ),
    }

    if show_plot:
        figure, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(common_times, np.max(np.abs(difference), axis=0))
        axes[0].set_title("Maximum solution difference")
        axes[1].plot(common_times, personal_mass, label="Personal")
        axes[1].plot(common_times, scipy_mass, "--", label="SciPy")
        axes[1].set_title("Mass")
        axes[1].legend()
        axes[2].plot(common_times, personal_energy, label="Personal")
        axes[2].plot(common_times, scipy_energy, "--", label="SciPy")
        axes[2].set_title("Energy")
        axes[2].legend()
        for axis in axes:
            axis.set_xlabel("Time")
            axis.grid()
        figure.tight_layout()
        plt.show()

    return metrics

U_22_u_sol, times = method_of_lines(U_22_u, 300, 0.1, L=22)
print("U_22_u solution computed")
U_6_u_sol, _ = method_of_lines(U_6_u, 300, 0.1, L=6)
print("U_6_u solution computed")
U_22_p_sol, _ = method_of_lines(U_22_p, 300, 0.1, L=22)
print("U_22_p solution computed")

U_22_u_scipy, _ = scipy_differential_solution(U_22_u, times, L=22)
U_6_u_scipy, _ = scipy_differential_solution(U_6_u, times, L=6)
U_22_p_scipy, _ = scipy_differential_solution(U_22_p, times, L=22)
print("SciPy solutions computed")

comparison_metrics_22_u = compare_solutions(
    U_22_u_sol, times, U_22_u_scipy, times, L=22, show_plot=True)
comparison_matrics_6_u = compare_solutions(
    U_6_u_sol, times, U_6_u_scipy, times, L=6, show_plot=True)
comparison_metrics_22_p = compare_solutions(
    U_22_p_sol, times, U_22_p_scipy, times, L=22, show_plot=True)
print("Comparison metrics computed")


print("Animation saved as personal_solution_KS.mp4")
print("The solution is saved as a video file in the current directory.")
# Open the file in append mode

def integration(u,dx):
    return np.sum(u)*dx

u_mass_22_u = np.zeros(len(times))
u_mass_6_u = np.zeros(len(times))
u_mass_22_p = np.zeros(len(times))

for i in range(len(times)):
    u_mass_22_u[i] = integration(U_22_u_scipy[:,i],22/N)
    u_mass_6_u[i] = integration(U_6_u_scipy[:,i],6/N)
    u_mass_22_p[i] = integration(U_22_p_scipy[:,i],22/N)
print("Mass of every solution computed")
U_energy_22_u = np.zeros(len(times))
U_energy_6_u = np.zeros(len(times))
U_energy_22_p = np.zeros(len(times))

for i in range(len(times)):
    U_energy_22_u[i] = integration(U_22_u_scipy[:,i]**2,22/N)/2
    U_energy_6_u[i] = integration(U_6_u_scipy[:,i]**2,6/N)/2
    U_energy_22_p[i] = integration(U_22_p_scipy[:,i]**2,22/N)/2
print("Energy of every solution computed")

def u_to_string(u):
    string = ""
    for i in range(len(times)):
        for j in range(len(x)):
            string += f"{u[j,i]:.6f} "
        string += "\n"
    return string


with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Time values:\n")
    f.write(" ".join(f"{t:.6f}" for t in times))
    f.write("\nU_22_u solution:\n")
    f.write(u_to_string(U_22_u_scipy))
    f.write("\nU_22_u mass over time:\n")
    f.write(" ".join(f"{m:.6f}" for m in u_mass_22_u))
    f.write("\nU_22_u energy over time:\n")
    f.write(" ".join(f"{e:.6f}" for e in U_energy_22_u))
    f.write("\nU_6_u solution:\n")
    f.write(u_to_string(U_6_u_scipy))
    f.write("\nU_6_u mass over time:\n")
    f.write(" ".join(f"{m:.6f}" for m in u_mass_6_u))
    f.write("\nU_6_u energy over time:\n")
    f.write(" ".join(f"{e:.6f}" for e in U_energy_6_u))
    f.write("\nU_22_p mass over time:\n")
    f.write(" ".join(f"{m:.6f}" for m in u_mass_22_p))
    f.write("\nU_22_p energy over time:\n")
    f.write(" ".join(f"{e:.6f}" for e in U_energy_22_p))
    f.write("\n")
    f.write("\nU_22_p solution:\n")
    f.write(u_to_string(U_22_p_scipy))
    
