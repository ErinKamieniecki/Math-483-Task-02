import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


def read_section(lines, header):
    """Read numeric lines following a named output section."""
    start = lines.index(header) + 1
    end = start
    while end < len(lines) and not lines[end].endswith(":"):
        end += 1
    values = [float(value) for line in lines[start:end] for value in line.split()]
    return np.array(values)


with open("output.txt", "r", encoding="utf-8") as file:
    lines = [line.strip() for line in file]

Time = read_section(lines, "Time values:")
U_22_u = read_section(lines, "U_22_u solution:").reshape(len(Time), 128).T
U_mass_22_u = read_section(lines, "U_22_u mass over time:")
U_energy_22_u = read_section(lines, "U_22_u energy over time:")
U_6_u = read_section(lines, "U_6_u solution:").reshape(len(Time), 128).T
U_mass_6_u = read_section(lines, "U_6_u mass over time:")
U_energy_6_u = read_section(lines, "U_6_u energy over time:")
U_mass_22_p = read_section(lines, "U_22_p mass over time:")
U_energy_22_p = read_section(lines, "U_22_p energy over time:")
U_22_p = read_section(lines, "U_22_p solution:").reshape(len(Time), 128).T




# spatial_index = np.arange(U_22_u.shape[0])
# figure, axis = plt.subplots(figsize=(10, 6))
# unperturbed_line, = axis.plot(
#     spatial_index, U_22_u[:, 0], label="Unperturbed", color="blue"
# )
# perturbed_line, = axis.plot(
#     spatial_index, U_22_p[:, 0], label="Perturbed", color="orange"
# )
# axis.set_xlim(spatial_index.min(), spatial_index.max())
# axis.set_ylim(
#     min(U_22_u.min(), U_22_p.min()),
#     max(U_22_u.max(), U_22_p.max()),
# )
# axis.set_xlabel("Spatial Index")
# axis.set_ylabel("Solution Value")
# axis.legend()
# axis.grid()

# def update(frame):
#     unperturbed_line.set_ydata(U_22_u[:, frame])
#     perturbed_line.set_ydata(U_22_p[:, frame])
#     axis.set_title(f"KSE solutions at t = {Time[frame]:.1f}")
#     return unperturbed_line, perturbed_line

# animation = FuncAnimation(
#     figure,
#     update,
#     frames=len(Time),
#     interval=33,
#     blit=True,
# )
# animation.save("Compairison_KSE.mp4", writer=FFMpegWriter(fps=30))
# plt.show()

#plot of energy over time
plt.figure(figsize=(10, 6))
plt.plot(Time, U_energy_22_u, label="Unperturbed Energy", color="blue")
plt.plot(Time, U_energy_22_p, label="Perturbed Energy", color="orange")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy of KSE Solutions Over Time")
plt.legend()
plt.grid()
plt.savefig("./Graphics/Energy_Comparison_KSE.png")
plt.show()

#plot energy over time for L=22 and L=6
plt.figure(figsize=(10, 6))
plt.plot(Time, U_energy_22_u, label="L=22 Unperturbed Energy", color="blue")
plt.plot(Time, U_energy_6_u, label="L=6 Unperturbed Energy", color="green")
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy of KSE Solutions Over Time for Different L")
plt.legend()
plt.grid()
plt.savefig("./Graphics/Energy_Comparison_L22_L6.png")
plt.show()

# Compare the L=22 and L=6 solutions at four times.
x_22 = np.linspace(0, 22, U_22_u.shape[0], endpoint=False)
x_6 = np.linspace(0, 6, U_6_u.shape[0], endpoint=False)
comparison_times = [0, 100, 200, 300]
figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)

for axis, comparison_time in zip(axes.flat, comparison_times):
    index = np.argmin(np.abs(Time - comparison_time))
    axis.plot(x_22, U_22_u[:, index], label="L=22", color="blue")
    axis.plot(x_6, U_6_u[:, index], label="L=6", color="green")
    axis.set_title(f"t = {Time[index]:.1f}")
    axis.set_xlabel("x")
    axis.set_ylabel("U(x,t)")
    axis.grid()
    axis.legend()

figure.suptitle("KSE Solutions for L=22 and L=6")
figure.tight_layout()
figure.savefig("./Graphics/Solution_Comparison_L22_L6.png")
plt.show()

#Show the unperturbed and perturbed solutions at four times.
figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
for axis, comparison_time in zip(axes.flat, comparison_times):
    index = np.argmin(np.abs(Time - comparison_time))
    axis.plot(x_22, U_22_u[:, index], label="Unperturbed", color="blue")
    axis.plot(x_22, U_22_p[:, index], label="Perturbed", color="orange")
    axis.set_title(f"t = {Time[index]:.1f}")
    axis.set_xlabel("x")
    axis.set_ylabel("U(x,t)")
    axis.grid()
    axis.legend()
figure.suptitle("KSE Solutions for Unperturbed and Perturbed Cases")
figure.tight_layout()
figure.savefig("./Graphics/Solution_Comparison_Unperturbed_Perturbed.png")
plt.show()

#Caluclating RME between L=22 unperterbed and perterbed solutions on log scale.
difference = np.abs(U_22_u - U_22_p)
rme = np.sqrt(np.mean(difference**2, axis=0))
plt.figure(figsize=(10, 6))
plt.plot(Time, rme, color="purple")
plt.yscale("log")
plt.xlabel("Time")
plt.ylabel("Root Mean Square Error (RME)")
plt.title("RME between Unperturbed and Perturbed Solutions Over Time")
plt.grid()
plt.savefig("./Graphics/RME_Unperturbed_Perturbed.png")
plt.show()

def integrage_energy(U, Time):
    """Integrate the energy of the solution U over time."""
    dx = 22 / U.shape[0]  # Assuming L=22 for this function
    energy = dx * np.sum(U**2, axis=0) / 2
    return energy
