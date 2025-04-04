#include <future>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <random>
// #include <future>
#include <fstream>
#include <thread>
#include <cstdlib> // for system()
using namespace std;
#define M_PI 3.14159265358979323846
// Type alias for a spin configuration: each spin is an array {Sx, Sy, Sz}.
using Spins = std::vector<std::array<double, 3>>;

// -------------------------------------------------------------------------
// Function: initialize_spins
// -------------------------------------------------------------------------
// Initialize the spins in a noisy antiferromagnetic configuration.
// Here we assign small
// random in-plane components (with amplitude scaled by delta_ini) and then
// compute the z component (with alternating sign) so that each spin is a unit vector.
Spins initialize_spins(int N, double theta, double noise_amp, std::mt19937 &rng)
{
    const double delta_ini = 0.1;
    Spins spins(N);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    // Distribution for noise in the range [-pi/100, +pi/100]
    std::uniform_real_distribution<double> noise_dist(-M_PI/100, M_PI/100);

    for (int j = 0; j < N; j++)
    {
        // Random azimuthal angle in [0, 2pi)
        double theta_random = 2 * M_PI * dist(rng);
        // Add noise between -pi/100 and +pi/100
        theta_random += noise_dist(rng);

        // Random amplitude for the in-plane component (scaled by delta_ini)
        double r_random = delta_ini * dist(rng);
        double Sx = r_random * std::cos(theta_random);
        double Sy = r_random * std::sin(theta_random);
        // Ensure the spin is normalized: Sx^2 + Sy^2 + Sz^2 = 1
        double Sz_magnitude = std::sqrt(1.0 - Sx * Sx - Sy * Sy);
        // Impose AFM order by alternating the sign of Sz
        int sign = (j % 2 == 0) ? 1 : -1;
        double Sz = sign * Sz_magnitude;
        spins[j] = {Sx, Sy, Sz};
    }
    return spins;
}


// -------------------------------------------------------------------------
// Function: compute_H_ave
// -------------------------------------------------------------------------
// Compute the time-averaged Hamiltonian:
//   H_ave = 0.5 * sum_j [ J * S_z[j]*S_z[j+1] + h * S_z[j] + g * S_x[j] ]
// with periodic boundary conditions.
double compute_H_ave(const Spins &spins, double J, double h, double g)
{
    int N = spins.size();
    double sum = 0.0;
    for (int j = 0; j < N; j++)
    {
        int next = (j + 1) % N; // periodic BC
        sum += J * spins[j][2] * spins[next][2] + h * spins[j][2] + g * spins[j][0];
    }
    return 0.5 * sum;
}

// -------------------------------------------------------------------------
// Function: rotate_z
// -------------------------------------------------------------------------
// Rotate a given spin about the z-axis by an angle phi.
void rotate_z(std::array<double, 3> &spin, double phi)
{
    double x_old = spin[0];
    double y_old = spin[1];
    spin[0] = x_old * std::cos(phi) - y_old * std::sin(phi);
    spin[1] = x_old * std::sin(phi) + y_old * std::cos(phi);
    // spin[2] remains unchanged
}

// -------------------------------------------------------------------------
// Function: rotate_x
// -------------------------------------------------------------------------
// Rotate a given spin about the x-axis by an angle theta.
void rotate_x(std::array<double, 3> &spin, double theta_angle)
{
    double y_old = spin[1];
    double z_old = spin[2];
    spin[1] = y_old * std::cos(theta_angle) - z_old * std::sin(theta_angle);
    spin[2] = y_old * std::sin(theta_angle) + z_old * std::cos(theta_angle);
    // spin[0] remains unchanged
}

// -------------------------------------------------------------------------
// Function: update_spins
// -------------------------------------------------------------------------
// Evolve the spin configuration for one full driving period T.
// The evolution is done in two half-steps:
// 1. A rotation about the z-axis by an angle angle1 = (κ * T)/2,
//    where κ depends on the neighboring spins’ z-components.
// 2. A rotation about the x-axis by an angle (g * T)/2.
void update_spins(Spins &spins, double J, double h, double g, double T)
{
    int N = spins.size();
    std::vector<double> kappa(N);
    // Compute the spin-dependent rotation frequency κ_j
    for (int j = 0; j < N; j++)
    {
        int prev = (j - 1 + N) % N;
        int next = (j + 1) % N;
        kappa[j] = J * (spins[prev][2] + spins[next][2]) + h;
    }
    double angle_x = (g * T) / 2.0;
    // Update each spin: first rotate about z, then about x.
    for (int j = 0; j < N; j++)
    {
        double angle1 = (kappa[j] * T) / 2.0;
        rotate_z(spins[j], angle1);
        rotate_x(spins[j], angle_x);
    }
}

// -------------------------------------------------------------------------
// Function: simulate_run
// -------------------------------------------------------------------------
// Simulate one run for L driving cycles.
// Returns a vector Q_vals such that Q_vals[l] = H(lT)
// (before normalization). A new RNG (seeded uniquely) is used.
std::vector<double> simulate_run(int L, int N, double J, double h, double g,
                                 double T, double theta, double noise_amp, unsigned int seed)
{
    std::mt19937 rng(seed);
    Spins spins = initialize_spins(N, theta, noise_amp, rng);
    std::vector<double> Q_vals(L + 1, 0.0);
    Q_vals[0] = 0.0; // By definition at t = 0

    for (int l = 1; l <= L; l++)
    {
        update_spins(spins, J, h, g, T);
        double H = compute_H_ave(spins, J, h, g);
        Q_vals[l] = H;
    }
    return Q_vals;
}

// -------------------------------------------------------------------------
// Function: ensemble_average
// -------------------------------------------------------------------------
// Compute the ensemble-averaged Q(lT) by running num_runs independent
// simulations in parallel and then averaging their Q_vals.
// After averaging, a normalization is applied so that Q = 0 when H = E0
// and Q = 1 when H = 0.
std::vector<double> ensemble_average(int num_runs, int L, int N, double J, double h,
                                     double g, double T, double theta, double noise_amp, double E0)
{
    std::vector<std::future<std::vector<double>>> futures;
    // Launch simulation runs in parallel using std::async.
    for (int run = 0; run < num_runs; run++)
    {
        // Seed for each run (using std::random_device combined with run index)
        unsigned int seed = std::random_device{}() + run;
        futures.push_back(std::async(std::launch::async, simulate_run, L, N, J, h, g, T, theta, noise_amp, seed));
    }
    // Initialize the ensemble average vector.
    std::vector<double> Q_ensemble(L + 1, 0.0);
    // Accumulate the results from each run.
    for (int run = 0; run < num_runs; run++)
    {
        std::vector<double> Q_run = futures[run].get();
        for (int l = 0; l <= L; l++)
        {
            Q_ensemble[l] += Q_run[l];
        }
    }
    // Average and then normalize: Q = (Q - E0) / (-E0)
    for (int l = 0; l <= L; l++)
    {
        Q_ensemble[l] /= num_runs;
        Q_ensemble[l] = (Q_ensemble[l] - E0) / (-E0);
    }
    return Q_ensemble;
}

// -------------------------------------------------------------------------
// Main function
// -------------------------------------------------------------------------
int main()
{
    // -- Simulation parameters --
    int N = 100;      // number of spins in the chain
    int L = 20000000; // number of driving cycles (you might reduce this for testing)
    double J = 1.0;
    double g = 0.809;
    double h = 0.7045;
    double Omega = 3.8;            // driving frequency (in units of J)
    double T = 2 * M_PI / Omega;   // period T = 2π/Omega
    double theta = M_PI / 4;       // parameter (not used in the current initialization)
    double noise_amp = M_PI / 100; // small noise amplitude
    int num_runs = 20;             // ensemble size

    // Initialize one spin configuration to compute the "ground state" energy E0.
    std::random_device rd;
    unsigned int seed0 = rd();
    std::mt19937 rng(seed0);
    Spins spins = initialize_spins(N, theta, noise_amp, rng);
    double E0 = compute_H_ave(spins, J, h, g);

    // Compute the ensemble-averaged Q(lT) using multithreading.
    std::vector<double> Q_avg = ensemble_average(num_runs, L, N, J, h, g, T, theta, noise_amp, E0);

    // Write the results to a file "results.txt"
    std::ofstream outfile("results.txt");
    if (!outfile)
    {
        std::cerr << "Error opening output file." << std::endl;
        return 1;
    }
    for (int l = 0; l <= L; l++)
    {
        outfile << l << " " << Q_avg[l] << "\n";
    }
    outfile.close();
    std::cout << "Simulation complete. Results saved to results.txt" << std::endl;

    // ---------------------
    // Plotting using Gnuplot
    // ---------------------
    //
    std::ofstream gp("plot.gp");
    if (!gp)
    {
        std::cerr << "Error opening Gnuplot script file." << std::endl;
        return 1;
    }
    gp << "set terminal pngcairo size 800,600 enhanced font 'Verdana,10'\n";
    gp << "set output 'plot.png'\n";
    gp << "set title 'Energy absorption in the driven spin chain'\n";
    gp << "set xlabel 'Driving cycles, l'\n";
    gp << "set ylabel 'Q(lT)'\n";
    gp << "set grid\n";
    gp << "plot 'results.txt' using 1:2 with lines title 'Q(lT)'\n";
    gp.close();

    // Call Gnuplot to display the plot.
    int ret = std::system("gnuplot plot.gp");
    if (ret != 0)
    {
        std::cerr << "Error: Gnuplot execution failed. Please ensure Gnuplot is installed." << std::endl;
    }
    else
    {
        std::cout << "Plot generated and saved as 'plot.png'." << std::endl;
    }

    return 0;
}
