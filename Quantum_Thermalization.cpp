#include <future>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <random>
#include <fstream>
#include <thread>
#include <cstdlib>

#define M_PI 3.14159265358979323846
using Spins = std::vector<std::array<double, 3>>;

Spins initialize_spins(int N, double theta, double noise_amp, std::mt19937 &rng)
{
    const double delta_ini = 0.1;
    Spins spins(N);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int j = 0; j < N; j++)
    {
        double theta_random = 2 * M_PI * dist(rng);
        double r_random = delta_ini * dist(rng);
        double Sx = r_random * std::cos(theta_random);
        double Sy = r_random * std::sin(theta_random);
        double Sz_magnitude = std::sqrt(1.0 - Sx * Sx - Sy * Sy);
        int sign = (j % 2 == 0) ? 1 : -1;
        double Sz = sign * Sz_magnitude;
        spins[j] = {Sx, Sy, Sz};
    }
    return spins;
}

double compute_H_ave(const Spins &spins, double J, double h, double g)
{
    int N = spins.size();
    double sum = 0.0;
    for (int j = 0; j < N; j++)
    {
        int next = (j + 1) % N;
        sum += J * spins[j][2] * spins[next][2] + h * spins[j][2] + g * spins[j][0];
    }
    return 0.5 * sum;
}

void rotate_z(std::array<double, 3> &spin, double phi)
{
    double x_old = spin[0];
    double y_old = spin[1];
    spin[0] = x_old * std::cos(phi) - y_old * std::sin(phi);
    spin[1] = x_old * std::sin(phi) + y_old * std::cos(phi);
}

void rotate_x(std::array<double, 3> &spin, double theta_angle)
{
    double y_old = spin[1];
    double z_old = spin[2];
    spin[1] = y_old * std::cos(theta_angle) - z_old * std::sin(theta_angle);
    spin[2] = y_old * std::sin(theta_angle) + z_old * std::cos(theta_angle);
}

void update_spins(Spins &spins, double J, double h, double g, double T)
{
    int N = spins.size();
    std::vector<double> kappa(N);
    for (int j = 0; j < N; j++)
    {
        int prev = (j - 1 + N) % N;
        int next = (j + 1) % N;
        kappa[j] = J * (spins[prev][2] + spins[next][2]) + h;
    }
    double angle_x = (g * T) / 2.0;
    for (int j = 0; j < N; j++)
    {
        double angle1 = (kappa[j] * T) / 2.0;
        rotate_z(spins[j], angle1);
        rotate_x(spins[j], angle_x);
    }
}

std::vector<double> simulate_run(int L, int N, double J, double h, double g,
                                 double T, double theta, double noise_amp, unsigned int seed)
{
    std::mt19937 rng(seed);
    Spins spins = initialize_spins(N, theta, noise_amp, rng);
    std::vector<double> Q_vals(L + 1, 0.0);
    Q_vals[0] = 0.0;
    for (int l = 1; l <= L; l++)
    {
        update_spins(spins, J, h, g, T);
        double H = compute_H_ave(spins, J, h, g);
        Q_vals[l] = H;
    }
    return Q_vals;
}

std::vector<double> ensemble_average(int num_runs, int L, int N, double J, double h,
                                     double g, double T, double theta, double noise_amp, double E0)
{
    std::vector<std::future<std::vector<double>>> futures;
    for (int run = 0; run < num_runs; run++)
    {
        unsigned int seed = std::random_device{}() + run;
        futures.push_back(std::async(std::launch::async, simulate_run, L, N, J, h, g, T, theta, noise_amp, seed));
    }
    std::vector<double> Q_ensemble(L + 1, 0.0);
    for (int run = 0; run < num_runs; run++)
    {
        std::vector<double> Q_run = futures[run].get();
        for (int l = 0; l <= L; l++)
        {
            Q_ensemble[l] += Q_run[l];
        }
    }
    for (int l = 0; l <= L; l++)
    {
        Q_ensemble[l] /= num_runs;
        Q_ensemble[l] = (Q_ensemble[l] - E0) / (-E0);
    }
    return Q_ensemble;
}

int main()
{
    int N = 100;
    int L = 20000000;
    double J = 1.0;
    double g = 0.809;
    double h = 0.7045;
    double Omega = 3.8;
    double T = 2 * M_PI / Omega;
    double theta = M_PI / 4;
    double noise_amp = M_PI / 100;
    int num_runs = 20;

    std::random_device rd;
    unsigned int seed0 = rd();
    std::mt19937 rng(seed0);
    Spins spins = initialize_spins(N, theta, noise_amp, rng);
    double E0 = compute_H_ave(spins, J, h, g);

    std::vector<double> Q_avg = ensemble_average(num_runs, L, N, J, h, g, T, theta, noise_amp, E0);

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
