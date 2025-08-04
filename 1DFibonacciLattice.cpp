#include<iostream>
#include<vector>
#include<algorithm>
#include<random>
#include<cmath>
#include<fstream>
#include<thread>
#include<mutex>
#include<chrono>
#include<iomanip>
#include<complex>
#include"Eigen/Dense"


class D1FibonacciLattice
{
private:
	static std::mutex logMutex;
public:
	double precision = 0.01;
	double p_precision = 0.01;
	double difference = 1.0;
	double t_ab;
	double t_aa;
	double mu0;
	double T; // tempreture
	double k_b = 1; //1.38e-23
	double g; // interaction factor
	std::vector<bool> lattice; // true if A-type, false if B-type
	std::vector<double> prefix_sum; // time sum
	std::vector<double> E; // energy levels
	std::vector<double> epsilon;
	std::vector<double> disorder;
	std::vector<double> delta;
	std::vector<std::vector<double>> u; // electron-like parts
	std::vector<std::vector<double>> v; // hole-like parts

	D1FibonacciLattice(double _t_aa, double _t_ab, double _mu0, double _T, double _g, unsigned _steps, double _disorder = 0.0) : t_ab(_t_ab), t_aa(_t_aa), mu0(_mu0), T(_T), g(_g)
	{
		this->generate_lattice(_steps);
		this->calculate_prefix_sum();
		this->disorder = std::vector<double>(this->lattice.size(), _disorder);
		this->set_initital_guess(0.0, 0.1, 0.1);
	}

	void set_initital_guess(double _E, double _u, double _v)
	{
		this->E = std::vector<double>(this->lattice.size(), _E);
		this->u = std::vector<std::vector<double>>(this->lattice.size(), std::vector<double>(this->lattice.size(), _u));
		this->v = std::vector<std::vector<double>>(this->lattice.size(), std::vector<double>(this->lattice.size(), _v));
		this->delta = this->calculate_delta();
		this->epsilon = std::vector<double>(this->lattice.size(), 0.0);
		this->calculate_epsilon();
	}

	void save_delta(std::string&& file)
	{
		std::ofstream out;
		out.open(file, std::ios::app);
		for (double& x : this->delta)
		{
			out << x << " ";
		}
		out << std::endl;
	}

	void log(std::string file)
	{
		std::lock_guard<std::mutex> lock(this->logMutex);
		auto now = std::chrono::system_clock::now();
		std::time_t now_time = std::chrono::system_clock::to_time_t(now);
		std::tm now_tm;
		localtime_s(&now_tm, &now_time);
		std::ofstream out;
		out.open(file, std::ios::app);
		out << "Date: " << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S") << std::endl;
		out << "T: " << this->T << " mu0: "  << this->mu0 << " t_a: " << this->t_aa << " t_b: " << this->t_ab << " g: " << this->g << " N: " << this->lattice.size() << " n: " << this->particle_number() / this->lattice.size() << "\n";
		out << "Disorder" << "\n";
		for (double& d : this->disorder)
		{
			out << d << " ";
		}
		out << "\n";
		out << "Energy" << "\n";
		for (double& x : this->E)
		{
			out << x << " ";
		}
		out << "\n";
		out << "Delta" << "\n";
		for (double& x : this->delta)
		{
			out << x << " ";
		}
		out << "\n";
		out << "U_components" << "\n";
		for (std::vector<double>& u_vec : this->u)
		{
			for (double& x : u_vec)
			{
				out << x << " ";
			}
			out << "\n";
		}
		out << "V_components" << "\n";
		for (std::vector<double>& v_vec : this->v)
		{
			for (double& x : v_vec)
			{
				out << x << " ";
			}
			out << "\n";
		}
		out << "\n";
		out.close();
	}

	double fermi(int n)
	{
		if (this->T == 0) return 0;
		else return 1.0 / (1.0 + exp(E[n] / (k_b * T)));
	}

	double average_n(int i)
	{
		double sum = 0;
		int N = this->lattice.size();
		for (int n = 0; n < N; ++n)
		{
			sum += v[i][n] * v[i][n] * (1 - fermi(n)) + u[i][n] * u[i][n] * fermi(n);
		}
		return 2 * sum;
	}

	double dirack_delta(double x)
	{
		double e = 1e-3 * this->delta[0];
		return e / (x * x + e * e) / 3.1459;
	}

	double density_of_states(double w)
	{
		double ans = 0;
		for (unsigned n = 0; n < this->lattice.size(); ++n)
		{
			for (unsigned i = 0; i < this->lattice.size(); ++i)
			{
				ans += u[n][i] * u[n][i] * dirack_delta(w - E[n]) + v[n][i] * v[n][i] * dirack_delta(w + E[n]);
			}
		}
		return ans;
	}

	double particle_number()
	{
		double sum = 0;
		int N = this->lattice.size();
		for (int i = 0; i < N; ++i)
		{
			sum += this->average_n(i);
		}
		return sum;
	}

	double norm(std::vector<double>& x, std::vector<double>& y)
	{
		double sum = 0;
		for (int i = 0; i < x.size(); ++i)
		{
			sum += ((x[i] - y[i]) * (x[i] - y[i]));
		}
		return sqrt(sum);
	}

	double self_norm(std::vector<double>& x)
	{
		double sum = 0;
		for (int i = 0; i < x.size(); ++i)
		{
			sum += (x[i] * x[i]);
		}
		return sqrt(sum);
	}

	void print_E()
	{
		for (double& x : this->E)
		{
			std::cout << x << " ";
		}
		std::cout << std::endl;
	}

	void print_u()
	{
		for (std::vector<double>& vec : this->u)
		{
			for (double& x : vec)
			{
				std::cout << x << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	void print_v()
	{
		for (std::vector<double>& vec : this->v)
		{
			for (double& x : vec)
			{
				std::cout << x << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	void calculate_prefix_sum()
	{
		if (this->lattice.empty()) _exception;
		std::vector<double> result;
		result.push_back(0);
		for (int i = 1; i < this->lattice.size(); ++i)
		{
			result.push_back(result[i - 1] + (this->lattice[i - 1] ? this->t_aa : this->t_ab));
		}
		this->prefix_sum = result;
	}

	void calculate_epsilon()
	{
		for (int i = 0; i < this->lattice.size(); ++i)
		{
			this->epsilon[i] = this->disorder[i] - this->mu0 - (this->g * this->average_n(i) / 2);
		}
	}

	std::vector<double> calculate_delta()
	{
		int N = this->lattice.size();
		std::vector<double> res(N, 0.0);
		for (int i = 0; i < N; ++i)
		{
			double result = 0;
			for (int n = 0; n < N; ++n)
			{
				result += u[i][n] * v[i][n] * (1.0 - 2.0 * this->fermi(n));
			}
			res[i] = this->g * result;
		}
		return res;
	}

	void generate_lattice(unsigned steps_number)
	{
		std::vector<bool> previous = { true };
		for (unsigned i = 1; i < steps_number; ++i)
		{
			std::vector<bool> current;
			for (bool A_type : previous)
			{
				if (A_type)
				{
					current.push_back(true);
					current.push_back(false);
				}
				else
				{
					current.push_back(true);
				}
			}
			previous = current;
		}
		this->lattice = previous;
	}

	std::vector<std::vector<double>> generate_Matrix()
	{
		if (this->lattice.empty()) _exception;
		int N = this->lattice.size();
		std::vector<std::vector<double>> M(2 * N, std::vector<double>(2 * N, 0.0));
		for (int i = 0; i < N; ++i)
		{
			M[i][N + i] = this->delta[i];
			if (i - 1 >= 0) M[i][i - 1] = this->prefix_sum[i] - this->prefix_sum[i - 1];
			if (i + 1 < N) M[i][i + 1] = this->prefix_sum[i + 1] - this->prefix_sum[i];
			M[i][i] = this->epsilon[i];
		}
		for (int i = N; i < 2 * N; ++i)
		{
			M[i][i - N] = this->delta[i - N];
			if (i - 1 >= N) M[i][i - 1] = -(this->prefix_sum[i - N] - this->prefix_sum[i - 1 - N]);
			if (i + 1 < 2 * N) M[i][i + 1] = -(this->prefix_sum[i + 1 - N] - this->prefix_sum[i - N]);
			M[i][i] = -(this->epsilon[i - N]);
		}
		return M;
	}

	void Cycle()
	{
		if (this->lattice.empty()) _exception;
		int N = this->lattice.size();
		std::vector<std::vector<double>> M = this->generate_Matrix();
		Eigen::MatrixXd matrix(M[0].size(), M.size());
		for (int i = 0; i < M[0].size(); ++i)
		{
			for (int j = 0; j < M.size(); ++j)
			{
				matrix(i, j) = M[i][j];
			}
		}
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(matrix);
		int idx = 0;
		for (int i = 0; i < 2 * N; ++i)
		{
			if (es.eigenvalues()(i) > 0)
			{
				this->E[idx] = es.eigenvalues()(i);
				for (int j = 0; j < N; ++j)
				{
					this->u[j][idx] = es.eigenvectors()(j, i);
				}
				for (int j = N; j < 2 * N; ++j)
				{
					this->v[j - N][idx] = es.eigenvectors()(j, i);
				}
				++idx;
			}
		}
		this->calculate_epsilon();
		std::vector<double> new_delta = this->calculate_delta();
		this->difference = norm(new_delta, this->delta) / self_norm(new_delta);
		this->delta = new_delta;
	}

	void solveBdG()
	{
		int iterations = 0;
		while (this->difference > this->precision)
		{
			this->Cycle();
			++iterations;
			if (iterations > 50)
			{
				std::cout << "MAX ITERATIONS OUT" << std::endl;
				break;
			}
		}
		this->difference = 1.0;
	}

	void disorderBdG(double concentration = -228.0)
	{
		double max_mu = 4 * this->t_aa;
		double min_mu = -1 * max_mu;
		double n0 = 0;
		if (concentration == -228.0)
		{
			std::vector<double> copy_disorder = this->disorder;
			this->disorder = std::vector<double>(this->lattice.size(), 0.0);
			this->set_initital_guess(0.0, 0.1, 0.1);
			this->solveBdG();
			n0 = this->particle_number();
			this->disorder = copy_disorder;
		}
		else
		{
			n0 = concentration * this->lattice.size();
		}
		this->set_initital_guess(0.0, 0.1, 0.1);
		this->solveBdG();
		double n = this->particle_number();
		if (abs(n - n0) / n0 < this->p_precision) return;
		while (abs(n - n0) / n0 > this->p_precision)
		{
			if (n > n0) max_mu = this->mu0;
			else min_mu = this->mu0;
			this->mu0 = (max_mu + min_mu) / 2;
			this->set_initital_guess(0.0, 0.1, 0.1);
			this->solveBdG();
			n = this->particle_number();
		}
	}

};

std::mutex D1FibonacciLattice::logMutex;

std::vector<double> correlation(std::vector<double> nums)
{
	std::vector<double> ans(nums.size(), 0.0);
	for (unsigned k = 0; k < nums.size(); ++k)
	{
		double sum = 0.0;
		unsigned i = 0;
		while (i + k < nums.size())
		{
			sum += nums[i] * nums[i + k];
			++i;
		}
		if (i != 0) ans[k] = sum / static_cast<double>(i);
	}
	return ans;
}


std::vector<double> calculateAverageDeltaCorrelation(double v)
{
	std::vector<double> ans(233, 0.0);
	int iterations = 25;
	for (int it = 0; it < iterations; ++it)
	{
		std::string notify = "Disorder param: " + std::to_string(v) + ", iteration: " + std::to_string(it + 1) + " \n";
		std::cout << notify;
		std::random_device rd;
		std::mt19937 generator(rd());
		std::uniform_real_distribution<double> dist(-v, v);
		D1FibonacciLattice test(0.8, 1.0, -1.0, 0.0, 1.5, 12);
		std::vector<double> dis(test.lattice.size(), 0.0);
		for (unsigned i = 0; i < dis.size(); ++i)
		{
			dis[i] = dist(generator);
		}
		test.disorder = dis;
		test.disorderBdG(0.875);
		std::vector<double> corr = correlation(test.delta);
		for (unsigned j = 0; j < test.lattice.size(); ++j)
		{
			ans[j] += corr[j];
		}
	}
	for (unsigned j = 0; j < ans.size(); ++j)
	{
		ans[j] /= static_cast<double>(iterations);
	}
	return ans;
}


void calculateData(double v, int it, std::vector<std::vector<double>>& disorder_data, std::vector<std::vector<double>>& delta_data, std::vector<std::vector<double>>& e_data, std::vector<std::vector<std::vector<double>>>& u_data, std::vector<std::vector<std::vector<double>>>& v_data)
{
	std::string notify = "Disorder param: " + std::to_string(v) + ", iteration: " + std::to_string(it + 1) + " \n";
	std::cout << notify;
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<double> dist(-v, v);
	D1FibonacciLattice test(0.8, 1.0, -1.0, 0.0, 1.5, 12);
	std::vector<double> dis(test.lattice.size(), 0.0);
	for (unsigned i = 0; i < dis.size(); ++i)
	{
		dis[i] = dist(generator);
		disorder_data[it][i] = dis[i];
	}
	test.disorder = dis;
	test.disorderBdG(0.875);
	for (unsigned i = 0; i < 233; ++i)
	{
		delta_data[it][i] = test.delta[i];
		e_data[it][i] = test.E[i];
	}
	for (unsigned n = 0; n < 233; ++n)
	{
		for (unsigned i = 0; i < 233; ++i) 
		{
			u_data[it][n][i] = test.u[n][i];
			v_data[it][n][i] = test.v[n][i];
		}
	}
}

void calculateTotalData(double v, std::vector<std::vector<double>>& disorder_data, std::vector<std::vector<double>>& delta_data, std::vector<std::vector<double>>& e_data, std::vector<std::vector<std::vector<double>>>& u_data, std::vector<std::vector<std::vector<double>>>& v_data)
{
	int iterations = 50;
	for (int it = 0; it < iterations; ++it)
	{
		calculateData(v, it, disorder_data, delta_data, e_data, u_data, v_data);
	}
}

void coherenceCalculation(double v, int it, std::vector<std::vector<double>>& diff)
{
	std::string notify = "Disorder param: " + std::to_string(v) + ", iteration: " + std::to_string(it + 1) + " \n";
	std::cout << notify;
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_real_distribution<double> dist(-v, v);
	D1FibonacciLattice test(0.8, 1.0, -1.0, 0.0, 1.5, 12);
	std::vector<double> dis(test.lattice.size(), 0.0);
	for (unsigned i = 0; i < dis.size(); ++i)
	{
		dis[i] = dist(generator);
	}
	test.disorder = dis;
	test.disorderBdG(0.875);
	std::vector<double> delta_prev = test.delta;
	test.disorder[115] += 2.5;
	test.disorderBdG(0.875);
	std::vector<double> delta_curr = test.delta;
	for (unsigned i = 0; i < test.lattice.size(); ++i)
	{
		diff[it][i] = abs(delta_prev[i] - delta_curr[i]);
	}
}

void coherenceTotalCalculation(double v, std::vector<std::vector<double>>& diff)
{
	int iterations = 50;
	for (int it = 0; it < iterations; ++it)
	{
		coherenceCalculation(v, it, diff);
	}
}

int main()
{
	std::vector<double> disorderParams = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
	std::vector<std::vector<std::vector<double>>> diff_data(10, std::vector<std::vector<double>>(50, std::vector<double>(233, 0.0)));
	std::vector<std::thread> threads;
	for (unsigned p = 0; p < 10; ++p)
	{
		threads.emplace_back([p, &disorderParams, &diff_data]() 
			{
				coherenceTotalCalculation(disorderParams[p], diff_data[p]);
			});
	}
	for (auto& thread : threads)
	{
		thread.join();
	}
	std::ofstream out;
	out.open("DeltaDiffCoherenceCalculation.txt");
	for (unsigned p = 0; p < 10; ++p)
	{
		for (unsigned it = 0; it < 50; ++it)
		{
			for (unsigned i = 0; i < 233; ++i)
			{
				out << diff_data[p][it][i] << " ";
			}
			out << "\n";
		}
		out << "\n";
	}
	return 0;
}