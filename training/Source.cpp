#include <array>
#include <iostream>
#include <omp.h>
#include <random>
#include <stdio.h>
#include <tuple>
#include <algorithm>
#include <functional>


double lambda = 0.5;
double n_o = 0.2;
double n_w = 2.0;
//double S_eps = DBL_MIN;
double S_eps = 1e-32;
const int size = 4;
double RTOL = 1e-10;
double ATOL = 1e-10;
int MAXSTEPS = 2000;
int NPOINTS = 1000000;
double diff_eps = 0.0001;
double diff_exp = 1.0; //2
const double Ng = 0.0;
const double M = 0.4;

size_t total_steps = 0;

std::tuple<double, double> Pcow_fun_grad(double Sw, double Pe) {
	Sw += S_eps;
	double val = Pe * std::pow(Sw, -lambda);
	double grad = -lambda * Pe * std::pow(Sw, lambda - 1);
	return { val, grad };

}



std::tuple<double, double>
fw_fun_grad(double Sw_in, double Sw_out,
	double dP_in, double Pe)
{
	Sw_in += S_eps;
	Sw_out += S_eps;
	auto Pc_in = Pcow_fun_grad(Sw_in, Pe);
	auto Pc_out = Pcow_fun_grad(Sw_out, Pe);

	const double dPcow_in = std::get<0>(Pc_out) - std::get<0>(Pc_in);
	const double dPcow_grad = -std::get<1>(Pc_in);

	const double dPw_in = dP_in - dPcow_in;
	const double dPw_grad = -dPcow_grad;

	const bool use_in_w = (dPw_in <= 0);
	const bool use_in_o = (dP_in <= 0);

	const double krw = std::pow(use_in_w ? Sw_in : Sw_out, n_w);
	const double krw_grad = use_in_w ? (n_w * std::pow(Sw_in + DBL_MIN, n_w - 1)) : 0.0;

	const double kro = std::pow(1.0 - (use_in_o ? Sw_in : Sw_out) + DBL_MIN, n_o);
	const double kro_grad = use_in_o ? (-n_o * std::pow(1.0 - Sw_in + DBL_MIN, n_o - 1)) : 0.0;

	const double N = krw * (1.0 - Ng * kro);
	const double D = krw + M * kro + S_eps;

	const double Np = krw_grad * (1.0 - Ng * kro) - krw * (Ng * kro_grad);
	const double Dp = krw_grad + M * kro_grad;

	double val = N / D;
	double grad = (Np * D - N * Dp) / (D * D);

	if (std::isnan(grad))
		std::cout << "NAN gradient!!" << std::endl;

	if (val < 0.0) { val = 0.0; grad = 0.0; }
	else if (val > 1.0) { val = 1.0; grad = 0.0; }

	return { val, grad };
}

std::tuple<double, double> resid_grad_fun(double Sw_in, double Sw_old, const double* Sw_neighs, const double* dP_in, const double* uT, double Pe, double alpha) {
	double val = Sw_in - Sw_old;
	double grad = 1.;
	for (int i = 0; i < size; i++) {
		auto val_grad = fw_fun_grad(Sw_in, Sw_neighs[i], dP_in[i], Pe);
		double uT_i = uT[i];
		//val -= uT_i * std::get<0>(val_grad) - alpha * (std::pow(Sw_neighs[i] + diff_eps, -diff_exp) - std::pow(Sw_in + diff_eps, -diff_exp));
		//grad -= uT_i * std::get<1>(val_grad) - diff_exp * alpha * (std::pow(Sw_in + diff_eps, -diff_exp - 1));

		val -= uT_i * std::get<0>(val_grad) + alpha * (std::exp((Sw_neighs[i]-1)*diff_exp) - std::exp((Sw_in-1)*diff_exp));
		grad -= uT_i * std::get<1>(val_grad) + diff_exp * alpha * (std::exp((Sw_neighs[i] - 1) * diff_exp) - std::exp((Sw_in - 1) * diff_exp));
	}
	return { val, grad };
}


std::tuple<double, bool> SCNewton(double Sw_guess, double Sw_old, const double* Sw_neighs, const double* dP_in, const double* uT, double Pe, double alpha, int verbose = 0) {
	bool converged = false;
	double Sw = Sw_guess;
	int i;
	for (i = 0; i < MAXSTEPS; i++) {
		auto resid_grad = resid_grad_fun(Sw, Sw_old, Sw_neighs, dP_in, uT, Pe, alpha);
		double resid = std::get<0>(resid_grad);
		double grad = std::get<1>(resid_grad);
		if (verbose) std::cout << "Resid: " << resid << std::endl;
		if (std::isnan(resid)) {
			std::cout << "nan encountered!" << std::endl;
			std::cout << "Sw: " << Sw_old << "\n";
			std::cout << "Sw neighs: " << Sw_neighs[0] << ", " << Sw_neighs[1] << ", " << Sw_neighs[2] << ", " << Sw_neighs[3] << "\n";
			std::cout << "Sw_old: " << Sw_old << "\n";
			std::cout << "Ut: " << uT[0] << ", " << uT[1] << ", " << uT[2] << ", " << uT[3] << "\n";
		}
		if (std::abs(resid) > RTOL) {
			double update = resid / grad;
			if (std::isnan(update)) {
				std::cout << "NAN encountered\n";
				break;
			}
			update = update < -0.2 ? -0.2 : (update > 0.2 ? 0.2 : update);
			Sw -= update;
			Sw = Sw < 0 ? 0 : (Sw > 1 ? 1 : Sw);
		}
		else {
			if (verbose) std::cout << "converged after " << i << " steps" << std::endl;
			converged = true;
			
			break;
		}
		total_steps += i;
	}

	return { Sw, converged };

}

// Assumes you already have:
//   std::tuple<double,double> resid_grad_fun(double Sw, double Sw_old, const double* Sw_neighs,
//                                            const double* dP_in, const double* uT,
//                                            double Pe, double alpha);
// and constants MAXSTEPS, RTOL, etc.

// Bisection on resid(Sw)=0 in [0,1]. Returns (Sw, converged).
// ---------------------------------------------------------------
// Illinois algorithm: superlinear convergence, bracket-safe,
// no gradient needed. Replaces bisection inner loop.
// Converges much faster than bisection near steep regions.
// ---------------------------------------------------------------
static std::tuple<double, bool>
Illinois(std::function<double(double)> f,
	double a, double b,
	double fa, double fb,
	int verbose = 0)
{
	// Illinois method: regula falsi + side-condition correction
	// Convergence is superlinear (~1.44) and always stays in bracket.
	double Sw = a;
	for (int i = 0; i < MAXSTEPS; ++i)
	{
		// Interpolate
		Sw = a - fa * (b - a) / (fb - fa);
		Sw = std::clamp(Sw, a, b);
		double fc = f(Sw);

		if (verbose)
			std::cout << "Illinois i=" << i
			<< " a=" << a << " b=" << b
			<< " Sw=" << Sw << " resid=" << fc << "\n";

		if (std::isnan(fc))
			return { Sw, false };

		if (std::abs(fc) <= RTOL || std::abs(b - a) <= ATOL)
			return { Sw, true };

		if (fa * fc < 0.0)
		{
			// Root in [a, Sw]: keep a, move b
			// Illinois correction: halve fa to prevent stagnation
			b = Sw; fb = fc;
			fa *= 0.5;   // Illinois modification
		}
		else
		{
			// Root in [Sw, b]: move a, keep b
			a = Sw; fa = fc;
			fb *= 0.5;   // Illinois modification
		}
	}
	return { Sw, false };
}

// ---------------------------------------------------------------
// Robust bracket finder:
// Scans [0,1] with n_scan sub-intervals to find ALL sign changes,
// then picks the one whose midpoint residual is smallest.
// This handles steep gradients by using a fine scan.
// ---------------------------------------------------------------
static std::tuple<double, double, double, double, bool>
FindBracket(std::function<double(double)> f,
	double Sw_guess,
	int n_scan = 200,
	int verbose = 0)
{
	// Build scan grid
	std::vector<double> sv(n_scan + 1);
	std::vector<double> fv(n_scan + 1);
	for (int k = 0; k <= n_scan; ++k)
		sv[k] = static_cast<double>(k) / static_cast<double>(n_scan);

	for (int k = 0; k <= n_scan; ++k)
	{
		fv[k] = f(sv[k]);
		if (std::isnan(fv[k])) fv[k] = 0.0;   // treat nan as zero crossing
	}

	// Collect all sign-changing sub-intervals
	struct Bracket { double a, b, fa, fb, mid_resid; };
	std::vector<Bracket> candidates;

	for (int k = 0; k < n_scan; ++k)
	{
		if (fv[k] * fv[k + 1] <= 0.0)
		{
			double mid = 0.5 * (sv[k] + sv[k + 1]);
			double fm = f(mid);
			candidates.push_back({ sv[k], sv[k + 1], fv[k], fv[k + 1], std::abs(fm) });
		}
	}

	if (candidates.empty())
	{
		if (verbose) std::cout << "FindBracket: no sign change found in [0,1]\n";
		return { 0.0, 1.0, fv[0], fv[n_scan], false };
	}

	// Pick the bracket closest to Sw_guess
	auto best = std::min_element(candidates.begin(), candidates.end(),
		[&](const Bracket& x, const Bracket& y)
		{
			double dx = std::abs(0.5 * (x.a + x.b) - Sw_guess);
			double dy = std::abs(0.5 * (y.a + y.b) - Sw_guess);
			return dx < dy;
		});

	if (verbose)
		std::cout << "FindBracket: best bracket ["
		<< best->a << ", " << best->b << "]\n";

	return { best->a, best->b, best->fa, best->fb, true };
}

// ---------------------------------------------------------------
// Main solver: Illinois inside a robust bracket, fallback chain
// ---------------------------------------------------------------
std::tuple<double, bool>
SCBisection(double Sw_guess, double Sw_old,
	const double* Sw_neighs, const double* dP_in, const double* uT,
	double Pe, double alpha, int verbose = 0)
{
	auto resid_only = [&](double Sw) -> double {
		return std::get<0>(resid_grad_fun(Sw, Sw_old, Sw_neighs, dP_in, uT, Pe, alpha));
		};

	Sw_guess = std::clamp(Sw_guess, 0.0, 1.0);

	// --- 1. Check if guess already satisfies tolerance ---
	{
		double fg = resid_only(Sw_guess);
		if (!std::isnan(fg) && std::abs(fg) <= RTOL)
			return { Sw_guess, true };
	}

	// --- 2. Check endpoints ---
	{
		double fa = resid_only(0.0);
		double fb = resid_only(1.0);
		if (!std::isnan(fa) && std::abs(fa) <= RTOL) return { 0.0, true };
		if (!std::isnan(fb) && std::abs(fb) <= RTOL) return { 1.0, true };

		// --- 3. Try direct bracket [0,1] first (cheap) ---
		if (!std::isnan(fa) && !std::isnan(fb) && fa * fb <= 0.0)
		{
			if (verbose) std::cout << "Direct bracket [0,1] found\n";
			auto [Sw, ok] = Illinois(resid_only, 0.0, 1.0, fa, fb, verbose);
			if (ok) return { std::clamp(Sw, 0.0, 1.0), true };
		}
	}

	// --- 4. No direct bracket: scan [0,1] finely to find sign change ---
	//    Use finer scan (500) since gradient can be extremely steep
	{
		auto [a, b, fa, fb, found] = FindBracket(resid_only, Sw_guess, 500, verbose);
		if (found)
		{
			if (verbose)
				std::cout << "Fine scan bracket: [" << a << ", " << b << "]\n";
			auto [Sw, ok] = Illinois(resid_only, a, b, fa, fb, verbose);
			if (ok) return { std::clamp(Sw, 0.0, 1.0), true };

			// Illinois failed inside a valid bracket - shouldn't happen,
			// but fall through to Newton as last resort
			if (verbose) std::cout << "Illinois failed inside valid bracket\n";
			return SCNewton(Sw, Sw_old, Sw_neighs, dP_in, uT, Pe, alpha, verbose);
		}
	}

	// --- 5. Absolute last resort: Newton multi-start ---
	if (verbose) std::cout << "No bracket found anywhere in [0,1]. Trying Newton.\n";

	return SCNewton(10, Sw_old, Sw_neighs, dP_in, uT, Pe, alpha, verbose);
}


int main() {
	//int n_threads = omp_get_max_threads();
	//const int n_div = 500;
	//const int data_size = (n_div + 1) * (n_div + 1);
	//std::vector<double> R(data_size, 0.0);
	//std::vector<double> R_CRIS(data_size, 0.0);
	//std::vector<double> X_red(2 * data_size);
	//std::vector<double> u(2 * data_size);
	////std::vector<double> R2 = R1;
	//const double Sw_l = 0.7;
	//const double Sw_r = 0.7;
	//const double Sw_old = 0.0;
	//const double Pe = 0;
	//const double M = 2;
	//const double alpha = 0;
	//const double mult = 1.0;
	////double R1, R2;
	//for (int i = 0; i <= n_div; ++i) {
	//	for (int j = 0; j <= n_div; ++j) {
	//		std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n_div + 1) + j;
	//		const double u1 = mult*static_cast<double>(i) / static_cast<double>(n_div);
	//		const double u2 = mult*static_cast<double>(j) / static_cast<double>(n_div);
	//		const double Sw_neighs1[4] = { Sw_l, u2, 0, 0 };
	//		const double Sw_neighs2[4] = { u1, Sw_r, 0, 0 };
	//		const double dP_in[4] = { 100, -100, 0, 0 };
	//		const double uT[4] = {100, -100, 0, 0};
	//		//bool conv1, conv2;
	//		auto [R1, conv1] = resid_grad_fun(u1, Sw_old, Sw_neighs1, dP_in, uT, Pe, alpha);
	//		auto [R2, conv2] = resid_grad_fun(u2, Sw_old, Sw_neighs2, dP_in, uT, Pe, alpha);
	//		if (!(conv1 && conv2))
	//			std::cout << "Not converged (" << u1 << "," << u2 << ")" << std::endl;
	//		R[idx] = std::sqrt(std::pow(R1, 2) + std::pow(R2, 2));
	//		R1 = u1 - std::get<0>(SCNewton(Sw_old, Sw_old, Sw_neighs1, dP_in, uT, Pe, alpha));
	//		R2 = u2 - std::get<0>(SCNewton(Sw_old, Sw_old, Sw_neighs2, dP_in, uT, Pe, alpha));
	//		R_CRIS[idx] = std::sqrt(std::pow(R1, 2) + std::pow(R2, 2));
	//		const double alphas1 = Sw_old + uT[0] * std::get<0>(fw_fun_grad(0, Sw_l, dP_in[0], Pe)); //Ignore diffusion
	//		const double alphas2 = Sw_old + uT[0] * std::get<0>(fw_fun_grad(0, u1, dP_in[0], Pe)); //Ignore diffusion
	//		//double gammas= uT[0];
	//		//double betas = 0;

	//		//std::cout << Sw_ + gammas * std::get<0>(fw_fun_grad(Sw_, 0, -1, Pe, M)) - alphas << std::endl;
	//		X_red[idx * 2] = alphas1;
	//		X_red[idx * 2 + 1] = alphas2;
	//		u[idx * 2] = u1;
	//		u[idx * 2 + 1] = u2;
	//	}
	//}
	//FILE* fp = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/SRDM_resid.bin", "wb");
	//FILE* fp2 = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/CRIS_resid.bin", "wb");
	//FILE* fp3 = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/CRIS_alphas.bin", "wb");
	//FILE* fp4 = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/u.bin", "wb");


	////FILE* fp = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/X_red_region3_diff_test.bin", "wb");
	////if (fp == NULL) {
	////	std::cout << "Error opening file" << std::endl;
	////	return 1;
	////}


	//fwrite(R.data(), sizeof(double), R.size(), fp);
	//fclose(fp);
	//if (fp2 == NULL) {
	//	std::cout << "Error opening file" << std::endl;
	//	return 1;
	//}
	//fwrite(R_CRIS.data(), sizeof(double), R_CRIS.size(), fp2);
	//fclose(fp2);
	//if (fp3 == NULL) {
	//	std::cout << "Error opening file" << std::endl;
	//	return 1;
	//}
	//fwrite(X_red.data(), sizeof(double), X_red.size(), fp3);
	//fclose(fp3);
	//if (fp4 == NULL) {
	//	std::cout << "Error opening file" << std::endl;
	//	return 1;
	//}
	//fwrite(u.data(), sizeof(double), u.size(), fp4);
	//fclose(fp4);

	//std::cout << "Done!" << std::endl;


	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(0., 1.0);
	std::normal_distribution<> disU(0, 10);
	int n_pts = 6;
	double Sw_old = 0.1;
	double Sw_neighs[4] = {1.0, 0.75, 0.5, 0.25};
	double dP_in[4] = { 10, -5, -12, 7 };
	double uT[4] = { 10, -5, -12, 7 };
	double Pe = 0.;
	//double M = 10;
	auto res = SCNewton(n_pts, Sw_old, Sw_neighs, dP_in, uT, Pe, 1);
	int n_threads = omp_get_max_threads();
	//const int col_size = size * 2 + 1;
	//double* X = (double*)malloc(sizeof(double)*NPOINTS*col_size);
	FILE* fp = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/X_region3_diff_test.bin", "rb");
	const std::size_t n_points{ 2811072 };
	//const std::size_t n_points =10;
	const std::size_t data_size = n_points * 9;
	double* X_buff = (double*) malloc(data_size * sizeof(double));
	double* y = (double*)malloc(n_points * sizeof(double));
	double* X_red = (double*)malloc(n_points * 3 * sizeof(double));
	
	if (fp == NULL) {
		std::cout << "Error opening file" << std::endl;
		return 1;
	}
	fread(X_buff, sizeof(double), data_size, fp);
	fclose(fp);
	for (std::size_t i{ 0 }; i < n_points; ++i) {
		//std::cout << "Point " << i + 1;
		double* Sw_neighs = X_buff + i * 9;
		//for (int i{ 0 }; i < 4; i++) Sw_neighs[i] += S_eps;
		const double Sw_old = Sw_neighs[4];
		double Sw_guess = Sw_old;
		//double disp = Sw_neighs[5];
		const double disp = 0.0;
		const double uT[4] = { Sw_neighs[6], Sw_neighs[7], Sw_neighs[8],-Sw_neighs[6] - Sw_neighs[7] - Sw_neighs[8] };
		const double* dP_in = uT;
		//auto [Sw_, converged] = SCNewton(Sw_guess, Sw_old, Sw_neighs, dP_in, uT, Pe,disp,  0);
		double Sw_;
		bool converged;
		if (n_o < 1)
			std::tie(Sw_, converged) = SCBisection(Sw_guess, Sw_old, Sw_neighs, dP_in, uT, Pe, disp, 0);
		else
			std::tie(Sw_, converged) = SCNewton(Sw_guess, Sw_old, Sw_neighs, dP_in, uT, Pe, disp, 0);
		
		//::cout << " converged to ";
		if (!converged) {
			auto resid_grad = resid_grad_fun(Sw_, Sw_old, Sw_neighs, dP_in, uT, Pe, disp);
			std::cout << "Not converged! (Mode 1) Data: \n";
			std::cout << "Sw: " << Sw_ << "\n";
			std::cout << "Sw neighs: " << Sw_neighs[0] << ", " << Sw_neighs[1] << ", " << Sw_neighs[2] << ", " << Sw_neighs[3] << "\n";
			std::cout << "Sw_old: " << Sw_old<<"\n";
			std::cout << "Ut: " << uT[0] << ", " << uT[1] << ", " << uT[2] << ", " << uT[3] << "\n";
			std::cout << "Resid: " << std::get<0>(resid_grad) << std::endl;
		}
		double alphas = 0.0;
		//double alphas = Sw_old;
		double gammas= -uT[3];
		//gammas = gammas < DBL_MIN ? DBL_MIN : gammas;
		//double betas = disp*4;
		double betas = Sw_old;
		for (int ii = 0; ii < 3; ii++) {
			
			//alphas += uT[ii] * std::get<0>(fw_fun_grad(0, Sw_neighs[ii], dP_in[ii], Pe)) - disp * std::exp((Sw_neighs[ii] - 1) * diff_exp);
			alphas -= (uT[ii] + DBL_MIN) / (uT[3] + DBL_MIN) * std::get<0>(fw_fun_grad(0, Sw_neighs[ii], dP_in[ii], Pe));
			//alphas += uT[ii] * std::get<0>(fw_fun_grad(0, Sw_neighs[ii], dP_in[ii], Pe)) - disp * std::pow(Sw_neighs[ii] + diff_eps, -diff_exp);
			//std::cout << std::endl << alphas <<" "<<Sw_old<<" " << Sw_ << std::endl;
		}
		double resid_SRDM = Sw_ + gammas * std::get<0>(fw_fun_grad(Sw_, 0, -1, Pe)) - alphas - betas * std::exp((Sw_ - 1) * diff_exp);
		//double resid_SRDM = Sw_ + gammas * std::get<0>(fw_fun_grad(Sw_, 0, -1, Pe)) - alphas - betas * std::pow(Sw_ + diff_eps, -diff_exp);


		//if (resid_SRDM > RTOL) {
		//	std::cout << "Not converged! (Mode 2) Data: \n";
		//	std::cout << "Sw: " << Sw_ << "\n";
		//	std::cout << "Sw neighs: " << Sw_neighs[0] << ", " << Sw_neighs[1] << ", " << Sw_neighs[2] << ", " << Sw_neighs[3] << "\n";
		//	std::cout << "Sw_old: " << Sw_old << "\n";
		//	std::cout << "Ut: " << uT[0] << ", " << uT[1] << ", " << uT[2] << ", " << uT[3] << "\n";
		//	std::cout << "Resid: " << resid_SRDM<<" " <<std::get<0>(resid_grad_fun(Sw_, Sw_old, Sw_neighs, dP_in, uT, Pe, disp))<< std::endl;
		//}

		//std::cout << Sw_ + gammas * std::get<0>(fw_fun_grad(Sw_, 0, -1, Pe, M)) - alphas << std::endl;
		X_red[i * 3] = alphas;
		X_red[i * 3 + 1] = gammas;
		X_red[i * 3 + 2] = betas;
		y[i] = Sw_;

	}

	
	fp = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/y_imp.bin", "wb");
	FILE* fp2 = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/X_imp.bin", "wb");
	if (fp == NULL || fp2 == NULL) {
		std::cout << "Error opening file" << std::endl;
		return 1;
	}
	if(y!=NULL) fwrite(y, sizeof(double), n_points, fp);
	if (X_red != NULL) fwrite(X_red, sizeof(double), n_points * 3, fp2);
	fclose(fp);
	std::cout << " Average Newton steps: " << total_steps / n_points << std::endl;
	std::cout << y[0] << " " << y[1] << " " << y[2] << std::endl;
	
//#pragma omp parallel for
//	for (int i = 0; i < NPOINTS; i++) {
//		bool converged = false;
//		size_t offset = i * col_size;
//		double col_data[col_size + 1];
//		while (!converged) {
//			double& Sw = *(col_data);
//			double& Pe = *(col_data+1);
//			double& M = *(col_data + 2);
//			double* Sw_neighs = col_data + 3;
//			double& Sw_old = *(col_data+size+3);
//			double* dP_in = col_data + size + 4;
//			std::array<double, size> uT_arr;
//			double* uT = col_data + size * 2 + 4;
//			Sw_old = dis(gen);
//			for (int j = 0; j < size; j++)Sw_neighs[j] = dis(gen);
//			double uT_last = 0;
//			for (int j = 0; j < size - 1; j++) {
//				double uTj = disU(gen);
//				uT_last -= uTj;
//				uT_arr[j] = uTj;
//			}
//			uT_arr[size - 1] = uT_last;
//			std::sort(uT_arr.begin(), uT_arr.end());
//			memcpy(uT, uT_arr.data(), size * sizeof(double));
//			for (int j = 0; j < size; j++) {
//				double dP_ = 1e5;
//				while (std::abs(dP_) >= 500) {
//					double scale = 1 + std::abs(disU(gen));
//					dP_ = uT[j] * scale;
//				}
//				
//				dP_in[j] = dP_;
//			}
//
//			
//			//Pe = dis(gen);
//			Pe = 0.;
//			double M_exp = dis(gen) * 2 - 1;
//			M = std::pow(2, M_exp);
//			//M = 2.0;
//			auto res = SCNewton(n_pts, Sw_old, Sw_neighs, dP_in, uT, Pe, M, 0);
//			Sw = std::get<0>(res);
//			converged = std::get<1>(res);
//		}
//		if (X!= NULL)memcpy(X + offset, col_data, col_size * sizeof(double));
//	}
//	FILE* fp = fopen("C:/Users/Akeem/Desktop/SP2025/MLPINN/build/MKLNet/R4Net/train_data.bin", "wb");
//	if (fp == NULL) {
//		std::cout << "Error opening file" << std::endl;
//		return 1;
//	}
//	if(X!=NULL) fwrite(X, sizeof(double), NPOINTS * col_size, fp);
//	fclose(fp);


}