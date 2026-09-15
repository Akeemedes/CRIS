#ifndef __CRIS_HPP__
#define __CRIS_HPP__

#include <cstddef>
#include <cstdint>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "DiscreteProblem/MultivariableNorm.hpp"
#include "CSR_Pattern.hpp"
#include "SimUtils.hpp"
#include "ExactRegularLocalInverse.hpp"




enum activation {
	RELU=1, TANH=2, NONE=3
};
struct LayerDesc {
	int iS;
	int oS;
	activation activ;
	bool bias;
};

namespace cris_model_package {

inline std::vector<double> double_values(torch::Tensor tensor, const std::string& name) {
	if (tensor.dim() != 1) throw std::runtime_error("Model package field '" + name + "' must be one-dimensional");
	auto values = tensor.cpu().to(torch::kFloat64).contiguous();
	const auto* first = values.data_ptr<double>();
	return { first, first + values.numel() };
}

inline std::vector<std::int64_t> integer_values(torch::Tensor tensor, const std::string& name) {
	if (tensor.dim() != 1) throw std::runtime_error("Model package field '" + name + "' must be one-dimensional");
	auto values = tensor.cpu().to(torch::kInt64).contiguous();
	const auto* first = values.data_ptr<std::int64_t>();
	return { first, first + values.numel() };
}

inline void load(const std::string& package_path, std::vector<LayerDesc>& layers,
	std::vector<double>& IR, std::vector<double>& OR, std::vector<double>& params,
	std::vector<int>* input_transform_codes = nullptr) {
	torch::serialize::InputArchive archive;
	archive.load_from(package_path, torch::Device(torch::kCPU));
	torch::Tensor format_version, parameter_layout_version, parameters;
	torch::Tensor input_min, input_max, output_min, output_max;
	torch::Tensor layer_widths, activation_codes, bias_flags, transform_codes;
	archive.read("format_version", format_version, true);
	archive.read("parameter_layout_version", parameter_layout_version, true);
	archive.read("parameters", parameters, true);
	archive.read("input_min", input_min, true);
	archive.read("input_max", input_max, true);
	archive.read("output_min", output_min, true);
	archive.read("output_max", output_max, true);
	archive.read("layer_widths", layer_widths, true);
	archive.read("activation_codes", activation_codes, true);
	archive.read("bias_flags", bias_flags, true);

	const auto format = integer_values(format_version, "format_version");
	const auto layout = integer_values(parameter_layout_version, "parameter_layout_version");
	if (format.size() != 1 || (format[0] != 1 && format[0] != 2) || layout.size() != 1 || layout[0] != 1) {
		throw std::runtime_error("Unsupported TwoPhaseTransport model package version: " + package_path);
	}
	const auto widths = integer_values(layer_widths, "layer_widths");
	const auto activations = integer_values(activation_codes, "activation_codes");
	const auto biases = integer_values(bias_flags, "bias_flags");
	if (widths.size() < 2 || activations.size() + 1 != widths.size() || biases.size() != activations.size()) {
		throw std::runtime_error("Inconsistent layer metadata in model package: " + package_path);
	}
	std::vector<int> transforms(static_cast<std::size_t>(widths.front()), 0);
	if (format[0] == 2) {
		archive.read("input_transform_codes", transform_codes, true);
		const auto raw_transforms = integer_values(transform_codes, "input_transform_codes");
		if (raw_transforms.size() != transforms.size()) {
			throw std::runtime_error("Input transform count does not match architecture: " + package_path);
		}
		for (std::size_t index = 0; index < transforms.size(); ++index) {
			if (raw_transforms[index] != 0 && raw_transforms[index] != 1) {
				throw std::runtime_error("Unsupported input transform code in model package: " + package_path);
			}
			transforms[index] = static_cast<int>(raw_transforms[index]);
		}
	}

	const auto input_lower = double_values(input_min, "input_min");
	const auto input_upper = double_values(input_max, "input_max");
	const auto output_lower = double_values(output_min, "output_min");
	const auto output_upper = double_values(output_max, "output_max");
	if (input_lower.size() != static_cast<std::size_t>(widths.front()) || input_upper.size() != input_lower.size() ||
		output_lower.size() != static_cast<std::size_t>(widths.back()) || output_upper.size() != output_lower.size()) {
		throw std::runtime_error("Normalization dimensions do not match model architecture: " + package_path);
	}
	IR = input_lower;
	IR.insert(IR.end(), input_upper.begin(), input_upper.end());
	OR = output_lower;
	OR.insert(OR.end(), output_upper.begin(), output_upper.end());
	if (input_transform_codes != nullptr) *input_transform_codes = std::move(transforms);

	layers.clear();
	std::size_t parameter_count = 0;
	for (std::size_t layer = 0; layer < activations.size(); ++layer) {
		if (widths[layer] <= 0 || widths[layer + 1] <= 0 || (biases[layer] != 0 && biases[layer] != 1)) {
			throw std::runtime_error("Invalid layer dimensions or bias flag in model package: " + package_path);
		}
		const auto code = activations[layer];
		if (code != RELU && code != TANH && code != NONE) {
			throw std::runtime_error("Unsupported activation code in model package: " + package_path);
		}
		const bool has_bias = biases[layer] != 0;
		layers.push_back({ static_cast<int>(widths[layer]), static_cast<int>(widths[layer + 1]),
			static_cast<activation>(code), has_bias });
		parameter_count += static_cast<std::size_t>(widths[layer]) * static_cast<std::size_t>(widths[layer + 1]);
		if (has_bias) parameter_count += static_cast<std::size_t>(widths[layer + 1]);
	}
	params = double_values(parameters, "parameters");
	if (params.size() != parameter_count) {
		throw std::runtime_error("Parameter count does not match model architecture: " + package_path);
	}
}

} // namespace cris_model_package


void parse_network(const std::string& model_path, std::vector<LayerDesc>& layers, std::vector<double>& IR,
	std::vector<double>& OR, std::vector<double>& params, std::vector<int>* input_transform_codes = nullptr) {
	if (std::filesystem::is_regular_file(model_path)) {
		cris_model_package::load(model_path, layers, IR, OR, params, input_transform_codes);
		return;
	}
	if (input_transform_codes != nullptr) input_transform_codes->clear();

	std::string config_path = model_path + "/config.txt";
	const std::string var_path = model_path + "/vars.bin";
	std::ifstream config_file(config_path);
	std::ostringstream msg;
	ASSERT_WITH_MSG(config_file.is_open(),
		"Could not open config file: " + config_path);
	FILE* fp = fopen(var_path.c_str(), "rb");
	if (!fp) {
		throw std::runtime_error("Could not open network weights file: " + var_path);
	}

	int n_layers;
	config_file >> n_layers;
	std::cout << n_layers << " layers" << std::endl;
	std::vector<int> dims(n_layers + 1);
	std::vector<bool> biases(n_layers);
	std::vector<activation> activs(n_layers);

	bool bias;
	int activ;
	int dim;
	layers.resize(n_layers);
	std::cout << "Bias: ";
	for (int i = 0; i < n_layers; i++) {
		if (!(config_file >> bias)) {
			throw std::runtime_error("Failed to read bias flag at layer " + std::to_string(i));
		}
		biases[i] = bias;
	}
	std::cout << std::endl;
	for (int i = 0; i < n_layers; i++) {
		config_file >> activ;
		switch (activ) {
		case NONE:
			activs[i] = NONE;
			break;
		case RELU:
			activs[i] = RELU;
			break;
		case TANH:
			activs[i] = TANH;
			break;
		default:
			break;
		};
	}

	for (int i = 0; i < n_layers + 1; i++) {
		config_file >> dim;
		dims[i] = dim;
	}
	std::cout << n_layers << " layers" << std::endl;

	size_t total_size = 0;
	for (int i = 0; i < n_layers; i++) {
		auto id = dims[i];
		auto od = dims[i + 1];
		std::cout << "Layer " << i + 1 << ": (" << id << "," << od << ")" << std::endl;
		total_size += static_cast<std::size_t>(id) * static_cast<std::size_t>(od);
		if (biases[i]) total_size += od;
		layers[i] = { id, od, activs[i], biases[i] };
	}
	std::cout << "Weight size: " << total_size << std::endl;
	params.resize(total_size);
	size_t total_size_bytes = sizeof(double) * total_size;
	std::cout << "Weight size (bytes): " << static_cast<double>(total_size_bytes) / 1024. << " kb" << std::endl;
	size_t n_inp = layers[0].iS;
	size_t n_out = layers[n_layers - 1].oS;
	size_t IR_size = 2 * n_inp;
	IR.resize(IR_size);
	size_t OR_size = 2 * n_out;
	total_size += IR_size + OR_size;
	OR.resize(OR_size);
	std::vector<double> buffer(total_size);
	const auto values_read = fread(buffer.data(), sizeof(double), total_size, fp);
	fclose(fp);
	if (values_read != total_size) throw std::runtime_error("Truncated network weights file: " + var_path);
	std::copy_n(buffer.data(), IR_size, IR.data());
	std::copy_n(buffer.data() + IR_size, OR_size, OR.data());
	std::copy_n(buffer.data() + IR_size + OR_size, params.size(), params.data());
}


struct FCN : torch::nn::Module {

    // Exact local-inverse mode is intentionally hosted by the same CRIS
    // network type as a learned package. That keeps the global discretization
    // and solver instantiation identical in the exact/learned comparison.
    explicit FCN(double exact_regular_mobility_ratio)
        : exact_regular_mode_(true), exact_regular_mobility_ratio_(exact_regular_mobility_ratio) {
        if (!(exact_regular_mobility_ratio_ > 0.0) || !std::isfinite(exact_regular_mobility_ratio_)) {
            throw std::runtime_error("Exact regular FCN mode requires a finite positive mobility ratio");
        }
    }

    FCN(double mobility_ratio, bool imp_law) : FCN(mobility_ratio) {
        exact_imp_mode_ = imp_law;
    }



	FCN(const std::vector<LayerDesc>& layer_descs, std::vector<double>& IR, std::vector<double>& OR,
		const std::vector<int>& input_transform_codes = {}) :mlayer_descs(layer_descs)

	{

		build_network(layer_descs, IR, OR, input_transform_codes);


	}

	FCN(const std::vector<LayerDesc>& layer_descs, std::vector<double>& IR, std::vector<double>& OR, double* data_ptr,
		const std::vector<int>& input_transform_codes = {}) :mlayer_descs(layer_descs)
	{

		build_network(layer_descs, IR, OR, input_transform_codes);
		//std::cout << "Network Constructed!" << std::endl;
		set_params(data_ptr);
		//std::cout << "Network parameters updated!" << std::endl;
		//std::cout << minput_min << minput_scale << moutput_min << moutput_scale << std::endl;
		//auto& layer = mlayers.back();
		//std::cout << layer->weight << layer->bias << std::endl;

	}

	void build_network(const std::vector<LayerDesc>& layer_descs,  std::vector<double>& IR,  std::vector<double>& OR,
		const std::vector<int>& input_transform_codes) {
		int layer_no = 0;
		for (auto& layer_desc : layer_descs) {
			if (!layer_no) minput_dim = layer_desc.iS;
			std::string layer_name = "fc" + std::to_string(layer_no);
			//std::cout << layer_name << std::endl;
			torch::nn::Linear linear_layer{ nullptr };
			linear_layer = register_module(layer_name, torch::nn::Linear(torch::nn::LinearOptions(layer_desc.iS, layer_desc.oS).bias(layer_desc.bias)));
			//std::cout << "Layer Registered" << std::endl;
			mlayers.push_back(linear_layer);
			//std::cout << "Layer Pushed" << std::endl;
			if (++layer_no == layer_descs.size()) moutput_dim = layer_desc.oS;


		}
		if (!input_transform_codes.empty() && input_transform_codes.size() != static_cast<std::size_t>(minput_dim)) {
			throw std::runtime_error("Input transform count does not match network input dimension");
		}
		minput_transform_codes = input_transform_codes.empty() ? std::vector<int>(minput_dim, 0) : input_transform_codes;
		std::vector<double> transformed_input_min(IR.begin(), IR.begin() + minput_dim);
		std::vector<double> transformed_input_max(IR.begin() + minput_dim, IR.begin() + 2 * minput_dim);
		for (int index = 0; index < minput_dim; ++index) {
			if (minput_transform_codes[index] == 1) {
				if (transformed_input_min[index] < 0.0 || transformed_input_max[index] < 0.0) {
					throw std::runtime_error("log1p input transform requires a nonnegative physical range");
				}
				transformed_input_min[index] = std::log1p(transformed_input_min[index]);
				transformed_input_max[index] = std::log1p(transformed_input_max[index]);
			} else if (minput_transform_codes[index] != 0) {
				throw std::runtime_error("Unsupported input transform code");
			}
		}
		torch::Tensor minput_max, moutput_max;
		minput_min = torch::from_blob(transformed_input_min.data(), { 1, minput_dim }, torch::kFloat64).clone();
		minput_max = torch::from_blob(transformed_input_max.data(), { 1, minput_dim }, torch::kFloat64).clone();
		minput_scale = 2 / (minput_max - minput_min);

		moutput_min = torch::from_blob(OR.data(), { 1, moutput_dim }, torch::kFloat64).clone();
		// `OR` belongs to the model-loading adapter and may be destroyed as soon
		// as construction returns. All normalization tensors therefore own their
		// storage; retaining a from_blob view here caused a use-after-free during
		// CRIS evaluation from the simulation factory.
		moutput_max = torch::from_blob(OR.data() + moutput_dim, { 1, moutput_dim }, torch::kFloat64).clone();
		moutput_scale = (moutput_max - moutput_min) / 2;


		//mOR = torch::from_blob(OR.data(), { 1, moutput_dim }, torch::TensorOptions().dtype(torch::kF64));
	}

	void disable_layer_gradients(int l) {
		auto& layer = mlayers[l];
		for (auto& param : layer->parameters()) param.requires_grad_(false);
	}

	void disable_gradients() {
		for (auto& layer : mlayers) {
			for (auto& param : layer->parameters()) param.requires_grad_(false);
		}
	}

	void set_params(double* data_ptr) {
		torch::Tensor weight_tensor, bias_tensor;
		std::size_t offset = 0;
		for (int l = 0; l < mlayers.size(); l++) {
			auto& layer_desc = mlayer_descs[l];
			auto& layer = mlayers[l];
			//weight_tensor = torch::from_blob(data_ptr, { layer_desc.iS, layer_desc.oS }, torch::kFloat64);
			layer->weight = torch::from_blob(data_ptr + offset, { layer_desc.oS, layer_desc.iS }, torch::kFloat64).clone();
			offset += layer_desc.iS * layer_desc.oS;
			if (layer_desc.bias) {
				layer->bias = torch::from_blob(data_ptr + offset, { layer_desc.oS }, torch::kFloat64).clone();
				offset += layer_desc.oS;
			}
		}
	}


	torch::Tensor forward(torch::Tensor x) {
		if (exact_imp_mode_) return cris_exact::imp_advection_local_inverse(std::move(x), exact_regular_mobility_ratio_);
		if (exact_regular_mode_) return cris_exact::regular_advection_local_inverse(std::move(x), exact_regular_mobility_ratio_);
		//std::cout << x << std::endl;

		//std::cout << minput_min << minput_scale << moutput_min << moutput_scale << std::endl;
		//auto& layer = mlayers.back();
		//std::cout << layer->weight << layer->bias << std::endl;

		x = x.reshape({ x.size(0), minput_dim });
		if (std::any_of(minput_transform_codes.begin(), minput_transform_codes.end(), [](int code) { return code != 0; })) {
			std::vector<torch::Tensor> columns;
			columns.reserve(static_cast<std::size_t>(minput_dim));
			for (int index = 0; index < minput_dim; ++index) {
				auto column = x.select(1, index);
				columns.push_back((minput_transform_codes[index] == 1 ? torch::log1p(column) : column).unsqueeze(1));
			}
			x = torch::cat(columns, 1);
		}
		x = minput_scale * (x - minput_min) - 1;
		//std::cout << "Scaled x: " << x << std::endl;
		//std::cout << x << std::endl;
		for (int l = 0; l < mlayers.size(); l++) {
			//std::cout << "Computing Layer " << l << std::endl;
			auto& layer = mlayers[l];
			x = layer->forward(x);
			switch (mlayer_descs[l].activ) {
			case TANH:
				x = torch::tanh(x);
				break;
			case RELU:
				x = torch::relu(x);
				break;
			default:
				break;
			}

		}
		return moutput_min + moutput_scale * (x + 1);



	}

private:
	int minput_dim, moutput_dim;
	std::vector<torch::nn::Linear> mlayers;
	std::vector<LayerDesc> mlayer_descs;
	std::vector<int> minput_transform_codes;
	torch::Tensor minput_min, minput_scale, moutput_min, moutput_scale;
	bool exact_regular_mode_{false};
	bool exact_imp_mode_{false};
	double exact_regular_mobility_ratio_{0.0};

};

template <typename Network_type, typename TPFA_Mesh, typename PhysicsAdapter>
class CRIS
	: public PhysicsAdapter::template Adapter<CRIS<Network_type, TPFA_Mesh, PhysicsAdapter>>
{
public:
	using Self = CRIS<Network_type, TPFA_Mesh, PhysicsAdapter>;
	using Base = typename PhysicsAdapter::template Adapter<Self>;
	using State = typename PhysicsAdapter::State; // both adapters should expose State

	//using ResidualNorm = MultivariableNorm<1, TwoNormFunct>;
	using ResidualNorm = MultivariableNorm<1, InfNormFunct>;
	using UpdateNorm = MultivariableNorm<1, InfNormFunct>;

	explicit CRIS(std::shared_ptr<Network_type> net,
		const TPFA_Mesh& _mesh,
		const PhysicsAdapter& phys,         //pass adapter object (movable)
		double _RTOL = 1.0e-8,
		double _DUTOL = 1.0e-7)
		: Base(std::move(phys))                     //constructs adapter base subobject
		, RTOL(_RTOL)
		, DUTOL(_DUTOL)
		, mNet(std::move(net))
		, mesh(_mesh)
		, resid(mesh.num_cells())
		, diag(mesh.num_cells())
		, k_12(mesh.num_intr_faces())
		, k_21(mesh.num_intr_faces())
	{
		mNet->disable_gradients();

		this->init_cris_storage_from_mesh();
	}

	std::size_t neqs() const { return mesh.num_cells(); }
	std::size_t neq_per_cell() const { return 1; }
	std::size_t max_nnz() const { return mesh.max_num_neighbors(); }

	template<typename M>
	void update_face_gradients(M& J, std::size_t f, double value_12, double value_21) {
		auto& v = J.value();
		v[k_12[f]] = static_cast<typename M::value_t>(value_12);
		v[k_21[f]] = static_cast<typename M::value_t>(value_21);
	}

	template<typename M>
	void update_diag_gradient(M& J, std::size_t l, double value) {
		auto& v = J.value();
		v[diag[l]] = static_cast<typename M::value_t>(value);
	}

	template<typename V1, typename V2, typename M>
	void allocate_containers(V1& du, V2& r, M& J) {
		J.block_size = neq_per_cell();
		J.reserve(neqs(), max_nnz());
		csr_pattern::build_pattern(J, mesh, diag, k_12, k_21);
		J.reset_size();

		r.resize(neqs(), 0.0);
		du.resize(neqs(), 0.0);

		for (std::size_t l = 0; l < neqs(); ++l)
			update_diag_gradient(J, l, 1.0);
	}

	template<typename V1, typename V2, typename M>
	void initialize_containers(V1& du, V2&, M&) {
		std::fill(du.begin(), du.end(), 0.0);
	}

	template <typename V1>
	void evaluate(const State& u, const std::vector<double>& uold, double DT, V1& resid_out, bool grad = false) {
		this->embed_cris_input(u, uold, DT, grad);

		y = mNet->forward(X);
		//std::cout << "Input: " << X << "\nOutput: " << y << std::endl;
		//auto max = torch::max(y, 0);
		//auto min = torch::min(y, 0);
		//std::cout << "max y: " << std::get<0>(max) << std::endl;
		//std::cout << "Min y: " << std::get<0>(min) << std::endl;

		auto* out = y.data_ptr<double>();
		for (std::size_t l = 0; l < neqs(); ++l)
			resid_out[l] = u[l] - out[l];
	}

	template<typename V1, typename M>
	bool discretize(const State& u, const std::vector<double>& uold, double DT, V1& resid_out, M& J) {
		evaluate(u, uold, DT, resid_out, true);
		this->zero_grad();
		y.backward(torch::ones_like(y));
		this->propagate_cris_gradients(J, DT, u);
		J.reset_size();
		//std::cout << "Resid" << std::endl;
		//for (auto val: resid_out)
		//	std::cout << val << std::endl;
		//std::cout << "Grad" << std::endl;
		//for (std::size_t i = 0; i < J.value().size(); i++)
		//	std::cout << J.colind()[i] << ":" << J.value()[i] << std::endl;
		//for (auto val : J.value())
		//	std::cout << val << std::endl;
		return true;
	}

	template<typename V>
	std::pair<bool, ResidualNorm> is_residual_norm_converged(const V& resid) const {
		ResidualNorm rnrm;
		rnrm.evaluate(resid);
		return { rnrm.isLessEqual(RTOL), rnrm };
	}

	template<typename V>
	std::pair<bool, UpdateNorm> is_update_norm_converged(const V& upd) const {
		UpdateNorm dunrm;
		dunrm.evaluate(upd);
		return { dunrm.isLessEqual(DUTOL), dunrm };
	}

public:
	const double RTOL;
	const double DUTOL;

	std::shared_ptr<Network_type> mNet;
	const TPFA_Mesh& mesh;

	std::vector<double> resid;
	torch::Tensor X, y;

	std::vector<std::size_t> diag, k_12, k_21;
};

#endif // __CRIS_HPP__ included
