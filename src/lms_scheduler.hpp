#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> std_randn_function(uint32_t seed, uint32_t h, uint32_t w) {
    std::vector<float> noise;
    {
        std::mt19937 gen{static_cast<unsigned long>(seed)};
        std::normal_distribution<float> normal{0.0f, 1.0f};
        noise.resize(h / 8 * w / 8 * 4 * 1);
        std::for_each(noise.begin(), noise.end(), [&](float& x) {
            x = normal(gen);
        });
    }
    return noise;
}

std::vector<float> np_randn_function() {
    // read np generated latents with defaut seed 42
    std::vector<float> latent_vector_1d;
    std::ifstream latent_copy_file;
    latent_copy_file.open("../scripts/np_latents_512x512.txt");
    std::vector<std::string> latent_vector_new;
    if (latent_copy_file.is_open()) {
        std::string word;
        while (latent_copy_file >> word)
            latent_vector_new.push_back(word);
        latent_copy_file.close();
    } else {
        std::cout << "could not find the np_latents_512x512.txt" << std::endl;
        exit(0);
    }

    latent_vector_new.insert(latent_vector_new.begin(), latent_vector_new.begin(), latent_vector_new.end());

    for (int i = 0; i < (int)latent_vector_new.size() / 2; i++) {
        latent_vector_1d.push_back(std::stof(latent_vector_new[i]));
    }

    return latent_vector_1d;
}

class LMSDiscreteScheduler {
public:
    // static const int order = 1;
    // config
    int num_train_timesteps_config;
    int original_inference_steps_config;
    std::vector<int> timesteps;
    float timestep_scaling = 10.0;
    std::string prediction_type_config;
    bool thresholding = false;
    bool clip_sample = false;
    float clip_sample_range = 1.0;
    float dynamic_thresholding_ratio = 0.995;
    float sample_max_value = 1.0;

    std::vector<float> log_sigma;
    int step = 20;
    std::vector<float> sigma;
    std::vector<std::vector<float>> derivative_list;

    // construct
    LMSDiscreteScheduler(int num_train_timesteps = 1000,
                         float beta_start = 0.00085,
                         float beta_end = 0.012,
                         std::string beta_schedule = "scaled_linear",
                         std::vector<float> trained_betas = {},
                         int original_inference_steps = 50,
                         bool set_alpha_to_one = true,
                         int steps_offset = 0,
                         std::string prediction_type = "epsilon",
                         std::string timestep_spacing = "leading",
                         bool rescale_betas_zero_snr = false

                         )
        : original_inference_steps_config(original_inference_steps),
          num_train_timesteps_config(num_train_timesteps),
          prediction_type_config(prediction_type) {
        // __init__

        // betas
        if (!trained_betas.empty()) {
            auto betas = trained_betas;
        } else if (beta_schedule == "linear") {
            for (int32_t i = 0; i < num_train_timesteps; i++) {
                betas.push_back(beta_start + (beta_end - beta_start) * i / (num_train_timesteps - 1));
            }
        } else if (beta_schedule == "scaled_linear") {
            // pytorch's self.betas has small precision gap
            // tensor([0.0008, 0.0009, 0.0009, 0.0009, 0.0009, ... 0.0119, 0.0119, 0.0119, 0.0119, 0.0120, 0.0120,
            // 0.0120])

            // c++ is more precise with the start-value 0.00085 instead of 0.0008
            // 0.00085, 0.000854699, 0.00085941, 0.000864135, 0.000868872, ... 0.0119472, 0.0119648, 0.0119824, 0.012

            // Without linspace
            // float beta_start_sqrt = std::pow(beta_start, 0.5);
            // float beta_end_sqrt = std::pow(beta_end, 0.5);
            float beta_start_sqrt = sqrt(beta_start);
            float beta_end_sqrt = sqrt(beta_end);
            float beta_step = (beta_end_sqrt - beta_start_sqrt) / (num_train_timesteps - 1);
            for (int i = 0; i < num_train_timesteps; ++i) {
                float current_beta_sqrt = beta_start_sqrt + i * beta_step;
                betas.push_back(std::pow(current_beta_sqrt, 2));
            }

            // for (int i = 0; i < num_train_timesteps; ++i) {
            //     std::cout << betas[i] << ", ";
            // }
        } else {
            std::cout << " beta_schedule must be one of 'linear' or 'scaled_linear' " << std::endl;
        }

        // alphas
        for (float b : betas) {
            alphas.push_back(1 - b);
        }

        // sigmas
        // std::vector<float> log_sigma;
        for (int32_t i = 1; i <= (int)alphas.size(); i++) {
            float alphas_cumprod =
                std::accumulate(std::begin(alphas), std::begin(alphas) + i, 1.0, std::multiplies<float>{});
            float sigma = sqrt((1 - alphas_cumprod) / alphas_cumprod);
            log_sigma.push_back(std::log(sigma));
        }
    }

    std::vector<float> set_timesteps(int num_inference_steps,
                                     int original_inference_steps = -1,
                                     double strength = 1.0) {
        // t_to_sigma
        sigma.resize(step);
        float delta = -999.0f / (step - 1);

        // transform interpolation to time range

        for (int32_t i = 0; i < step; i++) {
            float t = 999.0 + i * delta;
            int32_t low_idx = std::floor(t);
            int32_t high_idx = std::ceil(t);
            float w = t - low_idx;
            sigma[i] = std::exp((1 - w) * log_sigma[low_idx] + w * log_sigma[high_idx]);
        }

        sigma.push_back(0.f);

        return sigma;
    }

    // copied from diffusers.schedulers.scheduling_euler_discrete._sigma_to_t
    std::vector<int64_t> sigma_to_t(std::vector<float>& log_sigmas, float sigma) {
        double log_sigma = std::log(sigma);
        std::vector<float> dists(1000);
        for (int32_t i = 0; i < 1000; i++) {
            if (log_sigma - log_sigmas[i] >= 0)
                dists[i] = 1;
            else
                dists[i] = 0;
            if (i == 0)
                continue;
            dists[i] += dists[i - 1];
        }

        // get sigmas range
        int32_t low_idx = std::min(int(std::max_element(dists.begin(), dists.end()) - dists.begin()), 1000 - 2);
        int32_t high_idx = low_idx + 1;
        float low = log_sigmas[low_idx];
        float high = log_sigmas[high_idx];
        // interpolate sigmas
        double w = (low - log_sigma) / (low - high);
        w = std::max(0.0, std::min(1.0, w));

        int64_t t = std::llround((1 - w) * low_idx + w * high_idx);
        std::vector<int64_t> vector_t{t};
        return vector_t;
    }

    // adaptive trapezoidal integral function
    template <class F, class Real>
    Real trapezoidal(F f, Real a, Real b, Real tol = 1e-6, int max_refinements = 100) {
        Real h = (b - a) / 2.0;
        Real ya = f(a);
        Real yb = f(b);
        Real I0 = (ya + yb) * h;

        for (int k = 1; k <= max_refinements; ++k) {
            Real sum = 0.0;
            for (int j = 1; j <= (1 << (k - 1)); ++j) {
                sum += f(a + (2 * j - 1) * h);
            }

            Real I1 = 0.5 * I0 + h * sum;
            if (k > 1 && std::abs(I1 - I0) < tol) {
                return I1;
            }

            I0 = I1;
            h /= 2.0;
        }
        // If the desired accuracy is not achieved, return the best estimate
        return I0;
    }

    float lms_derivative_function(float tau,
                                  int32_t order,
                                  int32_t curr_order,
                                  std::vector<float> sigma_vec,
                                  int32_t t) {
        float prod = 1.0;

        for (int32_t k = 0; k < order; k++) {
            if (curr_order == k) {
                continue;
            }
            prod *= (tau - sigma_vec[t - k]) / (sigma_vec[t - curr_order] - sigma_vec[t - k]);
        }
        return prod;
    }

    // Predict the sample from the previous timestep by reversing the SDE.
    std::vector<float> step_func(const std::vector<float>& model_output,
                                 int32_t timestep,
                                 std::vector<float>& sample,
                                 int32_t order = 4) {
        // Predict the sample from the previous timestep by reversing the SDE.
        // This function propagates the diffusion process from the learned model outputs
        // (most often the predicted noise).

        // Args:
        //     model_output: The direct output from learned diffusion model.
        //     timestep: The current discrete timestep in the diffusion chain.
        //     sample: A current instance of a sample created by the diffusion process.
        //     order (`int`, defaults to 4): The order of the linear multistep method.

        std::vector<float> noise_pred_1d = model_output;
        int i = timestep;
        std::vector<float> latent_vector_1d_new = sample;
        std::vector<float> derivative_vec_1d;
        // Notice: latent_vector_1d = latent_vector_1d_new / 14.6146

        for (int32_t j = 0; j < static_cast<int>(latent_vector_1d_new.size()); j++) {
            // 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
            // defaut "epsilon"
            // std::cout << "latent_vector_1d_new: " << latent_vector_1d_new[j] << std::endl;
            // std::cout << "sigma[i]: " << sigma[i] << std::endl;
            // std::cout << "noise_pred_1d[i]: " << noise_pred_1d[i] << std::endl;

            float pred_latent = latent_vector_1d_new[j] / 14.6146 - sigma[i] * noise_pred_1d[j];
            // std::cout << "pred_latent" << pred_latent << std::endl;

            // 2. Convert to an ODE derivative
            derivative_vec_1d.push_back((latent_vector_1d_new[j] / 14.6146 - pred_latent) / sigma[i]);
        }

        derivative_list.push_back(derivative_vec_1d);
        // keep the list size within 4
        if ((int)derivative_list.size() > order) {
            derivative_list.erase(derivative_list.begin());
        }

        // 3. Compute linear multistep coefficients

        order = std::min(i + 1, order);

        std::vector<float> lms_coeffs;
        for (int32_t curr_order = 0; curr_order < order; curr_order++) {
            // add this for the sigma
            auto f = [this, order, curr_order, i](float tau) {
                return lms_derivative_function(tau, order, curr_order, this->sigma, i);
            };

            auto integrated_coeff_new =
                trapezoidal(f, static_cast<double>(sigma[i]), static_cast<double>(sigma[i + 1]), 1e-4);
            lms_coeffs.push_back(integrated_coeff_new);
        }
        // std::cout << " lms_coeff: ";
        // for (auto lms_coeff : lms_coeffs) {
        //     std::cout << lms_coeff << ", ";
        // }
        // std::cout << std::endl;

        // 4. Compute previous sample based on the derivatives path
        // prev_sample = sample + sum(coeff * derivative for coeff, derivative in zip(lms_coeffs,
        // reversed(self.derivatives))) Reverse list of tensors this.derivatives
        std::vector<std::vector<float>> rev_derivative = derivative_list;
        std::reverse(rev_derivative.begin(), rev_derivative.end());

        // derivative * coeffs
        for (int32_t m = 0; m < order; m++) {
            float coeffs_const{lms_coeffs[m]};
            std::for_each(rev_derivative[m].begin(), rev_derivative[m].end(), [coeffs_const](float& i) {
                i *= coeffs_const;
            });
        }

        // sum of derivative
        std::vector<float> derivative_sum = rev_derivative[0];
        if (order > 1) {
            for (int32_t d = 0; d < order - 1; d++) {
                std::transform(derivative_sum.begin(),
                               derivative_sum.end(),
                               rev_derivative[d + 1].begin(),
                               derivative_sum.begin(),
                               [](float x, float y) {
                                   return x + y;
                               });
            }
        }

        // latent + sum of derivative
        std::transform(derivative_sum.begin(),
                       derivative_sum.end(),
                       latent_vector_1d_new.begin(),
                       latent_vector_1d_new.begin(),
                       [](float x, float y) {
                           return x + y;
                       });
        return latent_vector_1d_new;
    }

private:
    // Define class members
    std::vector<float> betas;
    std::vector<float> alphas;
    std::vector<float> alphas_cumprod;
    float final_alpha_cumprod;
    float initNoiseSigma;
    int _step_index;
    int num_inference_steps_;
};
