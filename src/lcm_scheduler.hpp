#include <cmath>
#include <cassert>
#include <stdexcept>
#include <algorithm>
#include <numeric>  
#include <vector>
#include <string>
#include <iostream>
#include <random>
#include <filesystem>
#include <fstream>
#include <iostream>

std::vector<float> read_randn_function(const std::string& file_path) {
    std::ifstream rand_file;
    rand_file.open(file_path.data()); // 
    std::vector<std::string> string_data;
    if (rand_file.is_open()) {
        std::string word;
        while (rand_file >> word)
            string_data.push_back(word);
        rand_file.close();
    } else {
        std::cout << "could not find file: " << file_path << std::endl;
        exit(0);
    }
    std::vector<float> output;
    for (int i = 0; i < (int)string_data.size(); i++) {
        output.push_back(std::stof(string_data[i]));
    }
    return output;
}

// Function to get w_embedding
std::vector<float> get_w_embedding(float guidance_scale = 8.0, uint32_t embedding_dim = 512) {
    float w = guidance_scale * 1000;
    uint32_t half_dim = embedding_dim / 2;

    float emb = log(10000) / (half_dim - 1);

    std::vector<float> emb_vec(half_dim);
    std::iota(emb_vec.begin(), emb_vec.end(), 0);

    std::transform(emb_vec.begin(), emb_vec.end(), emb_vec.begin(),
                    std::bind(std::multiplies<float>(), std::placeholders::_1, -emb));

    std::transform(emb_vec.begin(), emb_vec.end(), emb_vec.begin(), [](float x){return std::exp(x);});

    std::transform(emb_vec.begin(), emb_vec.end(), emb_vec.begin(),
                    std::bind(std::multiplies<float>(), std::placeholders::_1, w));

    std::vector<float> res_vec(half_dim), emb_cos(half_dim);
    std::transform(emb_vec.begin(), emb_vec.end(), res_vec.begin(), [](float x){return std::sin(x);});
    std::transform(emb_vec.begin(), emb_vec.end(), emb_cos.begin(), [](float x){return std::cos(x);});
    res_vec.insert(res_vec.end(), emb_cos.begin(), emb_cos.end());

    if (embedding_dim % 2 == 1)
        res_vec.insert(res_vec.end(), 0);

    assert(res_vec.size() == embedding_dim);

    // std::cout << "get_w_embedding:" << std::endl;
    // std::cout << res_vec.size() << std::endl;
    // for (auto i: res_vec)
    //     std::cout << i << ' ';
    // std::cout << std::endl;
    
    return res_vec;
}


class LCMSchedulerOutput {
public:
    LCMSchedulerOutput(const std::vector<float>& prevSample, const std::vector<float>& denoised) : prevSample(prevSample), denoised(denoised) {}

    std::vector<float> getPrevSample() const { return prevSample; }
    std::vector<float> getDenoised() const { return denoised; }

private:
    std::vector<float> prevSample;
    std::vector<float> denoised;
};

class LCMScheduler {
public:
    static const int order = 1;
    using FloatTensor = std::vector<float>;
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
    bool read_noise_config = false;
    // construct
    LCMScheduler(
        int num_train_timesteps = 1000,
        float beta_start = 0.00085,
        float beta_end = 0.012,
        bool read_noise = false,
        std::string beta_schedule = "scaled_linear",
        std::vector<float> trained_betas = {},
        int original_inference_steps = 50,
        bool set_alpha_to_one = true,
        int steps_offset = 0,
        std::string prediction_type = "epsilon"
        
    ) : read_noise_config(read_noise),
        original_inference_steps_config(original_inference_steps),
        num_train_timesteps_config(num_train_timesteps),
        prediction_type_config(prediction_type)
    {
        // Initialize the class members based on the constructor parameters
        std::vector<float> trainedBetasVector(trained_betas);
        if (!trainedBetasVector.empty()) {
            betas = trainedBetasVector;
        } else if (beta_schedule == "linear") {
            float beta_step = (beta_end - beta_start) / (num_train_timesteps - 1);
            for (int i = 0; i < num_train_timesteps; ++i) {
                betas.push_back(beta_start + i * beta_step);
            }
        } else if (beta_schedule == "scaled_linear") {
            // pytorch's self.betas has small precision gap 
            // tensor([0.0008, 0.0009, 0.0009, 0.0009, 0.0009, ... 0.0119, 0.0119, 0.0119, 0.0119, 0.0120, 0.0120, 0.0120])

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
            throw std::runtime_error(beta_schedule + " is not implemented for " + typeid(*this).name());
        }

        alphas.reserve(num_train_timesteps);
        std::transform(betas.begin(), betas.end(), std::back_inserter(alphas),
                       [](float beta) { return 1.0 - beta; });

        alphas_cumprod.resize(num_train_timesteps);
        std::partial_sum(alphas.begin(), alphas.end(), alphas_cumprod.begin(), std::multiplies<float>());
        
        // alphas_cumprod: 
        // pytorch: 0.9991, 0.9983, 0.9974 ...  0.0048, 0.0047, 0.0047
        // C++:     0.99915, 0.998296, 0.997438 ... 0.00477389, 0.00471669, 0.00466009
        // std::cout << "alphas_cumprod: " << std::endl;
        // for (int i = 0; i < num_train_timesteps; ++i) {
        //     std::cout << alphas_cumprod[i] << ", ";
        // }
        // std::cout << std::endl;
        final_alpha_cumprod = set_alpha_to_one ? 1.0 : alphas_cumprod[0];
        initNoiseSigma = 1.0;

        timesteps.resize(num_train_timesteps);
        std::iota(timesteps.rbegin(), timesteps.rend(), 0);
        // std::cout << "timesteps: " << std::endl;
        // for (int i = 0; i < num_train_timesteps; ++i) {
        //     std::cout << timesteps[i] << ", ";
        // }
        // std::cout << std::endl;
        _step_index = 0; 
    }
    
    // Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    // TODO: has bug
    void _init_step_index(int timestep) {
    auto iter = std::find(timesteps.begin(), timesteps.end(), timestep);
        if (iter != timesteps.end()) {
            size_t index = std::distance(timesteps.begin(), iter);
            std::cout << "index: " << index << "\n";

            if (index < timesteps.size() ) {
                _step_index = index + 1;
            } else {
                _step_index = index;
            }
        }
    }

    int step_index() const {
        return _step_index;
    }

    FloatTensor scale_model_input(const FloatTensor& sample, int timestep = -1) const {
        return sample;
    }

    // Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    std::vector<float> _threshold_sample(const std::vector<float>& flat_sample) {
        /* 
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."
        https://arxiv.org/abs/2205.11487
        */

        std::vector<float> thresholded_sample;
        // Calculate abs
        std::vector<float> abs_sample(flat_sample.size());
        std::transform(flat_sample.begin(), flat_sample.end(), abs_sample.begin(), [](float val) { return std::abs(val); });

        // Calculate s, the quantile threshold
        std::sort(abs_sample.begin(), abs_sample.end());
        const int s_index = std::min(static_cast<int>(std::round(dynamic_thresholding_ratio * flat_sample.size())), 
                                    static_cast<int>(flat_sample.size()) - 1);
        float s = abs_sample[s_index];
        s = std::clamp(s, 1.0f, sample_max_value);

        // Threshold and normalize the sample
        for (float& value : thresholded_sample) {
            value = std::clamp(value, -s, s) / s;
        }

        return thresholded_sample;
    }

    // Sets the discrete timesteps used for the diffusion chain (to be run before inference).
    std::vector<int> set_timesteps(
        int num_inference_steps,
        int original_inference_steps = -1,
        double strength = 1.0
    ) {
        if (num_inference_steps > num_train_timesteps_config) {
            throw std::invalid_argument("num_inference_steps cannot be larger than num_train_timesteps.");
        }

        num_inference_steps_ = num_inference_steps;
        int original_steps = (original_inference_steps != -1) ? original_inference_steps : original_inference_steps_config;

        if (original_steps > num_train_timesteps_config) {
            throw std::invalid_argument("original_steps cannot be larger than num_train_timesteps.");
        }

        if (num_inference_steps > original_steps) {
            throw std::invalid_argument("num_inference_steps cannot be larger than original_inference_steps.");
        }

        // LCM Timesteps Setting
        // The skipping step parameter k from the paper.
        int k = num_train_timesteps_config / original_steps;
        // std::cout << "num_train_timesteps_config: " << num_train_timesteps_config << 
        // ", original_steps: " << original_steps <<", K: " << k << "\n";

        std::vector<int> lcm_origin_timesteps;

        for (int i = 1; i <= static_cast<int>(original_steps * strength); ++i) {
            lcm_origin_timesteps.push_back(i * k - 1);
        }
        // std::cout << "lcm_origin_timesteps: " << std::endl;
        // for (float value : lcm_origin_timesteps) {
        //     std::cout << value << " ";
        // }
        // std::cout << std::endl;
        int skipping_step = lcm_origin_timesteps.size() / num_inference_steps;
        // std::cout << "skipping_step: " << skipping_step << "\n";

        if (skipping_step < 1) {
            throw std::invalid_argument("Invalid combination of original_steps and strength.");
        }

        // LCM Inference Steps Schedule
        std::reverse(lcm_origin_timesteps.begin(), lcm_origin_timesteps.end());

        // std::cout << "reverse lcm_origin_timesteps: " << std::endl;
        // for (float value : lcm_origin_timesteps) {
        //     std::cout << value << " ";
        // }
        // std::cout << std::endl;

        std::vector<int> inference_indices;
        inference_indices.reserve(static_cast<size_t>(num_inference_steps));

        for (double i = 0; i < num_inference_steps; ++i) {
            int index = static_cast<int>(std::floor(i * lcm_origin_timesteps.size() / num_inference_steps));
            inference_indices.push_back(index);
        }

        timesteps.clear();
        timesteps.reserve(inference_indices.size());
        for (int index : inference_indices) {
            timesteps.push_back(lcm_origin_timesteps[index]);
        }

        // Print the result for demonstration purposes
        // std::cout << "Timesteps: ";
        // for (int timestep : timesteps) {
        //     std::cout << timestep << " ";
        // }
        // std::cout << std::endl;
        return timesteps;
    }

    
    std::pair<float, float> get_scalings_for_boundary_condition_discrete(int timestep) {
        float scaled_timestep = timestep * timestep_scaling;
        float sigma_data = 0.5; // Default: 0.5

        float c_skip = sigma_data * sigma_data / (scaled_timestep * scaled_timestep + sigma_data * sigma_data);
        float c_out = scaled_timestep / sqrt(scaled_timestep * scaled_timestep + sigma_data * sigma_data);


        return {c_skip, c_out};
    }

    // for torch.randn()
    std::vector<float> randn_function(uint32_t size, uint32_t seed) {
        std::vector<float> noise(size);
        {
            std::mt19937 gen{static_cast<unsigned long>(seed)};
            std::normal_distribution<float> normal{0.0f, 1.0f};
            std::for_each(noise.begin(), noise.end(), [&](float& x) {
                x = normal(gen);
            });
        }
        return noise;
    }
    
    LCMSchedulerOutput step_func(const std::vector<float>& model_output, int timestep, 
                            const std::vector<float>& sample, int seed) {
        // Predict the sample from the previous timestep by reversing the SDE. 
        // if (num_inference_steps == 0) {
        //     throw std::runtime_error("Number of inference steps is 0. Run 'set_timesteps' after creating the scheduler.");
        // }
        // std::cout << "_step_index before _init_step_index: " << _step_index << std::endl;
        // if (_step_index == -1) {
        //     _init_step_index(timestep);
        // }
        // std::cout << "_step_index after _init_step_index: " << _step_index << std::endl;

        // std::cout << "1. get previous step value\n";

        // 1. get previous step value
        int prev_step_index = _step_index + 1;
        int prev_timestep = (prev_step_index < timesteps.size()) ? timesteps[prev_step_index] : timestep;
        // std::cout << "timesteps.size(): "<< timesteps.size() << ", timesteps[prev_step_index]: " << timesteps[prev_step_index]<<  "\n";
        // std::cout << "_step_index: " << _step_index << std::endl;

        // std::cout << "2. compute alphas, betas\n";

        // 2. compute alphas, betas
        float alpha_prod_t = alphas_cumprod[timestep];
        float alpha_prod_t_prev = (prev_timestep >= 0) ? alphas_cumprod[prev_timestep] : final_alpha_cumprod;
        // std::cout << "timestep: " << timestep << ", prev_timestep: " << prev_timestep << ", final_alpha_cumprod: " << final_alpha_cumprod << "\n";

        float beta_prod_t = 1 - alpha_prod_t;
        float beta_prod_t_prev = 1 - alpha_prod_t_prev;

        // std::cout << "alpha_prod_t: " << alpha_prod_t << ", alpha_prod_t_prev: " << alpha_prod_t_prev << "\n";
        // std::cout << "beta_prod_t: " << beta_prod_t << ", beta_prod_t_prev: " << beta_prod_t_prev << "\n";
        
        // std::cout << "3. Get scalings for boundary conditions\n";

        // 3. Get scalings for boundary conditions
        std::pair c_pair = get_scalings_for_boundary_condition_discrete(timestep);
        float c_skip = c_pair.first;
        float c_out = c_pair.second; 
        // std::cout << "c_skip: " << c_skip << ", c_out: " << c_out << "\n";

        // std::cout << "4. Compute the predicted original sample\n";

        // 4. Compute the predicted original sample x_0 based on the model parameterization
        // default "epsilon": noise-prediction
        std::vector<float> predicted_original_sample;
        predicted_original_sample.resize(sample.size());

        if (prediction_type_config == "epsilon") {  // noise-prediction
            float beta_prod_t_sqrt = static_cast<float>(sqrt(beta_prod_t));
            float alpha_prod_t_sqrt = sqrt(alpha_prod_t);
            for (std::size_t i = 0; i < sample.size(); ++i) {
                predicted_original_sample[i] = (sample[i] - beta_prod_t_sqrt * model_output[i]) / alpha_prod_t_sqrt;
            }
        }
     
        // std::cout << "predicted_original_sample: \n"; 
        // for (std::size_t i = 0; i < 5; ++i) {
        //     std::cout << predicted_original_sample[i] << ", ";
        // }        
        // std::cout << "\n"; 

        // std::cout << "5. Clip or threshold predicted x_0\n";

        // 5. Clip or threshold "predicted x_0"
        if (thresholding) {
            predicted_original_sample = _threshold_sample(predicted_original_sample);
        } else if (clip_sample) {
            for (float& value : predicted_original_sample) {
                value = std::clamp(value, - clip_sample_range, clip_sample_range);
            }
        }

        // std::cout << "predicted_original_sample after clip: ";
        // for (std::size_t i = 0; i < 5; ++i) {
        //     std::cout << predicted_original_sample[i] << ", ";
        // }        
        // std::cout << "\n"; 

        // std::cout << "6. Denoise model output using boundary conditions\n";

        // 6. Denoise model output using boundary conditions
        std::vector<float> denoised(predicted_original_sample.size());
        for (std::size_t i = 0; i < predicted_original_sample.size(); ++i) {
            denoised[i] = c_out * predicted_original_sample[i] + c_skip * sample[i];
        }

        // std::cout << "denoised: ";
        // for (std::size_t i = 0; i < 5; ++i) {
        //     std::cout << denoised[i] << ", ";
        // }    
        // std::cout << "\n"; 

        // std::cout << "7. Sample and inject noise\n";

        // 7. Sample and inject noise z ~ N(0, I) for MultiStep Inference
        // Noise is not used on the final timestep of the timestep schedule.
        // This also means that noise is not used for one-step sampling.
        std::vector<float> prev_sample;
        // std::cout << "_step_index: " << _step_index << ", num_inference_steps_: " << num_inference_steps_ << "\n";
        if (_step_index != num_inference_steps_ - 1) {
            
            std::vector<float> noise;

            if (read_noise_config == true) {
                // read for lcm pipeline
                std::string file_path = "../scripts/torch_noise_step_" + std::to_string(_step_index) + ".txt"; 
                noise = read_randn_function(file_path);
            } else {
                noise = randn_function(model_output.size(), seed);
                // std::cout << "randn_function: " << _step_index << "\n";
            }

            // read noise for lcm_scheduler.cpp

            // std::vector<float> noise;
            // if (_step_index == 0) {
            //     noise = { 1.5410, -0.2934, -2.1788,  0.5684 };
            // } else {
            //     noise = { -1.0845, -1.3986,  0.4033,  0.8380 };
            // }

            // std::cout << "randn noise:";
            // for (std::size_t i = 0; i < model_output.size(); ++i) {
            //     std::cout << noise[i] << ", ";
            // }
            // std::cout << "\n";
            for (std::size_t i = 0; i < model_output.size(); ++i) {
                prev_sample.push_back(sqrt(alpha_prod_t_prev) * denoised[i] + sqrt(beta_prod_t_prev) * noise[i]);
            }

        } else {
            prev_sample = denoised;
        }

        // upon completion increase step index by one
        _step_index += 1;
        // std::cout << "finished step()\n";

        return LCMSchedulerOutput(prev_sample, denoised);
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

