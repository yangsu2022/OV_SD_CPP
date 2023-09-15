#include <cstdio>
#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <openvino/openvino.hpp>
#include "openvino/pass/manager.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace py = pybind11;


class InsertLoRA : public ov::pass::MatcherPass {
    public:
        OPENVINO_RTTI("InsertLoRA","0");
        std::map<std::string, std::vector<float>>* local_lora_map;
        explicit InsertLoRA(std::map<std::string, std::vector<float>>& lora_map){
            local_lora_map=&lora_map;
            auto label = ov::pass::pattern::wrap_type<ov::op::v0::Convert>();
            //auto label = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
            ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
                auto root = std::dynamic_pointer_cast<ov::op::v0::Convert>(m.get_match_root());
                //auto root = std::dynamic_pointer_cast<ov::op::v0::Constant>(m.get_match_root());
                if (!root) {
                    return false;
                }
                ov::Output<ov::Node> root_output = m.get_match_value();
                std::string root_name = root->get_friendly_name();
                std::replace(root_name.begin(),root_name.end(),'.','_');
                std::map<std::string,std::vector<float>>::iterator it = local_lora_map->begin();
                while(it!=local_lora_map->end()){
                    if((root_name).find(it->first) != std::string::npos){
                        // std::cout << root_name << std::endl;
                        std::set<ov::Input<ov::Node>> consumers = root_output.get_target_inputs();
                        std::shared_ptr<ov::Node> lora_const = ov::op::v0::Constant::create(ov::element::f32,ov::Shape{root->get_output_shape(0)},it->second);
                        //std::cout << "lora_const:" << lora_const->get_output_shape(0) << std::endl;
                        auto lora_add = std::make_shared<ov::opset11::Add>(root,lora_const);
                        for (auto consumer:consumers){
                            consumer.replace_source_output(lora_add->output(0));
                        }
                        register_new_node(lora_add);
                        it = local_lora_map->erase(it);
                    }
                    else{
                        it++;
                    }  
                }
                return true;
            };
            // Register pattern with Parameter operation as a pattern root node
            auto m = std::make_shared<ov::pass::pattern::Matcher>(label, "InsertLoRA");
            // Register Matcher
            register_matcher(m, callback);
        }
};

std::vector<ov::CompiledModel> load_lora_weights(ov::Core& core, std::shared_ptr<ov::Model>& text_encoder_model, std::shared_ptr<ov::Model>& unet_model, std::string& device, std::map<std::string,float>& lora_models){
    py::scoped_interpreter guard{};
    std::vector<ov::CompiledModel> compiled_lora_models;
    std::string LORA_PREFIX_UNET = "lora_unet";
    std::string LORA_PREFIX_TEXT_ENCODER = "lora_te";
    // load_lora_file.py in the src dir instead of build dir
    py::module_ path = py::module_::import("os.path");
    std::string curdir_abs = path.attr("abspath")(path.attr("curdir")).cast<std::string>();
    std::regex build_regex("build");
    auto srcdir = std::regex_replace(curdir_abs, build_regex, "src");
    auto append_path =py::module_::import("sys").attr("path").attr("append")(srcdir.c_str());
    
    py::module_ load_lora_file = py::module_::import("load_lora_file");
    if(!lora_models.empty()){
        std::vector<std::string> encoder_layers;
        std::vector<std::string> unet_layers;

        std::map<std::string, std::vector<float>> lora_map;
        ov::pass::Manager manager;
        int flag = 0;
        try {
            auto start = std::chrono::steady_clock::now();
            for (std::map<std::string,float>::iterator it=lora_models.begin(); it!=lora_models.end(); ++it){
                auto py_lora_tensor = load_lora_file.attr("load_safetensors")(it->first);
                auto py_lora_weights = load_lora_file.attr("get_lora_weights")(py_lora_tensor,it->second);
                //-------------get text encoder vectors-----------
                encoder_layers =  load_lora_file.attr("get_encoder_layers")(py_lora_weights).cast<std::vector<std::string>>();
                auto py_encoder_weights = load_lora_file.attr("get_encoder_weights")(py_lora_weights);
                flag = 0;
                for(auto item: py_encoder_weights){
                    auto buffer = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(item);
                    std::vector<float> d(buffer.data(), buffer.data()+buffer.nbytes()/buffer.itemsize());
                    lora_map.insert(std::pair<std::string,std::vector<float>>(encoder_layers[flag],d));
                    flag++;
                }
                //-------------get unet vectors-------------------
                unet_layers =  load_lora_file.attr("get_unet_layers")(py_lora_weights).cast<std::vector<std::string>>();
                auto py_unet_weights = load_lora_file.attr("get_unet_weights")(py_lora_weights);
                //unet_layers = py_unet_layers.cast<std::vector<std::string>>();
                flag = 0;
                for(auto item: py_unet_weights){
                    auto buffer = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(item);
                    std::vector<float> d(buffer.data(), buffer.data()+buffer.nbytes()/buffer.itemsize());
                    lora_map.insert(std::pair<std::string,std::vector<float>>(unet_layers[flag],d));
                    flag++;
                }
                auto end = std::chrono::steady_clock::now();
                std::cout << "lora_extract:" << std::chrono::duration <double, std::milli> (end-start).count() << " ms" << std::endl;
                /*for(auto ii:lora_map){
                    std::cout << ii.first << ":" << ii.second.size() << std::endl;
                }*/
            }
            manager.register_pass<InsertLoRA>(lora_map);
            auto start_txt = std::chrono::steady_clock::now();
            if(!encoder_layers.empty()){
                manager.run_passes(text_encoder_model);
            }
            auto end_txt = std::chrono::steady_clock::now();
            compiled_lora_models.push_back(core.compile_model(text_encoder_model, device));
            auto start_unet = std::chrono::steady_clock::now();
            manager.run_passes(unet_model);
            auto end_unet = std::chrono::steady_clock::now();
            compiled_lora_models.push_back(core.compile_model(unet_model, device));

            std::cout << "text_encoder run pass:" << std::chrono::duration <double, std::milli> (end_txt-start_txt).count() << " ms" << std::endl;
            std::cout << "unet run pass:" << std::chrono::duration <double, std::milli> (end_unet-start_unet).count() << " ms" << std::endl;
            
        }
        catch (...) {
            std::cout<< "Error loading the lora model. Please check your SD IR model or lora weights file(*.safetensors).\n";
            compiled_lora_models.push_back(core.compile_model(text_encoder_model, device));
            compiled_lora_models.push_back(core.compile_model(unet_model, device));
            return compiled_lora_models;
        }

    }

    return compiled_lora_models;
}