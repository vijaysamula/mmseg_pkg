#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
// yamlcpp
#include "yaml-cpp/yaml.h"
//TODO: REFACTOR TO PIPLINE

namespace mmseg {

class ModelLoader {

    public:
        
        ModelLoader(const std::string& model_path,const std::string& cfg_path);
        
        ~ModelLoader();
        
        //inference the model
        cv::Mat inference(const cv::Mat& img);

        //
        cv::Mat color(const cv::Mat& argmax);

        //Convert the torch tensor to opencv Mat
        cv::Mat Tensor2Mat(torch::Tensor& tensor);
    
        void verbosity(const bool verbose) { _verbose = verbose; }
    private:
        torch::jit::script::Module _model;
        // device for inference
        std::unique_ptr<torch::Device> _device;
        std::vector<cv::Vec3b> _argmax_to_bgr; 
        std::vector<std::string> _labels_map; 
        YAML::Node _cfg;
        std::string _model_path;
        std::string _cfg_path;
        bool _verbose;
        // image properties
        int _img_h, _img_w, _img_d;  // height, width, and depth for inference
        std::vector<float> _img_means, _img_stds;  // mean and std per channel (RGB)
        
        cv::Mat preprocess(const cv::Mat& image);
        cv::Mat postprocess(const cv::Mat& img, const cv::Mat& argmax);
        void mappingColor();
        void mappingLabel();
        // Convert the Opencv Mat to Torch Tensor
        // from cv::Mat {height, width, channels} to torch::Tensor {1, channels, height, width}
        void Mat2Tensor(const cv::Mat &img, torch::Tensor& input_image, bool unsqueeze = true, int64_t unsqueeze_dim = 0);
};
}