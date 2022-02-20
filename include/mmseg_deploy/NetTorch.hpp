#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "yaml-cpp/yaml.h"

namespace mmseg {

class ModelLoader {

    public:
        
        ModelLoader(const std::string& model_path,const std::string& cfg_path);
        
        ~ModelLoader();
        
        void inference( const cv::Mat& img , cv::Mat& softmax_img, cv::Mat& argmax_img);

        cv::Mat color(const cv::Mat& argmax);

        cv::Mat Tensor2Mat(torch::Tensor& tensor);
    
        void verbosity(const bool verbose) { _verbose = verbose; }
    private:
        torch::jit::script::Module _model;
        std::unique_ptr<torch::Device> _device;
        std::vector<cv::Vec3b> _argmax_to_bgr; 
        std::vector<std::string> _labels_map; 
        YAML::Node _cfg;
        std::string _model_path;
        std::string _cfg_path;
        bool _verbose;
        int _img_h, _img_w, _img_d;  
        std::vector<float> _img_means, _img_stds;  
        
        cv::Mat preprocess(const cv::Mat& image);
        cv::Mat postprocess(const cv::Mat& img, const cv::Mat& argmax);
        void mappingColor();
        void mappingLabel();
        void Mat2Tensor(const cv::Mat &img, torch::Tensor& input_image, bool unsqueeze = true, int64_t unsqueeze_dim = 0);
};
}