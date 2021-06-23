//
// Created by sshuair on 2021/4/8.
//

#include "mmseg_deploy/NetTorch.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <torch/torch.h>
#include <torch/script.h>

namespace mmseg {
  ModelLoader::ModelLoader(const std::string& model_path,const std::string& cfg_path): 
  _model_path (model_path),_cfg_path(cfg_path),_verbose(true){
      
      // set default verbosity level
    verbosity(_verbose);

    try {
      try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        _model = torch::jit::load(_model_path,torch::kCUDA);
        _device = std::unique_ptr<torch::Device>(new torch::Device(torch::kCUDA));
      }
      catch (...) {
      std::cout << "Could not send model to GPU, using CPU" << std::endl;
      _model = torch::jit::load(_model_path, torch::kCPU);
      _device = std::unique_ptr<torch::Device>(new torch::Device(torch::kCPU));
      }
    }
    catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    }

    

    try {
      _cfg = YAML::LoadFile(_cfg_path);
    } catch (YAML::Exception& ex) {
      throw std::runtime_error("Can't open cfg.yaml from " + _cfg_path);
    }

    // get image size
  _img_h = _cfg["dataset"]["img_prop"]["height"].as<int>();
  _img_w = _cfg["dataset"]["img_prop"]["width"].as<int>();
  _img_d = _cfg["dataset"]["img_prop"]["depth"].as<int>();

  // get normalization parameters
  YAML::Node img_means, img_stds;
  try {
    img_means = _cfg["dataset"]["img_means"];
    img_stds = _cfg["dataset"]["img_stds"];
  } catch (YAML::Exception& ex) {
    std::cerr << "Can't open one the mean or std dictionary from cfg"
              << std::endl;
    throw ex;
  }
  // fill in means from yaml node
  YAML::const_iterator it;
  for (it = img_means.begin(); it != img_means.end(); ++it) {
    // Get value
    float mean = it->as<float>();
    // Put in indexing vector
    _img_means.push_back(mean);
  }
  // fill in stds from yaml node
  for (it = img_stds.begin(); it != img_stds.end(); ++it) {
    // Get value
    float std = it->as<float>();
    // Put in indexing vector
    _img_stds.push_back(std);
  }
  }

  ModelLoader::~ModelLoader()  {}
 
  cv::Mat ModelLoader::preprocess(const cv::Mat& image) {
    // create temp mat to fill in with processed data
    cv::Mat preprocessed;

    // resize if image is not as desired
    if (image.rows != _img_h || image.cols != _img_w) {
      // Complain that I am resizing just to make sure the user knows
      if (_verbose) {
        std::cout << "Watch out, I'm resizing internally. Input should be "
                  << _img_h << "x" << _img_w << ", but is " << image.rows << "x"
                  << image.cols << std::endl;
      }

      // resize
      cv::resize(image, preprocessed, cv::Size(_img_w, _img_h), 0, 0,
                cv::INTER_LINEAR);
    } else {
      // just put in preprocessed
      preprocessed = image.clone();
    }

    // Make float
    preprocessed.convertTo(preprocessed, CV_32F);

    // return the vector organized as CHW, normalized, and as float
    // RVO should move this
    return preprocessed;
  }

  cv::Mat ModelLoader::postprocess(const cv::Mat& img, const cv::Mat& argmax) {
    // create temp mat to fill in with processed data
    cv::Mat postprocessed;

    // resize if image is not as desired
    if (img.rows != argmax.rows || img.cols != argmax.cols) {
      // Complain that I am resizing just to make sure the user knows
      if (_verbose) {
        std::cout << "Watch out, I'm resizing output internally (NN)."
                  << std::endl;
      }

      // resize
      cv::resize(argmax, postprocessed, cv::Size(img.cols, img.rows), 0, 0,
                cv::INTER_NEAREST);
    } else {
      // just put in preprocessed
      postprocessed = argmax.clone();
    }

    // RVO should move this
    return postprocessed;
  }
    

  void ModelLoader::Mat2Tensor(const cv::Mat& img, torch::Tensor& tensor_image, bool unsqueeze, int64_t unsqueeze_dim){
    
    tensor_image = torch::from_blob(img.data, { img.rows, img.cols, 3 }, at::kFloat);
    tensor_image = tensor_image.to(*_device);
    tensor_image = tensor_image.permute(torch::IntArrayRef {2,0,1});
    if (unsqueeze)
    {
      tensor_image.unsqueeze_(unsqueeze_dim);
    }
    
    
    tensor_image[0][0] = tensor_image[0][0].div(255).sub(_img_means[0]).div(_img_stds[0]);;
    tensor_image[0][1] = tensor_image[0][1].div(255).sub(_img_means[1]).div(_img_stds[1]);
    tensor_image[0][2] = tensor_image[0][2].div(255).sub(_img_means[2]).div(_img_stds[2]);
    
  }

  
  //torch::Tensor inference(torch::jit::script::Module* model, const torch::Tensor& input_tensor,  const std::string& device, float_t threshold){
  cv::Mat ModelLoader::inference( const cv::Mat& img){

    
    cv::Mat input_preproc = preprocess(img);
    torch::Tensor input_tensor;
    
    Mat2Tensor(input_preproc, input_tensor ,true);
    if (_verbose)
      std::cout<<input_tensor.sizes()<<std::endl;
    
    
    
    if (_verbose)
      std::cout << *_device<<" "<<input_tensor.device() << std::endl;
    auto begin = std::chrono::high_resolution_clock::now();
    torch::Tensor out_tensor = _model.forward({input_tensor}).toTensor();
    auto end = std::chrono::high_resolution_clock::now();
    torch::Tensor argmax_tensor = out_tensor.argmax(1, false);
    if (_verbose) {
    std::cout << "Time to infer: "
              << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                      begin)
                         .count() /
                     1000000.0
              << "ms" << std::endl; 
    std::cout<<"the argmax shape is :"<<argmax_tensor.sizes()<<"\n";
              }

   

    

    // torch::Tensor softmax_person_tensor;
    // mappingLabel();
    // for (int i = 0;i<_labels_map.size();i++) {
    //   if(_labels_map[i]=="person")
    //     softmax_person_tensor = argmax_tensor[0][i];
      
    // }
    
    
    cv::Mat out_img = Tensor2Mat(argmax_tensor);
    
    
    out_img = postprocess(img,out_img);
    
    
    
    return out_img;
  }


  cv::Mat ModelLoader::Tensor2Mat(torch::Tensor& tensor)
  {
      
    if (_verbose)
      std::cout<<tensor.sizes()<<std::endl;
    
    tensor=tensor.toType(at::kByte);
    tensor = tensor.to(torch::kCPU);
    

    try
    {
        cv::Mat output_mat(cv::Size{ _img_w, _img_h }, CV_8UC1, tensor.data_ptr<uchar>());
        return output_mat.clone();
    }
    catch (const torch::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
  }

  void ModelLoader::mappingLabel() {
    
    // Get label dictionary from yaml cfg
    YAML::Node labels;
    try {
      labels = _cfg["dataset"]["labels"];
    } catch (YAML::Exception& ex) {
      std::cerr << "Can't open one the labels from cfg in " + _cfg_path
                << std::endl;
      throw ex;
    }
    
    
    // Generate string map from xentropy indexes (that we'll get from argmax)
    YAML::const_iterator it;
    _labels_map.resize(labels.size());
    
    for (it = labels.begin(); it != labels.end(); ++it) {
      // Get label and key
      
      int key = it->first.as<int>();  // <- key
      
      std::string label = labels[key].as<std::string>();
          
      
      _labels_map[key] = label;
    }
  
      
  }

  void ModelLoader::mappingColor() {
    
    // Get label dictionary from yaml cfg
    YAML::Node color_map;
    try {
      color_map = _cfg["dataset"]["color_map"];
    } catch (YAML::Exception& ex) {
      std::cerr << "Can't open one the label dictionary from cfg in " + _cfg_path
                << std::endl;
      throw ex;
    }
    
    
    // Generate string map from xentropy indexes (that we'll get from argmax)
    YAML::const_iterator it;
    _argmax_to_bgr.resize(color_map.size());
    
    for (it = color_map.begin(); it != color_map.end(); ++it) {
      // Get label and key
      
      int key = it->first.as<int>();  // <- key
      cv::Vec3b color = {
          static_cast<uint8_t>(color_map[key][0].as<unsigned int>()),
          static_cast<uint8_t>(color_map[key][1].as<unsigned int>()),
          static_cast<uint8_t>(color_map[key][2].as<unsigned int>())};
      
      _argmax_to_bgr[key] = color;
    }
  
      
  }

  cv::Mat ModelLoader::color(const cv::Mat& argmax) {
    cv::Mat colored(argmax.rows, argmax.cols, CV_8UC3);
    mappingColor();
    
    colored.forEach<cv::Vec3b>(
        [&](cv::Vec3b& pixel, const int* position) -> void {
          // argmax ist int32, ojo!
          int c = argmax.at<int8_t>(cv::Point(position[1], position[0]));
          pixel = _argmax_to_bgr[c];
        });
    return colored;
  }

}