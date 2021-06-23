// STD
#include <unistd.h>
#include <string>

#include<sensor_msgs/image_encodings.h>

// net stuff
#include "mmseg_deploy/mmseg_handler.hpp"


MmsegHandler::MmsegHandler(ros::NodeHandle& nodeHandle)
    : _node_handle(nodeHandle),_it(nodeHandle) {
        // Try to read the necessary parameters
        if (!readParameters()) {
            ROS_ERROR("Could not read parameters.");
            ros::requestShutdown();
        }

        // Subscribe to images to infer
        ROS_INFO("Subscribing to image topics.");
        _img_subscriber =
            _it.subscribe(_input_image_topic, 1, &MmsegHandler::imageCallback, this);

        // Advertise our topics
        ROS_INFO("Advertising our outputs.");
        // Advertise our topics
        _argmax_publisher = _it.advertise(_argmax_topic, 1);
        _color_publisher = _it.advertise(_color_topic, 1);

        ROS_INFO("Generating CNN and setting verbosity");
        // create a network
        _net = std::unique_ptr<mmseg::ModelLoader> (new mmseg::ModelLoader(_model_path ,_cfg_path));
        
        _net->verbosity(_verbose);

        ROS_INFO("Initializing ActionServer");
        //Init action server
        segmentationActionServer_.reset(new mmSegmentationActionServer(_node_handle, "action",boost::bind(&MmsegHandler::executeCB, this, _1), false));
        segmentationActionServer_->start();

        segmentationService_ = _node_handle.advertiseService("mmsegmentation", &MmsegHandler::serviceCallback, this); 


        ROS_INFO("Successfully launched node.");
    }

    /*!
 * Destructor.
 */
MmsegHandler::~MmsegHandler() {}

bool MmsegHandler::readParameters() {
  if (!_node_handle.getParam("model_path", _model_path) ||
      !_node_handle.getParam("verbose", _verbose) ||
      !_node_handle.getParam("cfg_path", _cfg_path) ||
      !_node_handle.getParam("input_image", _input_image_topic) ||
      !_node_handle.getParam("output_argmax", _argmax_topic) ||
      !_node_handle.getParam("output_color", _color_topic) )
      
    return false;
  return true;
}

void MmsegHandler::imageCallback(const sensor_msgs::ImageConstPtr& img_msg) {
  if (_verbose) {
    // report that we got something
    ROS_INFO("Image received.");
    ROS_INFO("Image encoding: %s", img_msg->encoding.c_str());
  }

  // Only do inference if somebody is subscribed (save energy and temp)
  uint32_t argmax_subs = _argmax_publisher.getNumSubscribers();
  uint32_t color_subs = _color_publisher.getNumSubscribers();
  
  uint32_t total_subs = argmax_subs + color_subs ;

  if (_verbose) {
    std::cout << "Subscribers:  " << std::endl
              << "Argmax: " << argmax_subs << std::endl
              << "Color: " << color_subs << std::endl;
             
  }

  if (total_subs > 0) {
    // Get the image
    cv_bridge::CvImageConstPtr cv_img;
    cv_img = cv_bridge::toCvShare(img_msg);

    // change to bgr according to encoding
    cv::Mat cv_img_rgb(cv_img->image.rows, cv_img->image.cols, CV_8UC3);
    ;
    if (img_msg->encoding == "bayer_rgb8") {
      if (_verbose) ROS_INFO("Converting BAYER_RGGB8 to RGB for CNN");
      cv::cvtColor(cv_img->image, cv_img_rgb, cv::COLOR_BayerBG2RGB);
    } else if (img_msg->encoding == "bgr8") {
      if (_verbose) ROS_INFO("Converting BGR8 to RGB for CNN");
      cv::cvtColor(cv_img->image, cv_img_rgb, cv::COLOR_BGR2RGB);
    } else if (img_msg->encoding == "rgb8") {
      if (_verbose) ROS_INFO("Loading RGB8 for CNN");
      cv_img_rgb = cv_img->image;
    } else {
      if (_verbose) ROS_ERROR("Colorspace conversion non implemented. Skip...");
      return;
    }

 // infer 
 cv::Mat predMax = _net->inference(cv_img_rgb);

 // save the image
 
//  std::string out_fp("/home/wms/object_segmentation/mmseg_ws/src/mmseg_deploy/config/test.png");
//  cv::imwrite(out_fp, predMax);

if (argmax_subs > 0) {
    // Send the argmax (changing it to depth 8, wms needs that )
    

    sensor_msgs::ImagePtr argmax_msg =
        cv_bridge::CvImage(img_msg->header, "mono8", predMax).toImageMsg();
    cv::imwrite("/home/wms/object_segmentation/mmseg_ws/src/mmseg_deploy/pred_arg.jpg",predMax);
    _argmax_publisher.publish(argmax_msg);
}

if (color_subs > 0 ) {
    // get color
    cv::Mat color_mask = _net->color(predMax);
    cv::imwrite("/home/wms/object_segmentation/mmseg_ws/src/mmseg_deploy/pred_color.jpg",color_mask);
    // // save the image
    // std::string out_cp("/home/wms/object_segmentation/mmseg_ws/src/mmseg_deploy/config/testcolor.png");
    // cv::imwrite(out_cp, color_mask);

    // Send the color mask
    sensor_msgs::ImagePtr color_msg =
        cv_bridge::CvImage(img_msg->header, "bgr8", color_mask)
            .toImageMsg();
    _color_publisher.publish(color_msg);
    
 }
}
}

void MmsegHandler::executeCB(const mmseg_deploy::mmSegmentationGoalConstPtr& goal) {
    ROS_INFO("Action called");

    if (_verbose) {
        // report that we got something
        ROS_INFO("Image received.");
        ROS_INFO("Image encoding: %s", goal->image.encoding.c_str());
    }

    // Get the image
    cv_bridge::CvImageConstPtr cv_img;
    try
    {
        cv_img = cv_bridge::toCvCopy(goal->image, sensor_msgs::image_encodings::RGB8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        segmentationActionServer_->setAborted();
        return;
    }

    // change to bgr according to encoding
    cv::Mat cv_img_rgb(cv_img->image.rows, cv_img->image.cols, CV_8UC3);
    ;
    if (goal->image.encoding == "bayer_rgb8") {
      if (_verbose) ROS_INFO("Converting BAYER_RGGB8 to RGB for CNN");
      cv::cvtColor(cv_img->image, cv_img_rgb, cv::COLOR_BayerBG2RGB);
    } else if (goal->image.encoding == "bgr8") {
      if (_verbose) ROS_INFO("Converting BGR8 to RGB for CNN");
      cv::cvtColor(cv_img->image, cv_img_rgb, cv::COLOR_BGR2RGB);
    } else if (goal->image.encoding == "rgb8") {
      if (_verbose) ROS_INFO("Converting RGB8 to RGB for CNN");
      cv_img_rgb=cv_img->image;
    } else {
      if (_verbose) ROS_ERROR("Colorspace conversion non implemented. Skip...");
      return;
    }

    // infer
    cv::Mat softmax = _net->inference(cv_img_rgb);

    // Send the argmax (changing it to depth 16, since 32 can't be done)
    cv::Mat softmax_8;
    softmax.convertTo(softmax_8, CV_8UC1);


    sensor_msgs::ImagePtr softmax_msg =
        cv_bridge::CvImage(goal->image.header, "mono8", softmax_8).toImageMsg();

    ac_result_.image = *softmax_msg;

    segmentationActionServer_->setSucceeded(ac_result_,"Send segmentation image.");

}

bool MmsegHandler::serviceCallback(mmseg_deploy::mmSegmentation::Request &req, mmseg_deploy::mmSegmentation::Response &res) {
  if (_verbose) {
    // report that we got something
    ROS_INFO("Image received.");
    ROS_INFO("Image encoding: %s", req.image.encoding.c_str());
  }

  // Get the image
  cv_bridge::CvImageConstPtr cv_img;

  try
  {
    cv_img = cv_bridge::toCvCopy(req.image, sensor_msgs::image_encodings::RGB8);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return false;
  }

  // change to bgr according to encoding
    cv::Mat cv_img_rgb(cv_img->image.rows, cv_img->image.cols, CV_8UC3);
    ;
    if (req.image.encoding == "bayer_rgb8") {
        if (_verbose) ROS_INFO("Converting BAYER_RGGB8 to RGB for CNN");
        cv::cvtColor(cv_img->image, cv_img_rgb, cv::COLOR_BayerBG2RGB);
    } else if (req.image.encoding == "bgr8") {
        if (_verbose) ROS_INFO("Converting BGR8 to RGB for CNN");
        cv::cvtColor(cv_img->image, cv_img_rgb, cv::COLOR_BGR2RGB);
    } else if (req.image.encoding == "rgb8") {
        if (_verbose) ROS_INFO("Converting RGB8 to RGB for CNN");
        cv_img_rgb=cv_img->image;
    } else {
        if (_verbose) ROS_ERROR("Colorspace conversion non implemented. Skip...");
        return false;
    }

  // infer
  cv::Mat softmax = _net->inference(cv_img_rgb);

  // Send the argmax (changing it to depth 16, since 32 can't be done)
  cv::Mat softmax_8;
  softmax.convertTo(softmax_8, CV_8UC1);


  sensor_msgs::ImagePtr softmax_msg =
     cv_bridge::CvImage(req.image.header, sensor_msgs::image_encodings::MONO8, softmax_8).toImageMsg();

  res.image = *softmax_msg;
  return true;
}