#pragma once

 //ROS
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <ros/ros.h>
#include <mmseg_deploy/NetTorch.hpp>

#include <opencv2/imgcodecs.hpp> 

#include <actionlib/server/simple_action_server.h>
#include "mmseg_deploy/mmSegmentationAction.h"
#include "mmseg_deploy/mmSegmentation.h"
class MmsegHandler {
    public:
        /*!
        * Constructor.
        *
        * @param      nodeHandle  the ROS node handle.
        */
        MmsegHandler (ros::NodeHandle& nodeHandle);

        

            /*!
        * Destructor.
        */
        virtual ~MmsegHandler();

        /*!
        * loads the model path and model pt path.
        */
        void init();
    
    private:
        /*!
        * Reads and verifies the ROS parameters.
        *
        * @return     true if successful.
        */
        bool readParameters();

        /*!
        * ROS topic callback method.
        *
        * @param[in]  img_msg  The image message (to infer)
        */
        void imageCallback(const sensor_msgs::ImageConstPtr& img_msg);

        /*!
        * \brief executeCB for bonnet action
        * \param goal
        */
        void executeCB(const mmseg_deploy::mmSegmentationGoalConstPtr& goal);

        //! ROS node handle.
        ros::NodeHandle& _node_handle;

        //! ROS topic subscribers and publishers.
        image_transport::ImageTransport _it;
        image_transport::Subscriber _img_subscriber;
        image_transport::Publisher _argmax_publisher;
        image_transport::Publisher _color_publisher;
        

        //! ROS action server
        //! Using.
        using mmSegmentationActionServer = actionlib::SimpleActionServer<mmseg_deploy::mmSegmentationAction>;
        using mmSegmentationActionServerPtr = std::shared_ptr<mmSegmentationActionServer>;
        //! Segmantation action server.
        mmSegmentationActionServerPtr segmentationActionServer_;
        mmseg_deploy::mmSegmentationResult ac_result_;
        mmseg_deploy::mmSegmentationFeedback ac_feedback_;
        std::string action_name_;  

        ros::ServiceServer segmentationService_;
        bool serviceCallback(mmseg_deploy::mmSegmentation::Request &req, mmseg_deploy::mmSegmentation::Response &res);
        
        //! ROS topic names to subscribe to.
        std::string _input_image_topic;
        std::string _argmax_topic;
        std::string _color_topic;
        

        //! CNN related stuff
        std::unique_ptr<mmseg::ModelLoader> _net;
        std::string _model_path;
        bool _verbose;
        std::string _cfg_path;
};   
