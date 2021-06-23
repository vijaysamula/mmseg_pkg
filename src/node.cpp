#include <ros/ros.h>
#include "mmseg_deploy/mmseg_handler.hpp"

int main(int argc, char** argv) {
  ros::init(argc, argv, "mmseg_node");
  ros::NodeHandle nodeHandle("~");

  // init handler
  MmsegHandler h(nodeHandle);

  ros::spin();
  return 0;
}
