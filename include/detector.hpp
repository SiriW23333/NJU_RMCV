#pragma once

#include <opencv2/opencv.hpp>
#include "armor.hpp"
#include <list>

namespace auto_aim
{

class Detector
{
public:
  Detector();
  
  std::list<Armor> detect(const cv::Mat & bgr_img);

private:
  bool check_geometry(const Lightbar & lightbar);
  bool check_geometry(const Armor & armor);
  bool check_name(const Armor & armor);
  
  Color get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour);
  
  // 修改这里：改为与实现匹配的声明
  void get_pattern(const cv::Mat & bgr_img, const Armor & armor, cv::Mat & pattern);
  
  // 确保这里只有声明，不包含实现
  void classify(Armor & armor);

  cv::dnn::Net net_; // 深度学习模型
};

} // namespace auto_aim

