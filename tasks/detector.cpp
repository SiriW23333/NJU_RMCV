#include "detector.hpp"
#include <opencv2/opencv.hpp>

#include <fmt/format.h>

#include "img_tools.hpp"

namespace auto_aim
{
  Detector::Detector()
  {
    // 初始化代码
    try {
      // 加载神经网络模型
      net_ = cv::dnn::readNetFromONNX("/home/wxy/NJU_RMCV/tiny_resnet.onnx");
    }
    catch (const cv::Exception& e) {
      std::cerr << "无法加载神经网络模型: " << e.what() << std::endl;
    }
  }

  std::list<Armor> Detector::detect(const cv::Mat &bgr_img)
  {
    // 彩色图转灰度图
    static cv::Mat gray_img, binary_img; // 静态变量避免重复分配内存
    gray_img.create(bgr_img.size(), CV_8UC1);
    binary_img.create(bgr_img.size(), CV_8UC1);
    cv::cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
    cv::threshold(gray_img, binary_img, 170, 255, cv::THRESH_BINARY);

    // 获取轮廓点
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary_img, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    // 获取灯条
    std::size_t lightbar_id = 0;
    std::vector<Lightbar> lightbars;
    lightbars.reserve(50); // 预分配内存
    for (const auto &contour : contours)
    {
      auto rotated_rect = cv::minAreaRect(contour);
      auto lightbar = Lightbar(rotated_rect, lightbar_id);

      if (!check_geometry(lightbar))
        continue;

      lightbar.color = get_color(bgr_img, contour);
      lightbars.emplace_back(lightbar);
      lightbar_id += 1;
    }

    // 将灯条从左到右排序
    std::sort(lightbars.begin(), lightbars.end(),
              [](const Lightbar &a, const Lightbar &b) { return a.center.x < b.center.x; });

    // 获取装甲板
    std::list<Armor> armors;

    for (auto left = lightbars.begin(); left != lightbars.end(); left++)
    {
      for (auto right = std::next(left); right != lightbars.end(); right++)
      {
        if (left->color != right->color)
          continue;

        // 检查两个灯条之间是否有其他灯条
        // 预处理灯条位置信息
        // 在外层循环前创建灯条位置索引
        std::vector<std::pair<float, size_t>> x_positions;
        for (auto it = lightbars.begin(); it != lightbars.end(); ++it)
        {
          x_positions.emplace_back(it->center.x, it->id);
        }
        std::sort(x_positions.begin(), x_positions.end());

        // 然后在检查时只需查找位置索引
        bool has_lightbar_between = false;
        auto left_idx = std::find_if(x_positions.begin(), x_positions.end(),
                                      [&](const auto &p) { return p.second == left->id; });
        auto right_idx = std::find_if(x_positions.begin(), x_positions.end(),
                                       [&](const auto &p) { return p.second == right->id; });

        if (std::distance(left_idx, right_idx) > 1)
        {
          has_lightbar_between = true;
        }

        // 如果两个灯条之间有其他灯条，则不构成装甲板
        if (has_lightbar_between)
          continue;

        auto armor = Armor(*left, *right);
        if (!check_geometry(armor))
          continue;

        // 修改这里的调用以匹配新的函数签名
        armor.pattern = cv::Mat();  // 确保pattern已初始化
        get_pattern(bgr_img, armor, armor.pattern);

        classify(armor);
        if (!check_name(armor))
          continue;

        armors.emplace_back(armor);
      }
    }

    return armors;
  }

  bool Detector::check_geometry(const Lightbar &lightbar)
  {
    auto angle_ok = (lightbar.angle_error * 57.3) < 45; // degree
    auto ratio_ok = lightbar.ratio > 1.5 && lightbar.ratio < 20;
    auto length_ok = lightbar.length > 8;
    return angle_ok && ratio_ok && length_ok;
  }

  bool Detector::check_geometry(const Armor &armor)
  {
    auto ratio_ok = armor.ratio > 1 && armor.ratio < 5;
    auto side_ratio_ok = armor.side_ratio < 1.5;
    auto rectangular_error_ok = (armor.rectangular_error * 57.3) < 25;
    return ratio_ok && side_ratio_ok && rectangular_error_ok;
  }

  bool Detector::check_name(const Armor &armor)
  {
    auto name_ok = armor.name != ArmorName::not_armor;
    auto confidence_ok = armor.confidence > 0.8;

    return name_ok && confidence_ok;
  }

  // 修改后 - 用掩码和均值计算优化
  Color Detector::get_color(const cv::Mat &bgr_img, const std::vector<cv::Point> &contour)
  {
    cv::Mat mask = cv::Mat::zeros(bgr_img.size(), CV_8UC1);
    cv::drawContours(mask, std::vector<std::vector<cv::Point>>{contour}, 0, 255, cv::FILLED);

    cv::Scalar mean = cv::mean(bgr_img, mask);
    return mean[0] > mean[2] ? Color::blue : Color::red;
  }

  // 修改前：每次都创建新矩阵
  // cv::Mat Detector::get_pattern(const cv::Mat &bgr_img, const Armor &armor)
  // {
  //   // ... 计算 ROI ...
  //   return bgr_img(roi);
  // }

  // 修改后：避免不必要的复制，直接使用ROI
  void Detector::get_pattern(const cv::Mat &bgr_img, const Armor &armor, cv::Mat &pattern)
  {
    // 延长灯条获得装甲板角点
    // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
    auto tl = armor.left.center - armor.left.top2bottom * 1.125;
    auto bl = armor.left.center + armor.left.top2bottom * 1.125;
    auto tr = armor.right.center - armor.right.top2bottom * 1.125;
    auto br = armor.right.center + armor.right.top2bottom * 1.125;

    auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
    auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
    auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
    auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
    auto roi_tl = cv::Point(roi_left, roi_top);
    auto roi_br = cv::Point(roi_right, roi_bottom);
    auto roi = cv::Rect(roi_tl, roi_br);

    bgr_img(roi).copyTo(pattern);
  }

  void Detector::classify(Armor &armor)
  {
    // 直接使用已加载的网络 net
    cv::Mat gray;
    cv::cvtColor(armor.pattern, gray, cv::COLOR_BGR2GRAY);

    cv::Mat input = cv::Mat::zeros(32, 32, CV_8UC1);
    double x_scale = 32.0 / gray.cols;
    double y_scale = 32.0 / gray.rows;
    double scale = std::min(x_scale, y_scale);
    int w = static_cast<int>(gray.cols * scale);
    int h = static_cast<int>(gray.rows * scale);

    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(w, h));
    resized.copyTo(input(cv::Rect(0, 0, w, h)));

    auto blob = cv::dnn::blobFromImage(input, 1.0 / 255.0, cv::Size(), cv::Scalar());
    net_.setInput(blob);
    cv::Mat outputs = net_.forward();

    // softmax
    float max = *std::max_element(outputs.begin<float>(), outputs.end<float>());
    cv::exp(outputs - max, outputs);
    float sum = static_cast<float>(cv::sum(outputs)[0]);
    outputs /= sum;

    double confidence;
    cv::Point label_point;
    cv::minMaxLoc(outputs.reshape(1, 1), nullptr, &confidence, nullptr, &label_point);
    int label_id = label_point.x;

    armor.confidence = confidence;
    armor.name = static_cast<ArmorName>(label_id);
  }
} // namespace auto_aim
