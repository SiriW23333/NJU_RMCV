#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <deque>
#include <mutex>
#include <vector>
#include <string>
#include "ukf.hpp"

namespace auto_aim {

// 轨迹信息结构体定义
struct LineInfo {
    std::string color;
    std::string name;
    float confidence;
    cv::Point2f position;
    cv::Mat rvec;
    cv::Mat tvec;
    double yaw, pitch, roll;
    cv::Scalar color_value;
    
    // UKF相关数据
    Eigen::Matrix<double, UKF::n_x, 1> ukf_state;
    bool has_ukf = false;
};

// 轨迹缓存类
class TrajectoryBuffer {
public:
    // 添加点到轨迹
    void addPoint(const std::string& armor_id, const cv::Point3f& point);
    
    // 清除所有轨迹
    void clear();
    
    // 获取特定装甲板的轨迹
    std::deque<cv::Point3f> getTrajectory(const std::string& armor_id) const;

private:
    static const size_t MAX_POINTS = 100; // 最多保存100个历史点
    std::map<std::string, std::deque<cv::Point3f>> trajectories_; // 每个装甲板ID对应一条轨迹
    mutable std::mutex mutex_;
};

// 轨迹可视化类
class TrajectoryVisualizer {
public:
    // 构造函数
    TrajectoryVisualizer(int width = 500, int height = 500);
    
    // 创建并返回轨迹可视化面板
    cv::Mat createTrajectoryPanel(const TrajectoryBuffer& trajectories,
                                 const std::vector<LineInfo>& armors);
    
    // 自定义设置
    void setBackgroundColor(const cv::Scalar& color);
    void setGridColor(const cv::Scalar& color);
    void setGridSize(int size);
    void setPanelSize(int width, int height);
    void setScale(float scale);

private:
    // 面板尺寸
    int width_;
    int height_;
    
    // 显示设置
    cv::Scalar bg_color_ = cv::Scalar(0, 0, 0);
    cv::Scalar grid_color_ = cv::Scalar(30, 30, 30);
    int grid_size_ = 50;
    float scale_ = 100.0f;
    
    // 绘制网格线
    void drawGrid(cv::Mat& panel) const;
    
    // 绘制坐标轴
    void drawAxes(cv::Mat& panel) const;
    
    // 绘制图例
    void drawLegend(cv::Mat& panel) const;
};

} // namespace auto_aim