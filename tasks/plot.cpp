#include "plot.hpp"
#include "fmt/format.h"
#include <algorithm>

namespace auto_aim {



// TrajectoryBuffer 实现
void TrajectoryBuffer::addPoint(const std::string& armor_id, const cv::Point3f& point) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& traj = trajectories_[armor_id];
    traj.push_back(point);
    if (traj.size() > MAX_POINTS) {
        traj.pop_front();
    }
}

void TrajectoryBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    trajectories_.clear();
}

std::deque<cv::Point3f> TrajectoryBuffer::getTrajectory(const std::string& armor_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = trajectories_.find(armor_id);
    if (it != trajectories_.end()) {
        return it->second;
    }
    return std::deque<cv::Point3f>(); // 返回空轨迹
}

// TrajectoryVisualizer 实现
TrajectoryVisualizer::TrajectoryVisualizer(int width, int height)
    : width_(width), height_(height) {
}

void TrajectoryVisualizer::setBackgroundColor(const cv::Scalar& color) {
    bg_color_ = color;
}

void TrajectoryVisualizer::setGridColor(const cv::Scalar& color) {
    grid_color_ = color;
}

void TrajectoryVisualizer::setGridSize(int size) {
    grid_size_ = size;
}

void TrajectoryVisualizer::setPanelSize(int width, int height) {
    width_ = width;
    height_ = height;
}

void TrajectoryVisualizer::setScale(float scale) {
    scale_ = scale;
}

void TrajectoryVisualizer::drawGrid(cv::Mat& panel) const {
    // 绘制水平和垂直网格线
    for (int i = 0; i < panel.rows; i += grid_size_) {
        cv::line(panel, cv::Point(0, i), cv::Point(panel.cols, i), grid_color_, 1);
    }
    for (int i = 0; i < panel.cols; i += grid_size_) {
        cv::line(panel, cv::Point(i, 0), cv::Point(i, panel.rows), grid_color_, 1);
    }
}

void TrajectoryVisualizer::drawAxes(cv::Mat& panel) const {
    int center_x = panel.cols / 2;
    int center_y = panel.rows / 2;
    
    // 绘制坐标轴
    cv::line(panel, cv::Point(0, center_y), cv::Point(panel.cols, center_y), cv::Scalar(0, 0, 150), 2); // X轴
    cv::line(panel, cv::Point(center_x, 0), cv::Point(center_x, panel.rows), cv::Scalar(0, 150, 0), 2); // Y轴
    
    // 绘制标签
    cv::putText(panel, "X", cv::Point(panel.cols - 20, center_y - 10), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 150), 1);
    cv::putText(panel, "Y", cv::Point(center_x + 10, 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 150, 0), 1);

    // 添加标题
    cv::putText(panel, "Armor Trajectory (Top View)", 
                cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 
                0.7, cv::Scalar(255, 255, 255), 2);
}

void TrajectoryVisualizer::drawLegend(cv::Mat& panel) const {
    // 添加图例
    cv::rectangle(panel, cv::Rect(panel.cols - 140, panel.rows - 80, 130, 70), cv::Scalar(60, 60, 60), -1);
    cv::putText(panel, "Legend:", cv::Point(panel.cols - 130, panel.rows - 60), 
               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::line(panel, cv::Point(panel.cols - 130, panel.rows - 45), 
            cv::Point(panel.cols - 110, panel.rows - 45), cv::Scalar(255, 0, 0), 2);
    cv::putText(panel, "Blue Team", cv::Point(panel.cols - 105, panel.rows - 40), 
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::line(panel, cv::Point(panel.cols - 130, panel.rows - 30), 
            cv::Point(panel.cols - 110, panel.rows - 30), cv::Scalar(0, 0, 255), 2);
    cv::putText(panel, "Red Team", cv::Point(panel.cols - 105, panel.rows - 25), 
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    cv::line(panel, cv::Point(panel.cols - 130, panel.rows - 15), 
            cv::Point(panel.cols - 110, panel.rows - 15), cv::Scalar(0, 255, 255), 2);
    cv::putText(panel, "Prediction", cv::Point(panel.cols - 105, panel.rows - 10), 
               cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
}

cv::Mat TrajectoryVisualizer::createTrajectoryPanel(
    const TrajectoryBuffer& trajectories,
    const std::vector<LineInfo>& armors) {
    
    // 创建面板
    cv::Mat panel(height_, width_, CV_8UC3, bg_color_);
    
    // 绘制网格
    drawGrid(panel);
    
    // 绘制坐标轴
    drawAxes(panel);
    
    int center_x = panel.cols / 2;
    int center_y = panel.rows / 2;
    
    // 收集所有轨迹点，用于计算边界框
    std::vector<cv::Point3f> all_points;
    
    // 首先收集所有装甲板的所有轨迹点
    for (const auto& armor : armors) {
        std::string armor_id = fmt::format("{}_{}", armor.color, armor.name);
        auto trajectory = trajectories.getTrajectory(armor_id);
        all_points.insert(all_points.end(), trajectory.begin(), trajectory.end());
        
        // 如果有UKF数据，还需要加入预测点
        if (armor.has_ukf && !trajectory.empty()) {
            double x = armor.ukf_state(0); // xc
            double vx = armor.ukf_state(1); // vxc
            double ax = armor.ukf_state(2); // axc
            double z = armor.ukf_state(3); // zc
            double vz = armor.ukf_state(4); // vzc
            double az = armor.ukf_state(5); // azc
            
            const int predict_steps = 10;
            const float predict_dt = 0.1f;
            
            for (int i = 1; i <= predict_steps; ++i) {
                float t = predict_dt * i;
                double pred_x = x + vx * t + 0.5 * ax * t * t;
                double pred_z = z + vz * t + 0.5 * az * t * t;
                all_points.push_back(cv::Point3f(pred_x, 0, pred_z));
            }
        }
    }
    
    // 如果没有点，就使用默认缩放和偏移
    float auto_scale = scale_;
    cv::Point3f offset(0, 0, 0);
    
    // 计算边界框并自动调整缩放和中心
    if (!all_points.empty()) {
        // 找出X和Z的最大最小值
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();
        
        for (const auto& point : all_points) {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_z = std::min(min_z, point.z);
            max_z = std::max(max_z, point.z);
        }
        
        // 计算轨迹中心
        float center_point_x = (min_x + max_x) / 2.0f;
        float center_point_z = (min_z + max_z) / 2.0f;
        
        // 设置偏移，使轨迹中心与面板中心对齐
        offset = cv::Point3f(-center_point_x, 0, -center_point_z);
        
        // 计算合适的缩放因子，使轨迹占据面板80%的空间
        float width_scale = (width_ * 0.8f) / (max_x - min_x + 0.001f); // 避免除零
        float height_scale = (height_ * 0.8f) / (max_z - min_z + 0.001f);
        auto_scale = std::min(width_scale, height_scale);
        
        // 限制最小和最大缩放
        auto_scale = std::max(auto_scale, 10.0f);   // 最小缩放
        auto_scale = std::min(auto_scale, 500.0f);  // 最大缩放
    }
    
    // 为每个装甲板绘制轨迹，使用新的缩放和偏移
    for (const auto& armor : armors) {
        // 生成唯一ID
        std::string armor_id = fmt::format("{}_{}", armor.color, armor.name);
        
        // 获取轨迹
        auto trajectory = trajectories.getTrajectory(armor_id);
        if (trajectory.empty()) {
            continue;
        }
        
        // 根据装甲板颜色设置轨迹颜色
        cv::Scalar traj_color;
        if (armor.color == "BLUE") {
            traj_color = cv::Scalar(255, 0, 0);  // 蓝色
        } else if (armor.color == "RED") {
            traj_color = cv::Scalar(0, 0, 255);  // 红色
        } else {
            traj_color = cv::Scalar(0, 255, 0);  // 绿色
        }
        
        // 绘制轨迹线，应用新的缩放和偏移
        std::vector<cv::Point> trajectory_pixels;
        for (const auto& point : trajectory) {
            // 应用偏移并转换世界坐标到像素坐标
            int px = center_x + static_cast<int>((point.x + offset.x) * auto_scale);
            int py = center_y - static_cast<int>((point.z + offset.z) * auto_scale);
            
            trajectory_pixels.push_back(cv::Point(px, py));
        }
        
        // 绘制轨迹线段，线条粗细随时间变化
        for (size_t i = 1; i < trajectory_pixels.size(); ++i) {
            // 越新的点线条越粗
            int thickness = 1 + static_cast<int>((i * 3) / trajectory_pixels.size());
            cv::line(panel, trajectory_pixels[i-1], trajectory_pixels[i], traj_color, thickness);
        }
        
        // 显示当前位置（最新点）
        if (!trajectory_pixels.empty()) {
            cv::circle(panel, trajectory_pixels.back(), 5, traj_color, -1);
            
            // 在当前位置显示装甲板ID
            cv::putText(panel, armor_id, 
                       trajectory_pixels.back() + cv::Point(10, -10), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
        
        // 如果有UKF状态，绘制预测轨迹，也应用新的缩放和偏移
        if (armor.has_ukf) {
            // 显示速度矢量
            double vx = armor.ukf_state(1); // vxc
            double vz = armor.ukf_state(4); // vzc
            
            // 计算速度矢量终点，也应用新缩放
            cv::Point vel_end = trajectory_pixels.back() + 
                               cv::Point(static_cast<int>(vx * auto_scale * 0.5), 
                                        static_cast<int>(-vz * auto_scale * 0.5));
            
            // 绘制速度矢量
            cv::arrowedLine(panel, trajectory_pixels.back(), vel_end, 
                           cv::Scalar(0, 255, 255), 2);
            
            // 预测未来几个点的位置
            const int predict_steps = 10;
            const float predict_dt = 0.1f; // 每步预测时间间隔
            
            std::vector<cv::Point> predict_points;
            predict_points.push_back(trajectory_pixels.back());
            
            // 简单预测：根据当前位置、速度和加速度
            double x = armor.ukf_state(0); // xc
            double ax = armor.ukf_state(2); // axc
            double z = armor.ukf_state(3); // zc
            double az = armor.ukf_state(5); // azc
            
            for (int i = 1; i <= predict_steps; ++i) {
                float t = predict_dt * i;
                double pred_x = x + vx * t + 0.5 * ax * t * t;
                double pred_z = z + vz * t + 0.5 * az * t * t;
                
                // 转换到像素坐标，应用偏移和新缩放
                int px = center_x + static_cast<int>((pred_x + offset.x) * auto_scale);
                int py = center_y - static_cast<int>((pred_z + offset.z) * auto_scale);
                
                predict_points.push_back(cv::Point(px, py));
            }
            
            // 绘制预测轨迹
            for (size_t i = 1; i < predict_points.size(); ++i) {
                cv::line(panel, predict_points[i-1], predict_points[i], 
                        cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
            }
        }
    }
    
    // 显示当前缩放值（方便调试）
    cv::putText(panel, fmt::format("Scale: {:.1f}", auto_scale),
              cv::Point(10, height_ - 10), cv::FONT_HERSHEY_SIMPLEX,
              0.5, cv::Scalar(150, 150, 150), 1);
    
    // 绘制图例
    drawLegend(panel);
    
    return panel;
}

} // namespace auto_aim