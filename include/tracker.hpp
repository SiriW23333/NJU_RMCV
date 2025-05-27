#ifndef TRACKER_HPP
#define TRACKER_HPP

#include <opencv2/video/tracking.hpp>
#include <deque>

class ArmorPredictor {
public:
    ArmorPredictor();

    // 更新卡尔曼滤波器，输入为目标的三维位置（tvec: 3x1 CV_64F）
    void update(const cv::Mat& tvec);

    // 预测未来 t_future 秒后的位置
    cv::Point3f predict(float t_future);

    // 获取当前估计速度
    cv::Point3f getVelocity();

    // 获取历史位置
    const std::deque<cv::Point3f>& getHistory() const;

private:
    cv::KalmanFilter kf;
    cv::Mat measurement;
    std::deque<cv::Point3f> history;
    bool initialized;
    float dt;
    double last_tick;
};

#endif // TRACKER_HPP
