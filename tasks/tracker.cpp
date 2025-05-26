#include "tracker.hpp"

// 添加在 #include 部分
#include <opencv2/video/tracking.hpp>
#include <deque>

ArmorPredictor::ArmorPredictor() {
    // 初始化卡尔曼滤波器 (状态向量: x,y,z,vx,vy,vz)
    kf.init(6, 3, 0);
    
    // 状态转移矩阵 (F)
    // [1 0 0 dt 0  0 ]
    // [0 1 0 0  dt 0 ]
    // [0 0 1 0  0  dt]
    // [0 0 0 1  0  0 ]
    // [0 0 0 0  1  0 ]
    // [0 0 0 0  0  1 ]
    cv::setIdentity(kf.transitionMatrix);
    
    // 测量矩阵 (H)
    // [1 0 0 0 0 0]
    // [0 1 0 0 0 0]
    // [0 0 1 0 0 0]
    kf.measurementMatrix = cv::Mat::zeros(3, 6, CV_32F);
    kf.measurementMatrix.at<float>(0, 0) = 1.0f;
    kf.measurementMatrix.at<float>(1, 1) = 1.0f;
    kf.measurementMatrix.at<float>(2, 2) = 1.0f;
    
    // 过程噪声协方差矩阵 (Q)
    cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-5));
    
    // 测量噪声协方差矩阵 (R)
    cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2));
    
    // 后验错误协方差矩阵 (P)
    cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1));
    
    measurement = cv::Mat::zeros(3, 1, CV_32F);
    initialized = false;
    last_tick = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
    dt = 0.01f;  // 初始化时间步长
}

void ArmorPredictor::update(const cv::Mat& tvec) {
    // 计算时间步长
    double current_tick = static_cast<double>(cv::getTickCount()) / cv::getTickFrequency();
    dt = static_cast<float>(current_tick - last_tick);
    last_tick = current_tick;
    
    // 更新状态转移矩阵中的时间步长
    kf.transitionMatrix.at<float>(0, 3) = dt;
    kf.transitionMatrix.at<float>(1, 4) = dt;
    kf.transitionMatrix.at<float>(2, 5) = dt;
    
    // 转换测量值为 CV_32F 类型
    cv::Mat meas = cv::Mat::zeros(3, 1, CV_32F);
    if (tvec.type() == CV_64F) {
        tvec.convertTo(meas, CV_32F);
    } else {
        meas = tvec.clone();
    }
    
    if (!initialized) {
        // 第一次更新，初始化状态
        kf.statePost.at<float>(0) = meas.at<float>(0);
        kf.statePost.at<float>(1) = meas.at<float>(1);
        kf.statePost.at<float>(2) = meas.at<float>(2);
        kf.statePost.at<float>(3) = 0;
        kf.statePost.at<float>(4) = 0;
        kf.statePost.at<float>(5) = 0;
        initialized = true;
    } else {
        // 预测
        cv::Mat prediction = kf.predict();
        
        // 更新
        measurement.at<float>(0) = meas.at<float>(0);
        measurement.at<float>(1) = meas.at<float>(1);
        measurement.at<float>(2) = meas.at<float>(2);
        kf.correct(measurement);
    }
    
    // 添加到历史记录
    cv::Point3f current_pos(
        kf.statePost.at<float>(0),
        kf.statePost.at<float>(1),
        kf.statePost.at<float>(2)
    );
    history.push_back(current_pos);
    
    // 限制历史记录大小
    if (history.size() > 50) {
        history.pop_front();
    }
}

cv::Point3f ArmorPredictor::predict(float t_future) {
    if (!initialized) {
        return cv::Point3f(0, 0, 0);
    }
    
    // 预测未来位置
    cv::Mat prediction = kf.predict();
    cv::Mat future_state = kf.statePost.clone();
    
    // 计算未来位置: x' = x + vx * t_future
    future_state.at<float>(0) += future_state.at<float>(3) * t_future;
    future_state.at<float>(1) += future_state.at<float>(4) * t_future;
    future_state.at<float>(2) += future_state.at<float>(5) * t_future;
    
    return cv::Point3f(
        future_state.at<float>(0),
        future_state.at<float>(1),
        future_state.at<float>(2)
    );
}

cv::Point3f ArmorPredictor::getVelocity() {
    if (!initialized) {
        return cv::Point3f(0, 0, 0);
    }
    
    return cv::Point3f(
        kf.statePost.at<float>(3),
        kf.statePost.at<float>(4),
        kf.statePost.at<float>(5)
    );
}

const std::deque<cv::Point3f>& ArmorPredictor::getHistory() const {
    return history;
}