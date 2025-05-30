#include "ukf.hpp"
#include <cmath>

UKF::UKF() {
    lambda_ = alpha_ * alpha_ * (n_x + kappa_) - n_x;
    weights_m_.setZero();
    weights_c_.setZero();
    weights_m_(0) = lambda_ / (lambda_ + n_x);
    weights_c_(0) = weights_m_(0) + (1 - alpha_ * alpha_ + beta_);
    for (int i = 1; i < 2 * n_x + 1; ++i) {
        weights_m_(i) = 1.0 / (2 * (lambda_ + n_x));
        weights_c_(i) = weights_m_(i);
    }
    Q_ = 1e-2 * MatrixX::Identity();
    R_ = 1e-1 * MatrixZ::Identity();
}

void UKF::init(const VectorX& x0, const MatrixX& P0) {
    x_ = x0;
    P_ = P0;
}

UKF::VectorX UKF::f(const VectorX& x, double dt) const {
    VectorX x_pred = x;
    // 位置更新
    x_pred(0) += x(1) * dt + 0.5 * x(2) * dt * dt; // xc
    x_pred(3) += x(4) * dt + 0.5 * x(5) * dt * dt; // zc
    x_pred(6) += x(7) * dt + 0.5 * x(8) * dt * dt; // yc
    // 速度更新
    x_pred(1) += x(2) * dt; // vxc
    x_pred(4) += x(5) * dt; // vzc
    x_pred(7) += x(8) * dt; // vyc
    // 加速度不变
    // 偏航角和角速度
    x_pred(9) += x(10) * dt; // yaw
    // 角速度、半径不变
    return x_pred;
}

UKF::VectorZ UKF::h(const VectorX& x) const {
    VectorZ z_pred;
    // 观测模型：通过旋转中心、半径、偏航角计算装甲板实际位置
    double xc = x(0), zc = x(3), yc = x(6);
    double yaw = x(9), r = x(11);
    // 以xz平面为旋转平面，装甲板实际位置
    z_pred(0) = xc + r * std::cos(yaw); // xa
    z_pred(1) = zc + r * std::sin(yaw); // za
    z_pred(2) = yc;                     // ya
    z_pred(3) = yaw;                    // yaw
    z_pred(4) = r;                      // r
    return z_pred;
}

void UKF::generateSigmaPoints(const VectorX& x, const MatrixX& P, std::vector<VectorX>& Xsig) const {
    Eigen::Matrix<double, n_x, n_x> A = P.llt().matrixL();
    Xsig.resize(2 * n_x + 1);
    Xsig[0] = x;
    double scale = std::sqrt(lambda_ + n_x);
    for (int i = 0; i < n_x; ++i) {
        Xsig[i + 1]        = x + scale * A.col(i);
        Xsig[i + 1 + n_x]  = x - scale * A.col(i);
    }
}

void UKF::predictSigmaPoints(const std::vector<VectorX>& Xsig_in, double dt, std::vector<VectorX>& Xsig_out) const {
    Xsig_out.resize(Xsig_in.size());
    for (size_t i = 0; i < Xsig_in.size(); ++i) {
        Xsig_out[i] = f(Xsig_in[i], dt);
    }
}

void UKF::predictMeasurementSigmaPoints(const std::vector<VectorX>& Xsig, std::vector<VectorZ>& Zsig) const {
    Zsig.resize(Xsig.size());
    for (size_t i = 0; i < Xsig.size(); ++i) {
        Zsig[i] = h(Xsig[i]);
    }
}

void UKF::predict(double dt) {
    // 1. 生成sigma点
    std::vector<VectorX> Xsig;
    generateSigmaPoints(x_, P_, Xsig);

    // 2. sigma点传播
    std::vector<VectorX> Xsig_pred;
    predictSigmaPoints(Xsig, dt, Xsig_pred);

    // 3. 预测均值和协方差
    x_.setZero();
    for (size_t i = 0; i < Xsig_pred.size(); ++i)
        x_ += weights_m_(i) * Xsig_pred[i];

    P_.setZero();
    for (size_t i = 0; i < Xsig_pred.size(); ++i) {
        Eigen::Matrix<double, n_x, 1> dx = Xsig_pred[i] - x_;
        P_ += weights_c_(i) * dx * dx.transpose();
    }
    P_ += Q_;
}

void UKF::update(const VectorZ& z) {
    // 1. 生成sigma点
    std::vector<VectorX> Xsig;
    generateSigmaPoints(x_, P_, Xsig);

    // 2. sigma点通过观测模型
    std::vector<VectorZ> Zsig;
    predictMeasurementSigmaPoints(Xsig, Zsig);

    // 3. 预测观测均值
    VectorZ z_pred = VectorZ::Zero();
    for (size_t i = 0; i < Zsig.size(); ++i)
        z_pred += weights_m_(i) * Zsig[i];

    // 4. 观测协方差和状态-观测协方差
    MatrixZ S = MatrixZ::Zero();
    Eigen::Matrix<double, n_x, n_z> Tc = Eigen::Matrix<double, n_x, n_z>::Zero();
    for (size_t i = 0; i < Zsig.size(); ++i) {
        Eigen::Matrix<double, n_z, 1> dz = Zsig[i] - z_pred;
        Eigen::Matrix<double, n_x, 1> dx = Xsig[i] - x_;
        S += weights_c_(i) * dz * dz.transpose();
        Tc += weights_c_(i) * dx * dz.transpose();
    }
    S += R_;

    // 5. 卡尔曼增益
    Eigen::Matrix<double, n_x, n_z> K = Tc * S.inverse();

    // 6. 更新
    x_ += K * (z - z_pred);
    P_ -= K * S * K.transpose();
}