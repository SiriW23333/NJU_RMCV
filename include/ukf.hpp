#pragma once
#include <Eigen/Dense>
#include <vector>

class UKF {
public:
    static constexpr int n_x = 12; // 状态维度
    static constexpr int n_z = 5;  // 观测维度

    using VectorX = Eigen::Matrix<double, n_x, 1>;
    using MatrixX = Eigen::Matrix<double, n_x, n_x>;
    using VectorZ = Eigen::Matrix<double, n_z, 1>;
    using MatrixZ = Eigen::Matrix<double, n_z, n_z>;

    UKF();

    void init(const VectorX& x0, const MatrixX& P0);
    void predict(double dt);
    void update(const VectorZ& z);

    VectorX getState() const { return x_; }
    MatrixX getCovariance() const { return P_; }

private:
    // 状态转移函数
    VectorX f(const VectorX& x, double dt) const;
    // 观测函数
    VectorZ h(const VectorX& x) const;

    // sigma点生成与传播
    void generateSigmaPoints(const VectorX& x, const MatrixX& P, std::vector<VectorX>& Xsig) const;
    void predictSigmaPoints(const std::vector<VectorX>& Xsig_in, double dt, std::vector<VectorX>& Xsig_out) const;
    void predictMeasurementSigmaPoints(const std::vector<VectorX>& Xsig, std::vector<VectorZ>& Zsig) const;

    // UKF参数
    double alpha_ = 1e-3;
    double beta_ = 2.0;
    double kappa_ = 0.0;
    double lambda_;
    Eigen::Matrix<double, 2 * n_x + 1, 1> weights_m_;
    Eigen::Matrix<double, 2 * n_x + 1, 1> weights_c_;

    // 状态
    VectorX x_;
    MatrixX P_;

    // 噪声
    MatrixX Q_; // 过程噪声
    MatrixZ R_; // 观测噪声
};