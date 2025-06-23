#pragma once

#include <Eigen/Dense>

namespace auto_aim
{
    class EKF_Singer
    {
    public:
        EKF_Singer();
        void init(double dt);
        void setState(const Eigen::VectorXd &x0);
        void reset(const Eigen::VectorXd &x0);
        

        Eigen::VectorXd predict();
        Eigen::VectorXd Onlypredict();
        Eigen::VectorXd update(const Eigen::VectorXd &z);

    private:
        Eigen::VectorXd f(const Eigen::VectorXd &x);
        Eigen::MatrixXd computeFJacobian(const Eigen::VectorXd &x);
        Eigen::VectorXd h(const Eigen::VectorXd &x);
        Eigen::MatrixXd computeHJacobian(const Eigen::VectorXd &x);

        double dt_;
        double alpha_;
        Eigen::VectorXd x_;
        Eigen::MatrixXd P_;
        Eigen::MatrixXd Q_;
        Eigen::MatrixXd R_;
    };
}
