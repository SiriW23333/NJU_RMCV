#include "detector.hpp"
#include "PnP.hpp"
#include "ekf_singer.hpp" // 注意包含正确的头文件
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <deque>

#define VIDEO "/home/wxy/NJU_RMCV/test/video/linear_video/linear_difficult.avi"

using namespace auto_aim;

// 相机内参
static const cv::Mat camera_matrix =
    (cv::Mat_<double>(3, 3) << 1286.307063384126, 0, 645.34450819155256,
                               0, 1288.1400736562441, 483.6163720308021,
                               0, 0, 1);
// 畸变系数
static const cv::Mat distort_coeffs =
    (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);

int main(int argc, char *argv[])
{
    Detector detector;
    PNPSolver pnp_solver(camera_matrix, distort_coeffs);

    cv::VideoCapture cap(VIDEO);
    if (!cap.isOpened()) {
        std::cerr << "错误：无法打开视频文件！" << std::endl;
        return -1;
    }

    std::map<std::string, EKF_Singer> ekf_map;
    std::map<std::string, cv::Point3f> last_predict_pos;
    int frame_id = 0;

    // 打开CSV文件
    std::ofstream csv_file("/home/wxy/NJU_RMCV/prediction_error.csv");
    csv_file << "res_x,res_y,res_z" << std::endl;

    double fps = 60.0;
    double dt = 1.0 / fps;

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        frame_id++;

        auto armors = detector.detect(frame);

        for (const auto& armor : armors) {
            std::vector<cv::Point2f> img_points{
                armor.left.top,
                armor.right.top,
                armor.right.bottom,
                armor.left.bottom
            };
            cv::Mat rvec, tvec;
            double yaw_val;
            pnp_solver.solvePose(img_points, rvec, tvec, yaw_val);

            std::string armor_id = std::to_string(armor.color) + "_" + std::to_string(armor.name);
            if (ekf_map.find(armor_id) == ekf_map.end()) {
                EKF_Singer ekf;
                ekf.init(dt);
                Eigen::VectorXd x0 = Eigen::VectorXd::Zero(9);
                x0(0) = tvec.at<double>(0); // x
                x0(3) = tvec.at<double>(1); // y
                x0(6) = tvec.at<double>(2); // z
                ekf.setState(x0);
                ekf_map[armor_id] = ekf;
            }
            auto& ekf = ekf_map[armor_id];

            // 构造观测向量 (x, y, z)
            Eigen::VectorXd z(3);
            z << tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2);

            ekf.predict();
            ekf.update(z);

            // 预测下一帧位置
            Eigen::VectorXd x_pred = ekf.Onlypredict();
            cv::Point3f next_predict_pos(x_pred(0), x_pred(3), x_pred(6));

            // 误差分析
            cv::Point3f actual_pos(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
            if (last_predict_pos.find(armor_id) != last_predict_pos.end() && frame_id > 1) {
                cv::Point3f last_pred = last_predict_pos[armor_id];
                double res_x = actual_pos.x - last_pred.x;
                double res_y = actual_pos.y - last_pred.y;
                double res_z = actual_pos.z - last_pred.z;
                csv_file << std::fixed << std::setprecision(6)
                         << res_x << "," << res_y << "," << res_z << std::endl;
            }
            last_predict_pos[armor_id] = next_predict_pos;
        }

        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) break;
    }

    csv_file.close();
    cap.release();
    cv::destroyAllWindows();
    std::cout << "误差分析数据已保存到: /home/wxy/NJU_RMCV/prediction_error.csv" << std::endl;
    return 0;
}
