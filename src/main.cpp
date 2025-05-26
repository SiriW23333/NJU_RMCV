#include "tasks/detector.hpp"
#include "tools/img_tools.hpp"
#include "fmt/core.h"
#include "fmt/format.h"
#include <opencv2/opencv.hpp>
#include "tasks/armor.hpp"  // 确保该头文件包含了装甲板颜色和名称的定义
#include "tasks/tracker.hpp" // 确保该头文件包含了ArmorPredictor类的定义

using namespace auto_aim;

// clang-format off
//  相机内参
static const cv::Mat camera_matrix =
    (cv::Mat_<double>(3, 3) <<  1286.307063384126 , 0                  , 645.34450819155256, 
                                0                 , 1288.1400736562441 , 483.6163720308021 , 
                                0                 , 0                  , 1                   );
// 畸变系数
static const cv::Mat distort_coeffs =
    (cv::Mat_<double>(1, 5) << -0.47562935060124745, 0.21831745829617311, 0.0004957613589406044, -0.00034617769548693592, 0);
// clang-format on

static const double LIGHTBAR_LENGTH = 0.056; // 灯条长度    单位：米
static const double ARMOR_WIDTH = 0.135;     // 装甲板宽度  单位：米

// 装甲板坐标系下4个点的坐标
static const std::vector<cv::Point3f> object_points {
    { -ARMOR_WIDTH/2 , -LIGHTBAR_LENGTH/2 , 0 },  // 点 1
    { ARMOR_WIDTH/2 , -LIGHTBAR_LENGTH/2, 0 },  // 点 2
    { ARMOR_WIDTH/2 , LIGHTBAR_LENGTH/2, 0 },  // 点 3
    { -ARMOR_WIDTH/2 , LIGHTBAR_LENGTH/2, 0 }   // 点 4（修正了原代码中的错误）
};

int main(int argc, char *argv[])
{
    auto_aim::Detector detector;

    // 您可以根据需要选择视频源
    cv::VideoCapture cap("/home/wxy/RM/CLASS_5/Lesson_5/go_straight_and spin.mp4");
    // 如果需要使用另一个视频源，请取消下面这行的注释并注释上面的行
    // cv::VideoCapture cap("/home/wxy/RM/CLASS_4/CLASS_4/imgs/8radps.avi");
    
    cv::Mat img;

    // 添加装甲板预测器
    ArmorPredictor predictor;

    while (true)
    {
        cap >> img;
        if (img.empty()) // 读取失败 或 视频结尾
            break;

        auto armors = detector.detect(img);

        // 创建副本用于绘制，保持原始图像不变
        cv::Mat draw_img = img.clone();

        // 遍历所有识别到的装甲板
        for (const auto& armor : armors)
        {
            // 绘制装甲板点
            tools::draw_points(draw_img, armor.points);
            
            // 显示装甲板颜色、名称和置信度（来自CLASS_4的功能）
            tools::draw_text(
                draw_img,
                fmt::format("{},{},{:.2f}", COLORS[armor.color], ARMOR_NAMES[armor.name], armor.confidence),
                armor.left.top
            );

            // PnP姿态解算（来自CLASS_5的功能）
            std::vector<cv::Point2f> img_points{
                armor.left.top,
                armor.right.top,
                armor.right.bottom,
                armor.left.bottom
            };

            // 解算装甲板位姿
            cv::Mat rvec, tvec;
            cv::solvePnP(object_points, img_points, camera_matrix, distort_coeffs, rvec, tvec);

            // 显示位置向量和旋转向量
            tools::draw_text(draw_img, 
                fmt::format("tvec: x{: .2f} y{: .2f} z{: .2f}", tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2)), 
                cv::Point2f(10, 60), 1.7, cv::Scalar(0, 255, 255), 3);
            tools::draw_text(draw_img, 
                fmt::format("rvec: x{: .2f} y{: .2f} z{: .2f}", rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2)), 
                cv::Point2f(10, 120), 1.7, cv::Scalar(0, 255, 255), 3);

            // 转换旋转向量为欧拉角
            cv::Mat rmat;
            cv::Rodrigues(rvec, rmat);
            double yaw = atan2(rmat.at<double>(1, 0), rmat.at<double>(0, 0));
            double pitch = atan2(-rmat.at<double>(2, 0), 
                sqrt(rmat.at<double>(2, 1) * rmat.at<double>(2, 1) + rmat.at<double>(2, 2) * rmat.at<double>(2, 2)));
            double roll = atan2(rmat.at<double>(2, 1), rmat.at<double>(2, 2));

            // 显示欧拉角
            tools::draw_text(draw_img, 
                fmt::format("yaw{: .2f} pitch{: .2f} roll{: .2f}", yaw, pitch, roll), 
                cv::Point2f(10, 180), 1.7, cv::Scalar(0, 255, 255), 3);

            // 更新预测器
            predictor.update(tvec);
            
            // 获取估计的速度
            cv::Point3f velocity = predictor.getVelocity();
            
            // 预测0.1秒后的位置
            float predict_time = 0.1f; // 单位：秒
            cv::Point3f future_pos = predictor.predict(predict_time);
            
            // 显示预测信息
            tools::draw_text(draw_img, 
                fmt::format("vel: x{: .2f} y{: .2f} z{: .2f}", velocity.x, velocity.y, velocity.z), 
                cv::Point2f(10, 240), 1.7, cv::Scalar(0, 255, 0), 3);
                
            tools::draw_text(draw_img, 
                fmt::format("pred({:.1f}s): x{: .2f} y{: .2f} z{: .2f}", 
                            predict_time, future_pos.x, future_pos.y, future_pos.z), 
                cv::Point2f(10, 300), 1.7, cv::Scalar(0, 255, 0), 3);
            
            // 可视化历史轨迹
            const auto& history = predictor.getHistory();
            for (size_t i = 1; i < history.size(); ++i) {
                cv::Point3f p1 = history[i-1];
                cv::Point3f p2 = history[i];
                
                // 将3D点投影到图像平面
                std::vector<cv::Point3f> pts3d = {p1, p2};
                std::vector<cv::Point2f> pts2d;
                cv::projectPoints(pts3d, rvec, tvec, camera_matrix, distort_coeffs, pts2d);
                
                // 绘制轨迹线
                cv::line(draw_img, pts2d[0], pts2d[1], cv::Scalar(0, 255, 0), 2);
            }
            
            // 如果只处理第一个装甲板
            break;
        }

        
        // 显示前缩小图像
        cv::Mat small_img;
        float scale_factor = 0.2; // 缩放比例，可以根据需要调整(0.5-0.8之间比较合适)
        cv::resize(draw_img, small_img, cv::Size(), scale_factor, scale_factor);
        
        cv::imshow("Armor Detection & PnP (press q to quit)", small_img);

        if (cv::waitKey(20) == 'q')
            break;
    }

    cv::destroyAllWindows();
    return 0;
}