#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>
#include <random>

struct ReprojectionError {
    ReprojectionError(double observed_x, double observed_y, 
                     double point_x, double point_y, double point_z,
                     double fx, double fy, double cx, double cy)
        : observed_x_(observed_x), observed_y_(observed_y),
          point_x_(point_x), point_y_(point_y), point_z_(point_z),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(const T* const axis_angle, const T* const translation, T* residuals) const {
        T point[3] = {T(point_x_), T(point_y_), T(point_z_)};
        T transformed_point[3];
        ceres::AngleAxisRotatePoint(axis_angle, point, transformed_point);
        transformed_point[0] += translation[0];
        transformed_point[1] += translation[1];
        transformed_point[2] += translation[2];
        T predicted_x = (transformed_point[0] * T(fx_)) / transformed_point[2] + T(cx_);
        T predicted_y = (transformed_point[1] * T(fy_)) / transformed_point[2] + T(cy_);
        residuals[0] = predicted_x - T(observed_x_);
        residuals[1] = predicted_y - T(observed_y_);
        return true;
    }

    private:
        const double observed_x_, observed_y_;
        const double point_x_, point_y_, point_z_;
        const double fx_, fy_, cx_, cy_;
};

Eigen::Matrix4d ReadGroundTruthPose(const std::string& gt_path) {
    std::ifstream file(gt_path);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    if (file.is_open()) {
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) file >> T(i, j);
        file.close();
    }
    return T;
}

Eigen::Vector3d RotationMatrixToAxisAngle(const Eigen::Matrix3d& R) {
    Eigen::AngleAxisd angle_axis(R);
    return angle_axis.angle() * angle_axis.axis();
}

Eigen::Matrix3d AxisAngleToRotationMatrix(const Eigen::Vector3d& axis_angle) {
    double angle = axis_angle.norm();
    if (angle < 1e-10) return Eigen::Matrix3d::Identity();
    Eigen::Vector3d axis = axis_angle.normalized();
    Eigen::AngleAxisd angle_axis(angle, axis);
    return angle_axis.toRotationMatrix();
}

void SolveExtrinsicEstimation(const std::string& image_path, const std::string& gt_path,
                             double fx, double fy, double cx, double cy) {
    std::random_device rd;
    std::mt19937 gen(rd());
    const double rotation_noise_stddev = 10, translation_noise_stddev = 10;
    Eigen::Matrix4d T_ground_truth = ReadGroundTruthPose(gt_path);
    Eigen::Vector3d translation_gt = T_ground_truth.block<3, 1>(0, 3);
    Eigen::Vector3d axis_angle_gt = RotationMatrixToAxisAngle(T_ground_truth.block<3, 3>(0, 0));
    Eigen::Vector3d noisy_axis_angle = axis_angle_gt + Eigen::Vector3d::Random() * rotation_noise_stddev;
    Eigen::Vector3d noisy_translation = translation_gt + Eigen::Vector3d::Random() * translation_noise_stddev;
    double axis_angle[3] = {noisy_axis_angle.x(), noisy_axis_angle.y(), noisy_axis_angle.z()};
    double translation[3] = {noisy_translation.x(), noisy_translation.y(), noisy_translation.z()};
    ceres::Problem problem;
    std::ifstream file(image_path);
    if (!file.is_open()) return;
    double px, py, X, Y, Z, idx;
    while (file >> px >> py >> X >> Y >> Z >> idx) {
        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                new ReprojectionError(px, py, X, Y, Z, fx, fy, cx, cy));
        problem.AddResidualBlock(cost_function, nullptr, axis_angle, translation);
    }
    file.close();

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    Eigen::Matrix4d T_estimated = Eigen::Matrix4d::Identity();
    T_estimated.block<3, 3>(0, 0) = AxisAngleToRotationMatrix(Eigen::Vector3d(axis_angle[0], axis_angle[1], axis_angle[2]));
    T_estimated.block<3, 1>(0, 3) = Eigen::Map<Eigen::Vector3d>(translation);
    
    std::cout << "Estimated:\n" << T_estimated << "\nGround truth:\n" << T_ground_truth << "\n";

    std::cout << summary.BriefReport() << std::endl;

}

int main() {
    const double fx = 721.5, fy = 721.5, cx = 256, cy = 176;
    std::string source_dir = __FILE__;
    source_dir = source_dir.substr(0, source_dir.find_last_of("/\\")) + "/data";
    for (int i = 0; i < 5; i++) {
        std::string image_path = source_dir + "/Correspondences/pose_" + std::to_string(i) + ".txt";
        std::string gt_path = source_dir + "/Extrinsics/pose" + std::to_string(i) + ".txt";
        SolveExtrinsicEstimation(image_path, gt_path, fx, fy, cx, cy);
    }
    return 0;
}
