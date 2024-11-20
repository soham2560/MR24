#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <random>

class ExtrinsicCalibration {
private:
    struct ReprojectionError {
        ReprojectionError(const Eigen::Vector2d& observed, 
                          const Eigen::Vector3d& point,
                          const Eigen::Vector4d& camera_params)
            : observed_(observed), point_(point), 
              fx_(camera_params[0]), fy_(camera_params[1]), 
              cx_(camera_params[2]), cy_(camera_params[3]) {}

        template <typename T>
        bool operator()(const T* const axis_angle, const T* const translation, T* residuals) const {
            T point[3] = {T(point_[0]), T(point_[1]), T(point_[2])};
            T transformed_point[3];
            
            ceres::AngleAxisRotatePoint(axis_angle, point, transformed_point);
            transformed_point[0] += translation[0];
            transformed_point[1] += translation[1];
            transformed_point[2] += translation[2];
            
            T predicted_x = (transformed_point[0] * T(fx_)) / transformed_point[2] + T(cx_);
            T predicted_y = (transformed_point[1] * T(fy_)) / transformed_point[2] + T(cy_);
            
            residuals[0] = predicted_x - T(observed_[0]);
            residuals[1] = predicted_y - T(observed_[1]);
            
            return true;
        }

    private:
        const Eigen::Vector2d observed_;
        const Eigen::Vector3d point_;
        const double fx_, fy_, cx_, cy_;
    };

    static Eigen::Matrix4d readGroundTruthPose(const std::string& path) {
        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
        std::ifstream file(path);
        if (file.is_open()) {
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j) 
                    file >> T(i, j);
        }
        return T;
    }

    static Eigen::Vector3d rotationMatrixToAxisAngle(const Eigen::Matrix3d& R) {
        Eigen::AngleAxisd angle_axis(R);
        return angle_axis.angle() * angle_axis.axis();
    }

    static Eigen::Matrix3d axisAngleToRotationMatrix(const Eigen::Vector3d& axis_angle) {
        double angle = axis_angle.norm();
        if (angle < 1e-10) return Eigen::Matrix3d::Identity();
        
        Eigen::Vector3d axis = axis_angle.normalized();
        Eigen::AngleAxisd angle_axis(angle, axis);
        return angle_axis.toRotationMatrix();
    }

    static std::vector<std::pair<Eigen::Vector2d, Eigen::Vector3d>> 
    loadCorrespondences(const std::string& path) {
        std::vector<std::pair<Eigen::Vector2d, Eigen::Vector3d>> correspondences;
        std::ifstream file(path);
        if (!file.is_open()) return correspondences;

        double px, py, X, Y, Z, idx;
        while (file >> px >> py >> X >> Y >> Z >> idx) {
            correspondences.emplace_back(
                Eigen::Vector2d(px, py), 
                Eigen::Vector3d(X, Y, Z)
            );
        }
        return correspondences;
    }

public:
    static void solveExtrinsicEstimation(
        const std::string& image_path, 
        const std::string& gt_path,
        const Eigen::Vector4d& camera_params,
        bool use_random_initialization = false) 
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        const double rotation_noise_stddev = 2.0, translation_noise_stddev = 50.0;

        Eigen::Matrix4d T_ground_truth = readGroundTruthPose(gt_path);
        
        Eigen::Vector3d translation_gt = T_ground_truth.block<3, 1>(0, 3);
        Eigen::Vector3d axis_angle_gt = rotationMatrixToAxisAngle(T_ground_truth.block<3, 3>(0, 0));

        double axis_angle[3];
        double translation[3];

        if (use_random_initialization) {
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (int i = 0; i < 3; ++i) {
                axis_angle[i] = dist(gen);
                translation[i] = dist(gen) * translation_noise_stddev;
            }
        } else {
            Eigen::Vector3d noisy_axis_angle = axis_angle_gt + Eigen::Vector3d::Random() * rotation_noise_stddev;
            Eigen::Vector3d noisy_translation = translation_gt + Eigen::Vector3d::Random() * translation_noise_stddev;

            for (int i = 0; i < 3; ++i) {
                axis_angle[i] = noisy_axis_angle[i];
                translation[i] = noisy_translation[i];
            }
        }

        ceres::Problem problem;
        
        auto correspondences = loadCorrespondences(image_path);

        for (const auto& [obs, point] : correspondences) {
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                    new ReprojectionError(obs, point, camera_params)
                );
            problem.AddResidualBlock(cost_function, nullptr, axis_angle, translation);
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_num_iterations = 100;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        Eigen::Matrix4d T_estimated = Eigen::Matrix4d::Identity();
        
        T_estimated.block<3, 3>(0, 0) = axisAngleToRotationMatrix(
            Eigen::Vector3d(axis_angle[0], axis_angle[1], axis_angle[2])
        );
        
        T_estimated.block<3, 1>(0, 3) = Eigen::Map<Eigen::Vector3d>(translation);
        
        std::cout << "Estimated:\n" << T_estimated << "\nGround truth:\n" 
                  << T_ground_truth << "\n";
        
        std::cout << summary.BriefReport() << std::endl;
    }
};

int main() {
    const Eigen::Vector4d camera_params(721.5, 721.5, 256.0, 176.0);
    std::string source_dir = __FILE__;
    source_dir = source_dir.substr(0, source_dir.find_last_of("/\\")) + "/data";

    for (int i = 0; i < 5; ++i) {
        std::string image_path = source_dir + "/Correspondences/pose_" + std::to_string(i) + ".txt";
        std::string gt_path = source_dir + "/Extrinsics/pose" + std::to_string(i) + ".txt";
        std::cout << "Processing image " << i << std::endl;

        std::cout << "Estimating with good estimate..." << std::endl;
        ExtrinsicCalibration::solveExtrinsicEstimation(image_path, gt_path, camera_params);

        std::cout << "Estimating with random initialization..." << std::endl;
        ExtrinsicCalibration::solveExtrinsicEstimation(image_path, gt_path, camera_params, true);
    }

    return 0;
}