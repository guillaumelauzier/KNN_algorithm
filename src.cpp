#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Function that computes the Euclidean distance between two vectors
double euclideanDistance(const VectorXd& x, const VectorXd& y) {
return (x - y).norm();
}

// Function that performs KNN classification on the input data
int knn(const MatrixXd& X, const VectorXd& y, const VectorXd& x, int K) {
// Compute the distances between the input data and the query point
VectorXd distances = (X.rowwise() - x).rowwise().norm();

// Sort the distances in ascending order and return the indices of the K nearest neighbors
std::vector<std::pair<double, int>> sortedDistances;
for (int i = 0; i < distances.size(); i++) {
sortedDistances.push_back(std::make_pair(distances[i], i));
}
std::sort(sortedDistances.begin(), sortedDistances.end());

// Compute the majority label among the K nearest neighbors
std::map<int, int> labelCounts;
for (int i = 0; i < K; i++) {
int label = y[sortedDistances[i].second];
if (labelCounts.find(label) == labelCounts.end()) {
labelCounts[label] = 1;
} else {
labelCounts[label]++;
}
}

// Return the majority label
int maxLabel = -1;
int maxCount = -1;
for (const auto& entry : labelCounts) {
if (entry.second > maxCount) {
maxLabel = entry.first;
maxCount = entry.second;
}
}
return maxLabel;
}

int main() {
// Define the input data and labels
MatrixXd X(5, 3);
X << 1, 2, 3,
2, 3, 4,
3, 4, 5,
4, 5, 6,
5, 6, 7;
VectorXd y(5);
y << 0, 0, 0, 1, 1;

// Define the query point
VectorXd x(3);
x << 2, 2, 2;

// Perform KNN classification on the query point
int K = 3;
int label = knn(X, y, x, K);
std::cout << "KNN output: " << label << std::endl;

return 0;
}
