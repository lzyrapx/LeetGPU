#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace std;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

// Predict function
vector<double> predict(const vector<vector<double>> &X, const vector<double> &W) {
    int n = X.size();       // Number of samples
    int d = X[0].size();    // Number of features
    vector<double> predictions(n, 0.0);

    for (int i = 0; i < n; i++) {
        double linear_sum = 0.0;
        for (int j = 0; j < d; j++) {
            linear_sum += X[i][j] * W[j];
        }
        predictions[i] = sigmoid(linear_sum);
    }
    return predictions;
}

// Compute cross-entropy loss
double compute_loss(const vector<double> &y_true, const vector<double> &y_pred) {
    int n = y_true.size();
    double loss = 0.0;
    for (int i = 0; i < n; i++) {
        double clipped_pred = y_pred[i];
        loss += y_true[i] * log(clipped_pred) + (1 - y_true[i]) * log(1 - clipped_pred);
    }
    return -loss / n;
}

// Gradient descent function
void gradient_descent(const vector<vector<double>> &X, const vector<double> &y, vector<double> &w,
                      double learning_rate, int epochs) {
    int n = X.size();
    int d = X[0].size();

    for (int epoch = 0; epoch < epochs; epoch++) {
        vector<double> y_pred = predict(X, w);
        vector<double> gradient(d, 0.0);

        for (int j = 0; j < d; j++) {
            double grad_sum = 0.0;
            for (int i = 0; i < n; i++) {
                grad_sum += (y_pred[i] - y[i]) * X[i][j];
            }
            gradient[j] = grad_sum / n;
        }

        for (int j = 0; j < d; j++) {
            w[j] -= learning_rate * gradient[j];
        }

        if (epoch % 100 == 0) {
            double loss = compute_loss(y, y_pred);
            //cout << "Epoch " << epoch << " - Loss: " << loss << endl;
        }
    }
}

// Main function
int main() {
    // Example dataset (n = 8, d = 2)
    vector<vector<double>> X = {
        {1.0, 0.5}, 
        {1.2, 0.8}, 
        {0.8, 0.6}, 
        {1.5, 1.0},
        {-1.0,-0.5},
        {-1.2, -0.8},
        {-0.8,-0.6},
        {-1.5,-1.0}};
    vector<double> y = {1, 1, 1, 1, 0, 0, 0, 0};
    int d = X[0].size();

    vector<double> w(d, 0.0);

    double learning_rate = 0.01;
    int epochs = 10000;

    gradient_descent(X, y, w, learning_rate, epochs);

    cout << "Trained Weights: ";
    for (double wi : w) {
        cout << wi << " ";
    }
    cout << endl;

    vector<double> y_pred = predict(X, w);
    cout << "Predictions: ";
    for (double p : y_pred) {
        cout << (p > 0.5 ? 1 : 0) << " ";
    }
    cout << endl;

    return 0;
}