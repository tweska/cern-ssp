#include <cmath>

#include <numeric>
#include <random>

// ROOT
#include "TH1.h"

#include "RHnCUDA.h"

#define MAX_ERROR  0.000000000001

template <typename T>
T arraySum(T a[], size_t n)
{
    return std::accumulate(a, a + n, 0);
}

template <typename T>
T arrayProduct(T a[], size_t n)
{
    return std::accumulate(a, a + n, 1, std::multiplies<T>());
}

using namespace ROOT::Experimental;

template <unsigned int Dim>
class HistogramTest {
protected:
    double *h_histogram;
    double *h_binEdges;  // Can be NULL!
    int    *h_binEdgesIdx;
    int    *h_nBinsAxis;
    double *h_xMin;
    double *h_xMax;
    double *h_coords;
    double *h_weights;
    bool   *h_mask;

    double *d_histogram;
    double *d_binEdges;  // Can be NULL!
    int    *d_binEdgesIdx;
    int    *d_nBinsAxis;
    double *d_xMin;
    double *d_xMax;
    double *d_coords;
    double *d_weights;
    bool   *d_mask;

    size_t bulkSize;

public:
    HistogramTest(double *histogram, double *binEdges, int *binEdgesIdx, int *nBinsAxis,
                  double *xMin, double *xMax, double *coords, double *weights,
                  bool *mask, size_t bulkSize)
    {
        h_histogram = histogram;
        h_binEdges = binEdges;
        h_binEdgesIdx = binEdgesIdx;
        h_nBinsAxis = nBinsAxis;
        h_xMin = xMin;
        h_xMax = xMax;
        h_coords = coords;
        h_weights = weights;
        h_mask = mask;
        this->bulkSize = bulkSize;
    }

    ~HistogramTest()
    {
        // TODO: Proper cleanup!
    }

    void allocateDevice()
    {
        ERRCHECK(cudaMalloc(&d_histogram, sizeof(double) * arrayProduct(h_nBinsAxis, Dim)));
        if (h_binEdges)
            ERRCHECK(cudaMalloc(&d_binEdges, sizeof(double) * arraySum(h_nBinsAxis, Dim)));
        ERRCHECK(cudaMalloc(&d_binEdgesIdx, sizeof(int) * Dim));
        ERRCHECK(cudaMalloc(&d_nBinsAxis, sizeof(int) * Dim));
        ERRCHECK(cudaMalloc(&d_xMin, sizeof(double) * Dim));
        ERRCHECK(cudaMalloc(&d_xMax, sizeof(double) * Dim));
        ERRCHECK(cudaMalloc(&d_coords, sizeof(double) * bulkSize * Dim));
        ERRCHECK(cudaMalloc(&d_weights, sizeof(double) * bulkSize * Dim));
        ERRCHECK(cudaMalloc(&d_mask, sizeof(double) * bulkSize * Dim));
    }

    void transferDevice()
    {
        if (h_binEdges)
            ERRCHECK(cudaMemcpy(d_binEdges, h_binEdges, sizeof(double) * arraySum(h_nBinsAxis, Dim), cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_binEdgesIdx, h_binEdgesIdx, sizeof(int) * Dim, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_nBinsAxis, h_nBinsAxis, sizeof(int) * Dim, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_xMin, h_xMin, sizeof(double) * Dim, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_xMax, h_xMax, sizeof(double) * Dim, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_coords, h_coords, sizeof(double) * bulkSize * Dim, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_weights, h_weights, sizeof(double) * bulkSize * Dim, cudaMemcpyHostToDevice));
        ERRCHECK(cudaMemcpy(d_mask, h_mask, sizeof(bool) * bulkSize * Dim, cudaMemcpyHostToDevice));
    }

    void run()
    {
        HistogramGlobal<double, Dim><<<4, 128>>>(
            d_histogram,
            d_binEdges,
            d_binEdgesIdx,
            d_nBinsAxis,
            d_xMin,
            d_xMax,
            d_coords,
            d_weights,
            d_mask,
            bulkSize
        );

        cudaDeviceSynchronize();
        ERRCHECK(cudaPeekAtLastError());
    }

    void transferResult()
    {
        ERRCHECK(cudaMemcpy(h_histogram, d_histogram, sizeof(double) * arrayProduct(h_nBinsAxis, Dim), cudaMemcpyDeviceToHost));
    }

    void checkResult()
    {
        // Comparing against existing ROOT Histogram implementation.
        // This implementation is fixed 1D only, for the time being.

        auto histoROOT = new TH1D(
            "",              // Name
            ";x;y",          // Title
            h_nBinsAxis[0] - 2,
            h_xMin[0],
            h_xMax[0]
        );

        histoROOT->FillN(
            bulkSize,
            h_coords,
            h_weights
        );

        const double *t_histogram = histoROOT->GetArray();

        double maxError = 0.0;
        for (size_t i = 0; i < h_nBinsAxis[0]; ++i) {
            double error = fabsl((h_histogram[i] - t_histogram[i]) / t_histogram[i]);
            if (error > maxError)
                maxError = error;
        }

        if (maxError > MAX_ERROR) {
            std::cerr << "Test failed! Relative maximum error is " << maxError << std::endl;
        }
    }

    void fullTest()
    {
        allocateDevice();
        transferDevice();
        run();
        transferResult();
        checkResult();
    }
};

class Single1DFixedUniformHistogramTest : public HistogramTest<1> {
public:
    Single1DFixedUniformHistogramTest(
        const int nBins = 102,
        const double xMinVal = 0.0,
        const double xMaxVal = 1.0,
        const size_t bulkSize = 1000
        ) : HistogramTest(
          new double[nBins],
          nullptr,
          new int[1] {-1},
          new int[1] {nBins},
          new double[1] {xMinVal},
          new double[1] {xMaxVal},
          new double[bulkSize],
          new double[bulkSize],
          new bool[bulkSize],
          bulkSize
        )
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> coords_dis(xMinVal, xMaxVal);
        std::uniform_real_distribution<> weight_dis(0.0, 1.0);

        for (size_t i = 0; i < bulkSize; ++i) {
            h_coords[i] = coords_dis(gen);
            h_weights[i] = weight_dis(gen);
            h_mask[i] = true;
        }
    }
};

int main(int argc, char **argv)
{
    auto test = Single1DFixedUniformHistogramTest(
        10 + 2,  // nBins
        0.0,     // xMin
        1.0,     // xMax
        5000000  // BulkSize
    );
    test.fullTest();

    return 0;
}
