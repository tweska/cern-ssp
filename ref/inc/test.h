#ifndef TEST_H
#define TEST_H

template <typename T>
T arraySum(T a[], size_t n);

template <typename T>
T arrayProduct(T a[], size_t n);

class AbstractHistogramTest {
public:
    virtual ~AbstractHistogramTest() = default;
    virtual void allocateDevice() = 0;
    virtual void transferDevice() = 0;
    virtual void run() = 0;
    virtual void transferResult() = 0;
    virtual void checkResult() = 0;
    virtual void fullTest() = 0;
};

template <unsigned int Dim>
class HistogramTest : public AbstractHistogramTest {
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
                  bool *mask, size_t bulkSize);
    ~HistogramTest() override;
    void allocateDevice() override;
    void transferDevice() override;
    void run() override;
    void transferResult() override;
    void checkResult() override;
    void fullTest() override;
};

class BatchedHistogramTest : public AbstractHistogramTest {
public:
    double **d_histograms;
    double **d_binEdges;
    int **d_binEdgesIdx;
    int **d_nBinsAxis;
    double **d_xMin;
    double **d_xMax;
    double **d_coords;
    double **d_weights;
    bool **d_mask;

    double **h_histograms;
    double **h_binEdges;
    int **h_binEdgesIdx;
    int **h_nBinsAxis;
    double **h_xMin;
    double **h_xMax;
    double **h_coords;
    double **h_weights;
    bool **h_mask;

    int nHistograms;
    size_t bulkSize;

    BatchedHistogramTest(
        double **histograms, int nHistograms,
        double **binEdges, int **binEdgesIdx, int **nBinsAxis,
        double **xMin, double **xMax,
        double **coords, double **weights, bool **mask, size_t bulkSize
    );

    ~BatchedHistogramTest() override;
    void allocateDevice() override;
    void transferDevice() override;
    void run() override;
    void transferResult() override;
    void checkResult() override;
    // void fullTest() override;
};

class Single1DFixedUniformHistogramTest : public HistogramTest<1> {
public:
    Single1DFixedUniformHistogramTest(int nBins = 102, double xMinVal = 0.0,
                                      double xMaxVal = 1.0, size_t bulkSize = 1000);

};

template class HistogramTest<1>;
template class HistogramTest<2>;
template class HistogramTest<3>;

#endif //TEST_H
