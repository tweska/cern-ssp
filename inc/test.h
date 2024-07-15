#ifndef TEST_H
#define TEST_H

template <typename T>
T arraySum(T a[], size_t n);

template <typename T>
T arrayProduct(T a[], size_t n);

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
                  bool *mask, size_t bulkSize);
    ~HistogramTest();

    void allocateDevice();
    void transferDevice();
    void run();
    void transferResult();
    void checkResult();
    void fullTest();
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
