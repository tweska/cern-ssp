#include <iostream>
#include <vector>
#include <memory>
#include <random>

// Google Test
#include <gtest/gtest.h>

// Root Histogramming
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"

#include <util.h>

#include "types.h"
#include "GHisto.h"

#define MIN

template <usize Dim, usize BlockSize = 256>
class GHistoTest : public GHisto<f64, Dim, BlockSize> {
public:
    using GHisto<f64, Dim, BlockSize>::GHisto;

    std::unique_ptr<TH1> hROOT;

    GHistoTest(
        const usize *nBinsAxis,
        const f64 *xMin, const f64 *xMax,
        const f64 *binEdges, const isize *binEdgesOffset,
        usize maxBulkSize=256
    ) : GHisto<f64, Dim, BlockSize>(
        nBinsAxis,
        xMin, xMax,
        binEdges, binEdgesOffset,
        maxBulkSize
    ) {
        if (Dim == 1) {
            hROOT = std::make_unique<TH1D>(
                "", "",
                nBinsAxis[0] - 2, xMin[0], xMax[0]
            );
        } else if (Dim == 2) {
            hROOT = std::make_unique<TH2D>(
                "", "",
                nBinsAxis[0] - 2, xMin[0], xMax[0],
                nBinsAxis[1] - 2, xMin[1], xMax[1]
            );
        } else if(Dim == 3) {
            hROOT = std::make_unique<TH3D>(
                "", "",
                nBinsAxis[0] - 2, xMin[0], xMax[0],
                nBinsAxis[1] - 2, xMin[1], xMax[1],
                nBinsAxis[2] - 2, xMin[2], xMax[2]
            );
        } else {
            throw std::logic_error("Histogram test with more than 3 dimensions is not implemented!");
        }
    }

    void FillN(usize n, const f64 *coords, const f64 *weights = nullptr) {
        GHisto<f64, Dim, BlockSize>::FillN(n, coords, weights);

        const auto &histo = hROOT.get();
        if (auto h1d = dynamic_cast<TH1D*>(histo)) {
            h1d->FillN(
                n,
                coords,
                weights
            );
        } else if (auto h2d = dynamic_cast<TH2D*>(histo)) {
            h2d->FillN(
                n,
                &coords[0 * n],
                &coords[1 * n],
                weights
            );
        } else if (auto h3d = dynamic_cast<TH3D*>(histo)) {
            for (usize i = 0; i < n; i++) {
                h3d->Fill(
                    coords[0 * n + i],
                    coords[1 * n + i],
                    coords[2 * n + i],
                    weights ? weights[i] : 1.0
                );
            }
        } else {
            throw std::logic_error("Histogram test with more than 3 dimensions is not implemented!");
        }
    }

    b8 Check() {
        f64 result[this->nBins];
        this->RetrieveResults(result);

        const auto &histo = hROOT.get();
        if (auto h1d = dynamic_cast<TH1D*>(histo)) {
            const f64 *array = h1d->GetArray();
            const usize nBins = h1d->GetNbinsX() + 2;
            if (!checkArray(nBins, result, array)) {
                std::cerr << "Error encountered in 1D histo." << std::endl;
                std::cout << "observed="; printArray(result, nBins);
                std::cout << "expected="; printArray(array, nBins);
                return false;
            }
        } else if (auto h2d = dynamic_cast<TH2D*>(histo)) {
            const f64 *array = h2d->GetArray();
            const usize nBins = (h2d->GetNbinsX() + 2) * (h2d->GetNbinsY() + 2);
            if (!checkArray(nBins, result, array)) {
                std::cerr << "Error encountered in 2D histo." << std::endl;
                std::cout << "observed="; printArray(result, nBins);
                std::cout << "expected="; printArray(array, nBins);
                return false;
            }
        } else if (auto h3d = dynamic_cast<TH3D*>(histo)) {
            const f64 *array = h3d->GetArray();
            const usize nBins = (h3d->GetNbinsX() + 2) * (h3d->GetNbinsY() + 2) * (h3d->GetNbinsZ() + 2);
            if (!checkArray(nBins, result, array)) {
                std::cerr << "Error encountered in 3D histo." << std::endl;
                std::cerr << "observed="; printArray(result, nBins, std::cerr);
                std::cerr << "expected="; printArray(array, nBins, std::cerr);
                return false;
            }
        } else { return false; }

        return true;
    }

    b8 FullTest(usize nValues, const f64 *coords, usize bulkSize=256) {
        for (usize i = 0; i < nValues; i += bulkSize) {
            const usize n = min(nValues - i, bulkSize);
            FillN(n, coords);
        }
        return Check();
    }

    b8 FullRandomTest(usize nValues, b8 genWeight = true, usize bulkSize = 256) {
        auto *coords = new f64[bulkSize * Dim];
        f64 *weight = nullptr;
        if (genWeight) {
            weight = new f64[bulkSize];
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> coords_dis(-10.0, 110.0);
        std::uniform_real_distribution<> weight_dis(0.0, 1.0);

        for (usize i = 0; i < nValues; i += bulkSize) {
            const usize n = min(nValues - i, bulkSize);
            for (usize j = 0; j < n * Dim; ++j) {
                coords[j] = coords_dis(gen);
            }
            if (genWeight) {
                for (usize j = 0; j < n; ++j) {
                    weight[j] = weight_dis(gen);
                }
            }
            FillN(n, coords, weight);
        }

        delete[] coords;
        delete[] weight;
        return Check();
    }
};

template <usize Dim>
b8 runRandomTest(usize *nBinsAxis, usize nValues = 100, b8 genWeights = true) {
    auto *xMin = new f64[Dim];
    auto *xMax = new f64[Dim];
    auto *binEdgesOffset = new isize[Dim];
    std::fill_n(xMin, Dim,   0.0);
    std::fill_n(xMax, Dim, 100.0);
    std::fill_n(binEdgesOffset, Dim, -1);

    auto test = GHistoTest<Dim>(nBinsAxis, xMin, xMax, nullptr, binEdgesOffset);

    delete[] xMin;
    delete[] xMax;
    delete[] binEdgesOffset;

    return test.FullRandomTest(nValues, genWeights);
}

TEST(GHistoDTest, Single1DSmall) {
    usize nBins = 10;
    f64 xMin = 0.0;
    f64 xMax = 100.0;
    isize binEdgesOffset = -1;

    GHistoTest histos = GHistoTest<1>(
        &nBins,
        &xMin, &xMax,
        nullptr, &binEdgesOffset
    );
    f64 coords[5] = {0.42, 8.2, 42.0, 8.3, 6.5};
    ASSERT_TRUE(histos.FullTest(5, coords));
}

TEST(GHistoDTest, Random1D) {
    usize nBins = 10;
    ASSERT_TRUE(runRandomTest<1>(&nBins, 500, false));
    ASSERT_TRUE(runRandomTest<1>(&nBins, 500, true));
}

TEST(GHistoDTest, Random1DLarge) {
    usize nBins = 100;
    ASSERT_TRUE(runRandomTest<1>(&nBins, 5000000, false));
    ASSERT_TRUE(runRandomTest<1>(&nBins, 5000000, true));
}

TEST(GHistoDTest, Random2D) {
    usize nBins[2] = {10, 10};
    ASSERT_TRUE(runRandomTest<2>(nBins, 500, false));
    ASSERT_TRUE(runRandomTest<2>(nBins, 500, true));
}

TEST(GHistoDTest, Random2DUnequal) {
    usize nBins[2] = {10, 16};
    ASSERT_TRUE(runRandomTest<2>(nBins, 500, false));
    ASSERT_TRUE(runRandomTest<2>(nBins, 500, true));
}

TEST(GHistoDTest, Random3D) {
    usize nBins[3] = {10, 10, 10};
    ASSERT_TRUE(runRandomTest<3>(nBins, 500, false));
    ASSERT_TRUE(runRandomTest<3>(nBins, 500, true));
}

TEST(GHistoDTest, Random3DUnequal) {
    usize nBins[3] = {10, 16, 8};
    ASSERT_TRUE(runRandomTest<3>(nBins, 500, false));
    ASSERT_TRUE(runRandomTest<3>(nBins, 500, true));
}

TEST(GHistoDTest, FillOverflow) {
    usize nBins = 10;
    f64 xMin = 0.0;
    f64 xMax = 100.0;
    isize binEdgesOffset = -1;
    GHistoTest histos = GHistoTest<1>(
        &nBins,
        &xMin, &xMax,
        nullptr, &binEdgesOffset,
        5
    );
    ASSERT_TRUE(histos.FullRandomTest(500, true, 256));
}
