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
#include "GbHisto.h"

#define MIN

b8 checkArray(const usize n, const f64 *o, const f64 *t, const f64 maxError = 0.000000000001) {
    for (usize i = 0; i < n; ++i)
        if (fabsl((o[i] - t[i]) / t[i]) > maxError)
            return false;
    return true;
}

class GbHistoTest : public GbHisto<f64> {
public:
    using GbHisto::GbHisto;

    std::vector<std::unique_ptr<TH1>> hROOT;

    void Fill(u32 n, const f64 *coords, const f64 *weights = nullptr) {
        GbHisto::Fill(n, coords, weights);

        for (usize i = 0; i < nHistos; ++i) {
            const auto offset = h_histoOffset[i];
            auto *weightsPtr = weights ? &weights[i * n] : nullptr;
            const auto &histo = hROOT[i].get();
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Warray-bounds"
            if (auto h1d = dynamic_cast<TH1D*>(histo)) {
                h1d->FillN(
                    n,
                    &coords[offset * n],
                    weightsPtr
                );
            } else if (auto h2d = dynamic_cast<TH2D*>(histo)) {
                h2d->FillN(
                    n,
                    &coords[(offset + 0) * n],
                    &coords[(offset + 1) * n],
                    weightsPtr
                );
            } else if (auto h3d = dynamic_cast<TH3D*>(histo)) {
                for (usize j = 0; j < n; j++) {
                    h3d->Fill(
                        coords[(offset + 0) * n + j],
                        coords[(offset + 1) * n + j],
                        coords[(offset + 2) * n + j],
                        weightsPtr ? weightsPtr[j] : 1.0
                    );
                }
            } else {
                assert(0 == 1);
            }
            #pragma GCC diagnostic pop
        }
    }

    GbHistoTest(
        u32 nHistos, const u32 *nDims, const u32 *nBinsAxis,
        const f64 *xMin, const f64 *xMax,
        const f64 *binEdges, const i32 *binEdgesOffset,
        usize maxBulkSize=256
    ) : GbHisto(
        nHistos, nDims, nBinsAxis,
        xMin, xMax,
        binEdges, binEdgesOffset,
        maxBulkSize
    ) {
        for (usize i = 0; i < nHistos; ++i) {
            u32 offset = h_histoOffset[i];
            #pragma GCC diagnostic push
            #pragma GCC diagnostic ignored "-Warray-bounds"
            if (nDims[i] == 1) {
                hROOT.emplace_back(std::make_unique<TH1D>(
                    "", "",
                    nBinsAxis[offset + 0] - 2, xMin[offset + 0], xMax[offset + 0]
                ));
            } else if (nDims[i] == 2) {
                hROOT.emplace_back(std::make_unique<TH2D>(
                    "", "",
                    nBinsAxis[offset + 0] - 2, xMin[offset + 0], xMax[offset + 0],
                    nBinsAxis[offset + 1] - 2, xMin[offset + 1], xMax[offset + 1]
                ));
            } else if(nDims[i] == 3) {
                hROOT.emplace_back(std::make_unique<TH3D>(
                    "", "",
                    nBinsAxis[offset + 0] - 2, xMin[offset + 0], xMax[offset + 0],
                    nBinsAxis[offset + 1] - 2, xMin[offset + 1], xMax[offset + 1],
                    nBinsAxis[offset + 2] - 2, xMin[offset + 2], xMax[offset + 2]
                ));
            } else {
                assert(0 == 1);
            }
            #pragma GCC diagnostic pop
        }
    }

    b8 Check() {
        f64 fullResult[nBins];
        RetrieveResults(fullResult);

        for (usize i = 0; i < nHistos; ++i) {
            f64 *result = &fullResult[h_histoResultOffset[i]];
            const auto &histo = hROOT[i].get();
            if (auto h1d = dynamic_cast<TH1D*>(histo)) {
                const f64 *array = h1d->GetArray();
                const usize nBins = h1d->GetNbinsX() + 2;
                if (!checkArray(nBins, result, array)) {
                    std::cerr << "Error encountered in histo with id " << i << " (1D)." << std::endl;
                    std::cout << "observed="; printArray(result, nBins);
                    std::cout << "expected="; printArray(array, nBins);
                    return false;
                }
            } else if (auto h2d = dynamic_cast<TH2D*>(histo)) {
                const f64 *array = h2d->GetArray();
                const usize nBins = (h2d->GetNbinsX() + 2) * (h2d->GetNbinsY() + 2);
                if (!checkArray(nBins, result, array)) {
                    std::cerr << "Error encountered in histo with id " << i << " (2D)." << std::endl;
                    std::cout << "observed="; printArray(result, nBins);
                    std::cout << "expected="; printArray(array, nBins);
                    return false;
                }
            } else if (auto h3d = dynamic_cast<TH3D*>(histo)) {
                const f64 *array = h3d->GetArray();
                const usize nBins = (h3d->GetNbinsX() + 2) * (h3d->GetNbinsY() + 2) * (h3d->GetNbinsZ() + 2);
                if (!checkArray(nBins, result, array)) {
                    std::cerr << "Error encountered in histo with id " << i << " (3D)." << std::endl;
                    std::cerr << "observed="; printArray(result, nBins, std::cerr);
                    std::cerr << "expected="; printArray(array, nBins, std::cerr);
                    return false;
                }
            } else { return false; }
        }
        return true;
    }

    b8 FullTest(usize nValues, const f64 *coords, usize bulkSize=256) {
        for (usize i = 0; i < nValues; i += bulkSize) {
            const usize n = min(nValues - i, bulkSize);
            Fill(n, coords);
        }
        return Check();
    }

    b8 FullRandomTest(usize nValues, usize bulkSize = 256) {
        auto *coords = new f64[bulkSize * nAxis];

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> coords_dis(-10.0, 110.0);

        for (usize i = 0; i < nValues; i += bulkSize) {
            const usize n = min(nValues - i, bulkSize);
            for (usize j = 0; j < n; ++j) {
                coords[j] = coords_dis(gen);
            }
            Fill(n, coords);
        }

        std::destroy_at(coords);
        return Check();
    }
};

b8 runRandomTest(u32 nHistos, u32 *nDims, u32 *nBinsAxis, u32 nValues = 100) {
    u32 nAxis = 0;
    for (u32 i = 0; i < nHistos; ++i) {
        nAxis += nDims[i];
    }
    auto *xMin = new f64[nAxis];
    auto *xMax = new f64[nAxis];
    auto *binEdgesOffset = new i32[nAxis];
    std::fill_n(xMin, nAxis,   0.0);
    std::fill_n(xMax, nAxis, 100.0);
    std::fill_n(binEdgesOffset, nAxis, -1);

    auto test = GbHistoTest(nHistos, nDims, nBinsAxis, xMin, xMax, nullptr, binEdgesOffset);

    std::destroy_at(xMin);
    std::destroy_at(xMax);
    std::destroy_at(binEdgesOffset);

    return test.FullRandomTest(nValues);
}

TEST(GbHistoDTest, Single1DSmall) {
    u32 nHistos = 1;
    u32 nDims = 1;
    u32 nBins = 10;
    f64 xMin = 0.0;
    f64 xMax = 100.0;
    i32 binEdgesOffset = -1;

    GbHistoTest *histos = new GbHistoTest(
        nHistos, &nDims, &nBins,
        &xMin, &xMax,
        nullptr, &binEdgesOffset
    );
    f64 coords[5] = {0.42, 8.2, 42.0, 8.3, 6.5};
    ASSERT_TRUE(histos->FullTest(5, coords));
}

TEST(GbHistoDTest, Single1DRandom) {
    u32 nDims = 1;
    u32 nBins = 10;
    ASSERT_TRUE(runRandomTest(1, &nDims, &nBins, 500));
}

TEST(GbHistoDTest, DoubleEqual1DRandom) {
    u32 nDims[2] = { 1,  1};
    u32 nBins[2] = {10, 10};
    ASSERT_TRUE(runRandomTest(2, nDims, nBins, 500));
}

TEST(GbHistoDTest, DoubleUnequal1DRandom) {
    u32 nDims[2] = { 1,  1};
    u32 nBins[2] = {10, 15};
    ASSERT_TRUE(runRandomTest(2, nDims, nBins, 500));
}

TEST(GbHistoDTest, Single2DRandom) {
    u32 nDims = 2;
    u32 nBins[2] = {10, 10};
    ASSERT_TRUE(runRandomTest(1, &nDims, nBins, 500));
}

TEST(GbHistoDTest, DoubleEqual2DRandom) {
    u32 nDims[2] = {2,       2};
    u32 nBins[4] = {10, 10, 10, 10};
    ASSERT_TRUE(runRandomTest(2, nDims, nBins, 500));
}

TEST(GbHistoDTest, DoubleUnequal2DRandom) {
    u32 nDims[2] = {2,       2};
    u32 nBins[4] = {10, 12, 17, 15};
    ASSERT_TRUE(runRandomTest(2, nDims, nBins, 500));
}

TEST(GbHistoDTest, DoubleUnequal1D2DRandom) {
    u32 nDims[2] = {1,   2};
    u32 nBins[3] = {10, 17, 15};
    ASSERT_TRUE(runRandomTest(2, nDims, nBins, 500));
}

TEST(GbHistoDTest, DoubleUnequal2D3DRandom) {
    u32 nDims[2] = {2,       3};
    u32 nBins[5] = {10, 17, 15, 12, 14};
    ASSERT_TRUE(runRandomTest(2, nDims, nBins, 500));
}

TEST(GbHistoDTest, DoubleUnequal2D3DRandomLarge) {
    u32 nDims[2] = {2,       3};
    u32 nBins[5] = {10, 17, 15, 12, 14};
    ASSERT_TRUE(runRandomTest(2, nDims, nBins, 5000000));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
