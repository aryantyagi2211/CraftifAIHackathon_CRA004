// Custom detection (bbox) parser (generated).
// Scaffold validates the output layer and exposes the raw tensor(s); the DECODE
// block (per-model) populates `objectList` with NvDsInferObjectDetectionInfo.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "nvdsinfer_custom_impl.h"

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList);

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(ParseConfigDetPgie);

extern "C" bool ParseConfigDetPgie(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferObjectDetectionInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    auto const& layer0 = outputLayersInfo[0];

    const unsigned numClasses = detectionParams.numClassesConfigured;
    const float netW = static_cast<float>(networkInfo.width);
    const float netH = static_cast<float>(networkInfo.height);
    const float* data = static_cast<const float*>(layer0.buffer);

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "[HelmetDet] first call: layer=%s numDims=%d "
                "networkInput=%.0fx%.0f numClasses=%u outputs=%zu\n",
                layer0.layerName ? layer0.layerName : "?",
                layer0.inferDims.numDims, netW, netH, numClasses,
                outputLayersInfo.size());
    }

    // ==== BEGIN GENERATED DECODE ====
    // Engine output shape [1,7,8400] → after batch strip: numDims=2, d={7, 8400}
    // Layout: 7 rows x 8400 columns.  Row 0-3 = cx,cy,w,h; rows 4..6 = class scores.
    // We need to transpose: for anchor i, feature j → data[j * numAnchors + i].

    const unsigned int featureDim = layer0.inferDims.d[0]; // 7  (4 box + 3 classes)
    const unsigned int numAnchors = layer0.inferDims.d[1]; // 8400

    // Sanity: featureDim should be 4 + numClasses
    if (featureDim != 4u + numClasses) {
        // Try transposed layout: d={8400, 7}
        // Fall through with swapped interpretation if that matches
    }

    unsigned int actualFeature = featureDim;
    unsigned int actualAnchors = numAnchors;
    bool transposed = false; // data already in [anchors, feature] order

    if (actualFeature != 4u + numClasses && actualAnchors == 4u + numClasses) {
        // Swap: dims are [8400, 7] already row-major per anchor
        actualFeature = numAnchors;
        actualAnchors = featureDim;
        transposed = true;
    }

    for (unsigned int i = 0; i < actualAnchors; ++i) {
        // Find best class
        float bestScore = 0.0f;
        unsigned int bestClass = 0;
        for (unsigned int c = 0; c < numClasses; ++c) {
            float score;
            if (transposed) {
                score = data[i * actualFeature + 4 + c];
            } else {
                score = data[(4 + c) * actualAnchors + i];
            }
            if (score > bestScore) {
                bestScore = score;
                bestClass = c;
            }
        }

        // Suppress class 1 (LP) entirely
        if (bestClass == 1) continue;

        // Per-class threshold
        float thr = (bestClass < detectionParams.perClassPreclusterThreshold.size())
                     ? detectionParams.perClassPreclusterThreshold[bestClass]
                     : 0.25f;
        if (bestScore < thr) continue;

        // Scores are post-sigmoid probabilities from YOLOv8 export — threshold directly.

        float cx, cy, bw, bh;
        if (transposed) {
            cx = data[i * actualFeature + 0];
            cy = data[i * actualFeature + 1];
            bw = data[i * actualFeature + 2];
            bh = data[i * actualFeature + 3];
        } else {
            cx = data[0 * actualAnchors + i];
            cy = data[1 * actualAnchors + i];
            bw = data[2 * actualAnchors + i];
            bh = data[3 * actualAnchors + i];
        }

        // cx,cy,w,h are in network-input pixel coords for YOLOv8
        float left   = cx - bw * 0.5f;
        float top    = cy - bh * 0.5f;

        // Clamp to network input
        left  = std::max<float>(0.0f, std::min<float>(left, netW));
        top   = std::max<float>(0.0f, std::min<float>(top, netH));
        float right  = std::max<float>(0.0f, std::min<float>(cx + bw * 0.5f, netW));
        float bottom = std::max<float>(0.0f, std::min<float>(cy + bh * 0.5f, netH));

        float finalW = right - left;
        float finalH = bottom - top;

        // Drop degenerate boxes
        if (finalW <= 1.0f || finalH <= 1.0f) continue;

        NvDsInferObjectDetectionInfo obj;
        obj.classId = bestClass;
        obj.left = left;
        obj.top = top;
        obj.width = finalW;
        obj.height = finalH;
        obj.detectionConfidence = bestScore;
        objectList.push_back(obj);
    }
    // ==== END GENERATED DECODE ====
    return true;
}