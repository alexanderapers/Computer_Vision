#pragma once
#include "../BackgroundSubtraction.h"
#include <opencv2/video/background_segm.hpp>

namespace nl_uu_science_gmt {
    class MOG2BackgroundSubtraction :
        public BackgroundSubtraction
    {
    public:
        MOG2BackgroundSubtraction(int camera_id, float learning_rate) : BackgroundSubtraction(camera_id), learning_rate(learning_rate) {
            pMOG2 = createBackgroundSubtractorMOG2(500, 16.f, true);
        }
        void prepareBgRef() const override;
        void processForeground(Scene3DRenderer* renderer, Camera* camera) const override;
    private:
        float learning_rate;
        Ptr<BackgroundSubtractor> pMOG2;
    };
}


