#pragma once
#include "../BackgroundSubtraction.h"

namespace nl_uu_science_gmt {
	class Scene3DRenderer;

    class GaussianBackgroundSubtraction :
        public BackgroundSubtraction
    {
    public:
        GaussianBackgroundSubtraction(int camera_id) : BackgroundSubtraction(camera_id) {}
		void prepareBgRef() const override;
		void processForeground(Scene3DRenderer* renderer, Camera* camera) const override;
    };
}