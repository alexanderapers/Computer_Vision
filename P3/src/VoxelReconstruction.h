/*
 * VoxelReconstruction.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert, robertoost
 */

#pragma once

#include <string>
#include <vector>

namespace nl_uu_science_gmt
{
	class BackgroundSubtraction;
	class Scene3DRenderer;
	class Camera;

	class VoxelReconstruction
	{
		const std::string m_data_path;
		const int m_cam_views_amount;

		std::vector<Camera*> m_cam_views;
		std::vector<BackgroundSubtraction*> m_bg_subtractors;
	public:
		VoxelReconstruction(const std::string &, const int);
		virtual ~VoxelReconstruction();

		static void showKeys();
		void tuneRenderer(Scene3DRenderer& renderer, Camera* camera);

		void run(int, char**);
	};

} /* namespace nl_uu_science_gmt */

