#pragma once

namespace nl_uu_science_gmt
{
	class Scene3DRenderer;
	class Camera;

	class BackgroundSubtraction
	{
	public:
		BackgroundSubtraction(int camera_id) : camera_id(camera_id) {}
		int camera_id;
		virtual void prepareBgRef() const = 0;

		/// <summary>
		/// Create an 8 bit image where only the foreground of the scene is white(255).
		/// </summary>
		virtual void processForeground(Scene3DRenderer * renderer, Camera * camera) const = 0;
	};
}