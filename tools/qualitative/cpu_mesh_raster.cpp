// Deterministic CPU rasterizer for visualization-only colored ScanNet meshes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" void rasterize_mesh(
    const double* projected, const double* colors, const int32_t* faces,
    int64_t face_count, int width, int height, uint8_t* rgba) {
  std::vector<double> depth(static_cast<size_t>(width) * height,
                            std::numeric_limits<double>::infinity());
  std::fill(rgba, rgba + static_cast<size_t>(width) * height * 4, 0);
  for (int64_t f = 0; f < face_count; ++f) {
    const int32_t i0 = faces[3*f], i1 = faces[3*f+1], i2 = faces[3*f+2];
    const double *a = projected+3*i0, *b = projected+3*i1, *c = projected+3*i2;
    if (std::min({a[2], b[2], c[2]}) <= 0.05) continue;
    const double den = (b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
    if (std::abs(den) < 1e-12) continue;
    const int x0 = std::max(0, (int)std::floor(std::min({a[0], b[0], c[0]})));
    const int x1 = std::min(width-1, (int)std::ceil(std::max({a[0], b[0], c[0]})));
    const int y0 = std::max(0, (int)std::floor(std::min({a[1], b[1], c[1]})));
    const int y1 = std::min(height-1, (int)std::ceil(std::max({a[1], b[1], c[1]})));
    const double *ca = colors+3*i0, *cb = colors+3*i1, *cc = colors+3*i2;
    for (int y = y0; y <= y1; ++y) {
      for (int x = x0; x <= x1; ++x) {
        const double px = x+.5, py = y+.5;
        const double u = ((b[1]-c[1])*(px-c[0])+(c[0]-b[0])*(py-c[1]))/den;
        const double v = ((c[1]-a[1])*(px-c[0])+(a[0]-c[0])*(py-c[1]))/den;
        const double w = 1-u-v;
        if (std::min({u,v,w}) < -1e-8) continue;
        const double inv_z = u/a[2] + v/b[2] + w/c[2];
        const double z = 1/inv_z;
        const size_t index = static_cast<size_t>(y)*width+x;
        if (z >= depth[index]) continue;
        depth[index] = z;
        for (int k = 0; k < 3; ++k) {
          const double value = (u*ca[k]/a[2]+v*cb[k]/b[2]+w*cc[k]/c[2])/inv_z;
          rgba[4*index+k] = (uint8_t)std::clamp(value*255.0+.5, 0., 255.);
        }
        rgba[4*index+3] = 255;
      }
    }
  }
}
