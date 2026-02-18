# A toy software rasterizer

This is a multi-threaded, tiling software rasterizer written for fun.
It currently uses barycentric coordinates for triangle rasterization.

Here is a brief description of how it works:

1. Indexed geometry is passed into Pipeline::draw(). swr supports positions and normals so far and this particular vertex layout is hard-coded.
2. Vertex Shading. Each vertex undergoes "vertex shading", where the shader is a user-provided callback that normally transforms model-space vertices into "clip space" (projection onto 2D screen).
3. Triangle Binning. Triangles are binned into corresponding tiles based on their coordinates and sent to per-tile queues for rasterization. A single triangle can go into multiple tile queues if it spans multiple tiles.
4. Rasterization, Depth Test, and Fragment Shading. One of the threads picks up a tile for work and drains the tile queue. Rasterizer iterates over pixel within a triangle's bounding box and determines if it's located inside the triangle. Depth test is performed against the depth buffer and the fragment may be discarded if it is occluded.
   The fragment color is determined by calling a user-provided callback for "fragment shading".
5. Pipeline::end_frame() blocks on internal barriers waiting for all drawing operations to finish. After that, framebuffer can be safely copied to the output (e.g. for display in a window).

## Interesting but very challenging TODOs

* [ ] Implement "advanced rasterization": https://web.archive.org/web/20140514220546/http://devmaster.net/posts/6145/advanced-rasterization
* [ ] Consider SIMD optimizations for rasterization.
* [ ] Perspective-correct texture mapping.
* [ ] Multisample anti-aliasing (MSAA).
