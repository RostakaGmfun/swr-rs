use nalgebra_glm as glm;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use std::sync::Arc;
use std::time::{Duration, Instant};

mod obj;
mod swr;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

struct SimpleVS {
    pub mvp: glm::Mat4,
    pub color: glm::Vec3,
}

impl swr::VertexShader for SimpleVS {
    fn shade(&self, vin: &swr::Vertex, vout: &mut swr::VSOutput) {
        let clip_space = self.mvp * glm::vec4(vin.x, vin.y, vin.z, 1f32);
        vout.ndc_position = glm::vec4_to_vec3(&clip_space) / clip_space.w;

        let light_dir = glm::normalize(&glm::vec3(1f32, 1f32, 1f32)); // arbitrary light
        let fake_normal = glm::normalize(&vin); // treat position as normal

        let mut diffuse = glm::dot(&fake_normal, &light_dir);
        let ambient = 0.2f32;
        diffuse = ambient + (1f32 - ambient) * diffuse;

        vout.varyings[0] = self.color.x * diffuse;
        vout.varyings[1] = self.color.y * diffuse;
        vout.varyings[2] = self.color.z * diffuse;
    }
}

struct SimpleFS();

impl swr::FragShader for SimpleFS {
    fn shade(&self, fin: &swr::FSInput, fout: &mut swr::FSOutput) {
        *fout = glm::vec3(fin.varyings[0], fin.varyings[1], fin.varyings[2]);
    }
}

fn main() -> Result<(), String> {
    let sdl_context = sdl2::init().unwrap();
    let video = sdl_context.video().unwrap();

    let window = video
        .window("swr-rs demo", WIDTH, HEIGHT)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window
        .into_canvas()
        .software()
        .build()
        .unwrap();

    let texture_creator = canvas.texture_creator();

    let mut texture = texture_creator
        .create_texture_streaming(PixelFormatEnum::RGB24, WIDTH, HEIGHT)
        .unwrap();

    let mut pipeline = swr::Pipeline::new(WIDTH, HEIGHT, WIDTH / 5, HEIGHT / 5);

    let mut event_pump = sdl_context.event_pump().unwrap();

    let (verts, indices) = obj::load_obj("./data/suzanne.obj").unwrap();

    let camera_pos = glm::vec3(0f32, 0f32, -20f32);
    let camera_target = glm::vec3(0f32, 0f32, 0f32);
    let camera_up = glm::vec3(0f32, 1f32, 0f32);

    let view = glm::look_at(&camera_pos, &camera_target, &camera_up);

    let fov: f32 = glm::radians(&glm::vec1(60f32)).x;
    let aspect_ratio: f32 = WIDTH as f32 / HEIGHT as f32;
    let near_plane: f32 = 2f32;
    let far_plane: f32 = 100f32;

    let proj = glm::perspective(fov, aspect_ratio, near_plane, far_plane);

    let mesh = swr::VertexBuffer {
        vertices: Arc::from(verts),
        indices: Arc::from(indices),
    };

    let fs = Arc::new(SimpleFS {});

    let mut frame_count: u32 = 0;
    let mut last_fps_time = Instant::now();

    let mut mb_down = false;
    let mut last_mouse_x = 0;
    let mut last_mouse_y = 0;
    let mut rot_x = 0f32;
    let mut rot_y = 0f32;
    let mut scale = 1f32;

    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::MouseButtonDown {
                    mouse_btn: sdl2::mouse::MouseButton::Left,
                    x,
                    y,
                    ..
                } => {
                    mb_down = true;
                    last_mouse_x = x;
                    last_mouse_y = y;
                }

                Event::MouseButtonUp {
                    mouse_btn: sdl2::mouse::MouseButton::Left,
                    ..
                } => {
                    mb_down = false;
                }

                Event::MouseMotion { x, y, .. } => {
                    if mb_down {
                        let dx = (x - last_mouse_x) as f32;
                        let dy = (y - last_mouse_y) as f32;

                        rot_y -= dx * 0.01;
                        rot_x += dy * 0.01;

                        last_mouse_x = x;
                        last_mouse_y = y;
                    }
                }

                Event::MouseWheel { y, .. } => {
                    scale *= 1.0 + y as f32 * 0.1;
                    scale = scale.clamp(0.1, 10.0);
                }
                _ => {}
            }
        }

        let mut model = glm::scale(&glm::identity(), &glm::vec3(scale, scale, scale));
        model = glm::rotate(&model, rot_x, &glm::vec3(1.0, 0.0, 0.0));
        model = glm::rotate(&model, rot_y, &glm::vec3(0.0, 1.0, 0.0));
        let vs = Arc::new(SimpleVS {
            mvp: proj * view * model,
            color: glm::vec3(1f32, 1f32, 1f32),
        }); // TODO: optimize this later, we don't want to allocate every iteration?

        pipeline.begin_frame();

        pipeline.draw(mesh.clone(), vs.clone(), fs.clone());

        pipeline.end_frame();

        texture
            .update(
                None,
                pipeline.framebuffer().get_color_buffer(),
                (WIDTH * 3) as usize,
            )
            .unwrap();

        canvas.copy(&texture, None, None).unwrap();
        canvas.present();

        frame_count += 1;

        let elapsed = last_fps_time.elapsed();
        if elapsed >= Duration::from_secs(1) {
            let avg_frame_ms = (elapsed.as_secs_f64() * 1000.0) / frame_count as f64;
            let fps = frame_count as f64 / elapsed.as_secs_f64();

            println!("FPS: {:.2}, avg frame: {:.3} ms", fps, avg_frame_ms);

            frame_count = 0;
            last_fps_time = Instant::now();
        }
    }

    pipeline.stop();

    Ok(())
}
