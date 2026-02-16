use nalgebra_glm as glm;
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

fn triangle_area(a: glm::Vec2, b: glm::Vec2, c: glm::Vec2) -> f32 {
    // Shoelace formula
    // NOTE the sign - controls triangle winding
    0.5 * (a.x * b.y + b.x * c.y + c.x * a.y - a.y * b.x - b.y * c.x - c.y * a.x)
}

fn tiles_per_row(fb_w: u32, tile_w: u32) -> u32 {
    fb_w / tile_w
}

fn tiles_per_col(fb_h: u32, tile_h: u32) -> u32 {
    fb_h / tile_h
}

pub struct AlignedBuffer<T> {
    ptr: *mut T,
    len: usize,
    layout: Layout,
}

impl<T> AlignedBuffer<T> {
    fn new(len: usize, align: usize) -> Self {
        let size = len * std::mem::size_of::<T>();
        let layout = Layout::from_size_align(size, align).unwrap();

        unsafe {
            let ptr = alloc_zeroed(layout).cast::<T>();
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }

            Self { ptr, len, layout }
        }
    }

    fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl<T> Drop for AlignedBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.cast(), self.layout);
        }
    }
}

pub struct Framebuffer {
    width: u32,
    height: u32,

    tile_w: u32,
    tile_h: u32,

    color_buffer: AlignedBuffer<u8>,
    depth_buffer: AlignedBuffer<f32>,
}

impl Framebuffer {
    const ALIGNMENT: usize = 64;

    pub fn new(width: u32, height: u32, tile_w: u32, tile_h: u32) -> Self {
        let color_buffer =
            AlignedBuffer::new((width * height * 3).try_into().unwrap(), Self::ALIGNMENT);
        let depth_buffer =
            AlignedBuffer::new((width * height).try_into().unwrap(), Self::ALIGNMENT);

        Self {
            width,
            height,

            tile_w,
            tile_h,

            color_buffer,
            depth_buffer,
        }
    }

    fn clear(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.color_buffer.as_mut_ptr(), 0, self.color_buffer.len());
            std::ptr::write_bytes(self.depth_buffer.as_mut_ptr(), 0, self.depth_buffer.len());
        }
    }

    pub fn get_color_buffer(&self) -> &[u8] {
        self.color_buffer.as_slice()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    fn get_color_ptr(&self) -> *mut u8 {
        self.color_buffer.as_mut_ptr()
    }

    fn get_depth_ptr(&self) -> *mut f32 {
        self.depth_buffer.as_mut_ptr()
    }
}

pub type Vertex = glm::Vec3;
pub type FSOutput = glm::Vec3;

pub struct VertexBuffer {
    pub vertices: Arc<[Vertex]>,
    pub indices: Arc<[[u16; 3]]>,
}

impl Clone for VertexBuffer {
    fn clone(&self) -> Self {
        Self {
            vertices: self.vertices.clone(),
            indices: self.indices.clone(),
        }
    }
}

pub struct VSOutput {
    pub ndc_position: glm::Vec3,
    pub varyings: [f32; 3], // TODO: make this a vec and add VertexShader::num_varyings()?
}

pub struct FSInput {
    pub varyings: [f32; 3],
}

impl VSOutput {
    pub fn new() -> Self {
        Self {
            ndc_position: glm::Vec3::zeros(),
            varyings: [0f32; 3],
        }
    }
}

impl Clone for VSOutput {
    fn clone(&self) -> Self {
        Self {
            ndc_position: self.ndc_position.clone(),
            varyings: self.varyings,
        }
    }
}

pub trait VertexShader: Send + Sync {
    fn shade(&self, vin: &Vertex, vout: &mut VSOutput);
}

pub trait FragShader: Send + Sync {
    fn shade(&self, fin: &FSInput, fout: &mut FSOutput);
}

pub type VertexShaderFn = Arc<dyn VertexShader>;
pub type FragShaderFn = Arc<dyn FragShader>;

struct DrawCmdPayload {
    vb: VertexBuffer,
    vertex_shader: VertexShaderFn,
    frag_shader: FragShaderFn,
}

enum DrawCallCmd {
    Draw(DrawCmdPayload),
    Print(String),
}

pub struct Barrier {
    counter: AtomicUsize,
    done_mutex: Mutex<()>,
    done_cond: Condvar,
    name: &'static str,
}

impl Barrier {
    pub fn new(name: &'static str) -> Self {
        Self {
            counter: AtomicUsize::new(0),
            done_mutex: Mutex::new(()),
            done_cond: Condvar::new(),
            name: name,
        }
    }

    pub fn inc(&self) {
        self.counter.fetch_add(1, Ordering::SeqCst);
    }

    pub fn dec(&self) {
        let _guard = self.done_mutex.lock().unwrap();
        if self.counter.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.done_cond.notify_all();
        }
    }

    pub fn wait(&self) {
        let mut guard = self.done_mutex.lock().unwrap();

        while self.counter.load(Ordering::SeqCst) != 0 {
            guard = self.done_cond.wait(guard).unwrap();
        }
    }

    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let mut guard = self.done_mutex.lock().unwrap();
        let start = std::time::Instant::now();

        while self.counter.load(Ordering::SeqCst) != 0 {
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return false;
            }

            let remaining = timeout - elapsed;
            let (g, result) = self.done_cond.wait_timeout(guard, remaining).unwrap();
            guard = g;

            if result.timed_out() {
                return false;
            }
        }

        true
    }

    fn num_pending(&self) -> usize {
        return self.counter.load(Ordering::SeqCst);
    }
}

impl Drop for Barrier {
    fn drop(&mut self) {
        let _guard = self.done_mutex.lock().unwrap();
        self.counter.store(0, Ordering::SeqCst);
        self.done_cond.notify_all();
    }
}

struct TriangleBatch {
    frag_shader: Option<FragShaderFn>,
    triangles: Vec<[VSOutput; 3]>,
}

impl TriangleBatch {
    fn new() -> Self {
        Self {
            frag_shader: None,
            triangles: Vec::with_capacity(DrawCallWorker::TRI_BATCH_MAX_SIZE),
        }
    }

    fn clear(&mut self) {
        self.triangles.clear();
        self.frag_shader = None;
    }
}

struct Tile {
    job_counter: AtomicUsize,
    sender: crossbeam::channel::Sender<TriangleBatch>,
    receiver: crossbeam::channel::Receiver<TriangleBatch>,

    x: u32,
    y: u32,
    w: u32,
    h: u32,

    color_ptr: *mut u8,
    depth_ptr: *mut f32,
    fb_width: u32,
}

impl Tile {
    const TRIANGLE_BATCH_QUEUE_LEN: usize = 128;

    fn new(x: u32, y: u32, w: u32, h: u32, fb: &Framebuffer) -> Self {
        let (sender, receiver) = crossbeam::channel::bounded(Self::TRIANGLE_BATCH_QUEUE_LEN);
        Self {
            job_counter: AtomicUsize::new(0),
            sender,
            receiver,

            x,
            y,
            w,
            h,

            color_ptr: fb.get_color_ptr(),
            depth_ptr: fb.get_depth_ptr(),
            fb_width: fb.width(),
        }
    }

    fn x(&self) -> u32 {
        self.x
    }

    fn y(&self) -> u32 {
        self.y
    }

    fn w(&self) -> u32 {
        self.w
    }

    fn h(&self) -> u32 {
        self.h
    }

    fn get_color_row(&self, row_num: u32) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.color_ptr
                    .add(((self.x + (self.y + row_num) * self.fb_width) * 3) as usize),
                self.w as usize * 3,
            )
        }
    }

    fn get_depth_row(&self, row_num: u32) -> &mut [f32] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.depth_ptr
                    .add((self.x + (self.y + row_num) * self.fb_width) as usize),
                self.w as usize,
            )
        }
    }

    fn process_triangle_batch(&self, batch: TriangleBatch) {
        let frag_shader = match batch.frag_shader {
            Some(f) => f,
            None => panic!("expected frag shader"),
        };
        for t in batch.triangles {
            self.process_triangle(&t, &frag_shader);
        }
    }

    fn process_triangle(&self, tri: &[VSOutput; 3], frag_shader: &FragShaderFn) {
        let a = &tri[0].ndc_position;
        let b = &tri[1].ndc_position;
        let c = &tri[2].ndc_position;

        let full_area = triangle_area(
            glm::vec3_to_vec2(a),
            glm::vec3_to_vec2(b),
            glm::vec3_to_vec2(c),
        );

        let bb_min = glm::vec2(a.x.min(b.x).min(c.x), a.y.min(b.y).min(c.y));

        let bb_max = glm::vec2(a.x.max(b.x).max(c.x), a.y.max(b.y).max(c.y));

        let bbox_xmin = (bb_min.x.floor() as i32).max(self.x as i32);
        let bbox_ymin = (bb_min.y.floor() as i32).max(self.y as i32);

        let bbox_xmax = (bb_max.x.ceil() as i32).min(self.x as i32 + self.w as i32);
        let bbox_ymax = (bb_max.y.ceil() as i32).min(self.y as i32 + self.h as i32);

        for y in bbox_ymin - self.y as i32..bbox_ymax - self.y as i32 {
            let color_row = self.get_color_row(y as u32);
            let depth_row = self.get_depth_row(y as u32);
            for x in bbox_xmin - self.x as i32..bbox_xmax - self.x as i32 {
                let point = glm::vec2(
                    (x + self.x as i32) as f32 + 0.5f32,
                    (y + self.y as i32) as f32 + 0.5f32,
                );
                let mut ta = triangle_area(glm::vec3_to_vec2(b), glm::vec3_to_vec2(c), point);
                let mut tb = triangle_area(glm::vec3_to_vec2(c), glm::vec3_to_vec2(a), point);
                let mut tc = triangle_area(glm::vec3_to_vec2(a), glm::vec3_to_vec2(b), point);
                if ta < -0.5f32 || tb < -0.5f32 || tc < -0.5f32 {
                    continue;
                }
                ta /= full_area;
                tb /= full_area;
                tc = 1f32 - ta - tb;
                let depth = 1f32 / (a.z * ta + b.z * tb + c.z * tc);
                if depth_row[x as usize] < depth {
                    let mut fin = FSInput {
                        varyings: [0f32; 3],
                    };
                    for i in 0..fin.varyings.len() {
                        fin.varyings[i] = tri[0].varyings[i] * ta
                            + tri[1].varyings[i] * tb
                            + tri[2].varyings[i] * tc;
                    }
                    let mut fout = glm::vec3(0f32, 0f32, 0f32);
                    frag_shader.shade(&fin, &mut fout);
                    fout = glm::clamp(&fout, 0f32, 1f32);
                    color_row[(x * 3) as usize + 0] = (fout.x * 255f32) as u8;
                    color_row[(x * 3) as usize + 1] = (fout.y * 255f32) as u8;
                    color_row[(x * 3) as usize + 2] = (fout.z * 255f32) as u8;
                    depth_row[x as usize] = depth;
                }
            }
        }
    }
}

unsafe impl Sync for Tile {}
unsafe impl Send for Tile {}

struct TileScheduler {
    tile_threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
    tile_queue: crossbeam::channel::Sender<u32>,
    tile_queue_barrier: Arc<Barrier>,
    tiles: Vec<Tile>,
}

impl TileScheduler {
    const TILE_THREAD_COUNT: usize = 14;
    const TILE_QUEUE_LEN: usize = 32;

    fn new(fb: &Framebuffer, tile_w: u32, tile_h: u32) -> Arc<Self> {
        let num_tiles = tiles_per_row(fb.width(), tile_w) * tiles_per_col(fb.height(), tile_h);
        let tile_threads = Mutex::new(Vec::with_capacity(Self::TILE_THREAD_COUNT));
        let (tile_queue, tile_queue_rx) = crossbeam::channel::bounded(Self::TILE_QUEUE_LEN);
        let tile_queue_barrier = Arc::new(Barrier::new("tqb"));
        let mut tiles = Vec::with_capacity(num_tiles as usize);

        for y in 0..tiles_per_col(fb.height(), tile_h) {
            for x in 0..tiles_per_row(fb.width(), tile_w) {
                let tile_x = x * tile_w;
                let tile_y = y * tile_h;
                tiles.push(Tile::new(tile_x, tile_y, tile_w, tile_h, fb));
            }
        }

        let shared_self = Arc::new(Self {
            tile_threads,
            tile_queue,
            tile_queue_barrier,
            tiles,
        });

        for i in 0..Self::TILE_THREAD_COUNT {
            let rx = tile_queue_rx.clone();
            let done_barrier = shared_self.tile_queue_barrier.clone();
            let scheduler_weak = Arc::<TileScheduler>::downgrade(&shared_self);
            let builder = thread::Builder::new().name(format!("T#{}", i));
            let handle = builder
                .spawn(move || {
                    loop {
                        match rx.recv() {
                            Ok(tile_id) => {
                                let scheduler = match scheduler_weak.upgrade() {
                                    Some(arc) => arc,
                                    None => {
                                        break;
                                    }
                                };
                                let tile = &scheduler.tiles[tile_id as usize];

                                loop {
                                    match tile.receiver.try_recv() {
                                        Ok(job) => {
                                            tile.process_triangle_batch(job);
                                            if tile.job_counter.fetch_sub(1, Ordering::AcqRel) == 1
                                            {
                                                break;
                                            }
                                        }
                                        Err(_) => break,
                                    }
                                }

                                done_barrier.dec();
                            }
                            Err(_) => break,
                        }
                    }
                })
                .unwrap();

            shared_self.tile_threads.lock().unwrap().push(handle);
        }

        shared_self
    }

    fn submit_triangle_batch(&self, tile_id: u32, tb: TriangleBatch) {
        let tile = &self.tiles[tile_id as usize];

        let prev = tile.job_counter.fetch_add(1, Ordering::AcqRel);

        tile.sender.send(tb).unwrap();

        if prev == 0 {
            self.tile_queue_barrier.inc();
            self.tile_queue.send(tile_id).unwrap();
        }
    }

    fn wait(&self) {
        self.tile_queue_barrier.wait()
    }

    fn wait_timeout(&self, timeout: std::time::Duration) -> bool {
        self.tile_queue_barrier.wait_timeout(timeout)
    }
}

struct DrawCallWorker {
    fb_w: u32,
    fb_h: u32,
    tile_w: u32,
    tile_h: u32,

    triangle_write_cache: Vec<TriangleBatch>,
    tile_scheduler: Arc<TileScheduler>,
}

impl DrawCallWorker {
    const TRI_BATCH_MAX_SIZE: usize = 1024;

    fn new(
        fb_w: u32,
        fb_h: u32,
        tile_w: u32,
        tile_h: u32,
        tile_scheduler: Arc<TileScheduler>,
    ) -> Self {
        let num_tiles = tiles_per_row(fb_w, tile_w) * tiles_per_col(fb_h, tile_h);
        let mut triangle_write_cache = Vec::with_capacity(num_tiles as usize);
        for _ in 0..num_tiles {
            triangle_write_cache.push(TriangleBatch::new());
        }
        Self {
            fb_w,
            fb_h,
            tile_w,
            tile_h,
            triangle_write_cache,
            tile_scheduler,
        }
    }

    fn handle(&mut self, cmd: &DrawCmdPayload) {
        let mut v_outputs: Vec<VSOutput> = Vec::with_capacity(cmd.vb.vertices.len());
        for v in &*cmd.vb.vertices {
            let mut vout = VSOutput::new();
            cmd.vertex_shader.shade(&v, &mut vout);
            vout.ndc_position = vout
                .ndc_position
                .component_mul(&glm::vec3(0.5f32, 0.5f32, 0.5f32))
                + glm::vec3(0.5f32, 0.5f32, 0.5f32);
            vout.ndc_position.x *= self.fb_w as f32;
            vout.ndc_position.y *= self.fb_h as f32;
            v_outputs.push(vout);
        }

        for vi in &*cmd.vb.indices {
            let a = &v_outputs[vi[0] as usize];
            let b = &v_outputs[vi[1] as usize];
            let c = &v_outputs[vi[2] as usize];

            let bb_min = glm::vec2(
                a.ndc_position.x.min(b.ndc_position.x).min(c.ndc_position.x),
                a.ndc_position.y.min(b.ndc_position.y).min(c.ndc_position.y),
            );

            let bb_max = glm::vec2(
                a.ndc_position.x.max(b.ndc_position.x).max(c.ndc_position.x),
                a.ndc_position.y.max(b.ndc_position.y).max(c.ndc_position.y),
            );

            let bbox_xmin = bb_min.x.floor() as i32;
            let bbox_ymin = bb_min.y.floor() as i32;
            let bbox_xmax = bb_max.x.ceil() as i32;
            let bbox_ymax = bb_max.y.ceil() as i32;

            // frustum culling
            if bbox_xmin < 0
                || bbox_ymin < 0
                || bbox_xmax > self.fb_w as i32
                || bbox_ymax > self.fb_h as i32
            {
                continue;
            }

            let full_area = triangle_area(
                glm::vec3_to_vec2(&a.ndc_position),
                glm::vec3_to_vec2(&b.ndc_position),
                glm::vec3_to_vec2(&c.ndc_position),
            );
            // backface culling
            if full_area < 0f32 {
                continue;
            }

            // binning
            let tile_xmin = bbox_xmin / self.tile_w as i32;
            let tile_ymin = bbox_ymin / self.tile_h as i32;
            let tile_xmax = bbox_xmax / self.tile_w as i32;
            let tile_ymax = bbox_ymax / self.tile_h as i32;

            for x in tile_xmin..=tile_xmax {
                for y in tile_ymin..=tile_ymax {
                    self.submit_triangle(x as u32, y as u32, &a, &b, &c, cmd.frag_shader.clone());
                }
            }
        }
        self.flush_triangle_cache();
    }

    fn submit_triangle(
        &mut self,
        tile_x: u32,
        tile_y: u32,
        a: &VSOutput,
        b: &VSOutput,
        c: &VSOutput,
        frag_shader: FragShaderFn,
    ) {
        let tile_id = (tile_y * tiles_per_row(self.fb_w, self.tile_w) + tile_x) as usize;
        let batch = &mut self.triangle_write_cache[tile_id];

        match &batch.frag_shader {
            None => {
                batch.frag_shader = Some(frag_shader.clone());
            }
            Some(existing) if !Arc::ptr_eq(existing, &frag_shader) => {
                self.submit_batch(tile_id);
                let batch = &mut self.triangle_write_cache[tile_id];
                batch.frag_shader = Some(frag_shader.clone());
            }
            _ => {}
        }

        let batch = &mut self.triangle_write_cache[tile_id];
        batch.triangles.push([a.clone(), b.clone(), c.clone()]);

        if batch.triangles.len() == Self::TRI_BATCH_MAX_SIZE {
            self.submit_batch(tile_id);
        }
    }

    fn submit_batch(&mut self, tile_id: usize) {
        let batch = &mut self.triangle_write_cache[tile_id];

        if batch.triangles.is_empty() {
            return;
        }

        let to_send = std::mem::replace(batch, TriangleBatch::new());

        self.tile_scheduler
            .submit_triangle_batch(tile_id as u32, to_send);
    }

    fn flush_triangle_cache(&mut self) {
        for tile_id in 0..self.triangle_write_cache.len() {
            self.submit_batch(tile_id);
        }
    }
}

pub struct Pipeline {
    framebuffer: Framebuffer,
    draw_call_threads: Vec<std::thread::JoinHandle<()>>,
    draw_call_queue: crossbeam::channel::Sender<DrawCallCmd>,
    draw_call_barrier: Arc<Barrier>,
    tile_scheduler: Arc<TileScheduler>,
}

impl Pipeline {
    const DRAW_CALL_QUEUE_LEN: usize = 32;
    const DRAW_CALL_THREAD_COUNT: usize = 2;

    pub fn new(width: u32, height: u32, tile_w: u32, tile_h: u32) -> Self {
        let mut threads = Vec::with_capacity(Self::DRAW_CALL_THREAD_COUNT);
        let fb = Framebuffer::new(width, height, tile_w, tile_h);
        let (draw_call_queue, receiver) = crossbeam::channel::bounded(Self::DRAW_CALL_QUEUE_LEN);
        let draw_call_barrier = Arc::new(Barrier::new("dqb"));
        let tile_scheduler = TileScheduler::new(&fb, tile_w, tile_h);

        for i in 0..Self::DRAW_CALL_THREAD_COUNT {
            let rx = receiver.clone();
            let done_barrier = draw_call_barrier.clone();
            let tile_sched = tile_scheduler.clone();

            let builder = thread::Builder::new().name(format!("DC#{}", i));
            let handle = builder
                .spawn(move || {
                    let mut draw_call_worker =
                        DrawCallWorker::new(width, height, tile_w, tile_h, tile_sched);
                    loop {
                        match rx.recv() {
                            Ok(DrawCallCmd::Draw(cmd)) => {
                                draw_call_worker.handle(&cmd);
                                done_barrier.dec()
                            }

                            Ok(DrawCallCmd::Print(msg)) => {
                                thread::sleep(Duration::from_millis(1));
                                println!("Job: {}", msg);
                                std::io::stdout().flush().unwrap();
                                done_barrier.dec()
                            }

                            Err(_) => {
                                break;
                            }
                        }
                    }
                })
                .unwrap();

            threads.push(handle);
        }

        Self {
            framebuffer: fb,
            draw_call_threads: threads,
            draw_call_queue: draw_call_queue,
            draw_call_barrier: draw_call_barrier,
            tile_scheduler: tile_scheduler,
        }
    }

    pub fn stop(self) {
        drop(self.draw_call_queue);

        for handle in self.draw_call_threads {
            let _ = handle.join();
        }
    }

    pub fn test(&self, msg: &str) {
        self.draw_call_barrier.inc();
        self.draw_call_queue
            .send(DrawCallCmd::Print(msg.to_string()))
            .unwrap();
    }

    pub fn framebuffer(&self) -> &Framebuffer {
        &self.framebuffer
    }

    pub fn draw(
        &self,
        mesh: VertexBuffer,
        vertex_shader: VertexShaderFn,
        frag_shader: FragShaderFn,
    ) {
        self.draw_call_barrier.inc();
        let cmd = DrawCmdPayload {
            vb: mesh,
            vertex_shader: vertex_shader,
            frag_shader: frag_shader,
        };
        self.draw_call_queue.send(DrawCallCmd::Draw(cmd)).unwrap();
    }

    pub fn begin_frame(&mut self) {
        self.framebuffer.clear()
    }

    pub fn end_frame(&mut self) {
        if !self
            .draw_call_barrier
            .wait_timeout(std::time::Duration::from_millis(1000))
        {
            panic!("dc barrier timeout");
        }
        if !self
            .tile_scheduler
            .wait_timeout(std::time::Duration::from_millis(1000))
        {
            panic!("ts barrier timeout");
        }
    }
}
