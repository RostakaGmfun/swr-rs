use nalgebra_glm as glm;
use std::fs::File;
use std::io::{self, BufRead};

pub fn load_obj(path: &str) -> io::Result<(Vec<glm::Vec3>, Vec<[u16; 3]>)> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut verts = Vec::new();
    let mut indices = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let mut parts = line.split_whitespace();
        if let Some(tag) = parts.next() {
            match tag {
                "v" => {
                    let x: f32 = parts.next().unwrap().parse().unwrap();
                    let y: f32 = parts.next().unwrap().parse().unwrap();
                    let z: f32 = parts.next().unwrap().parse().unwrap();
                    verts.push(glm::Vec3::new(x, y, z));
                }
                "f" => {
                    let parse_index =
                        |s: &str| s.split('/').next().unwrap().parse::<u16>().unwrap() - 1;
                    let v1 = parse_index(parts.next().unwrap());
                    let v2 = parse_index(parts.next().unwrap());
                    let v3 = parse_index(parts.next().unwrap());
                    indices.push([v1, v2, v3]);
                }
                _ => {}
            }
        }
    }

    Ok((verts, indices))
}
