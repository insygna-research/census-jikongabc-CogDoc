use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

// 分块读取文件计算 SHA256，避免大文件一次性占满内存
pub fn calculate_sha256<P: AsRef<Path>>(path: P) -> std::io::Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break; // 读到文件末尾
        }
        hasher.update(&buffer[..bytes_read]); // 仅喂入本次实际读到的字节
    }

    Ok(hex::encode(hasher.finalize()))
}

// 单个 PDF 的指纹信息(文件名/大小/内容哈希)
pub struct ScannedFileInfo {
    pub name: String,
    pub size: u64,
    pub sha256: String,
}

// 扫描目录下所有 PDF，并行计算指纹，结果按文件名稳定排序返回
pub fn parallel_scan_manifest(doc_dir: &str) -> std::io::Result<Vec<ScannedFileInfo>> {
    let dir_path = Path::new(doc_dir);
    if !dir_path.exists() || !dir_path.is_dir() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "目标目录不存在",
        ));
    }

    let mut pdf_paths = Vec::new();
    for entry in std::fs::read_dir(dir_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file()
            && path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("pdf"))
        {
            pdf_paths.push(path);
        }
    }

    pdf_paths.sort(); // 先定序，保证指纹清单顺序稳定

    // 并行计算各文件哈希，保留下标用于还原排序
    let results: Vec<std::io::Result<(usize, ScannedFileInfo)>> = pdf_paths
        .par_iter()
        .enumerate()
        .map(|(idx, path)| {
            let metadata = std::fs::metadata(path)?;
            let file_name = path.file_name().unwrap().to_string_lossy().into_owned(); // read_dir 产出的文件路径必含文件名
            let file_size = metadata.len();
            let hash_str = calculate_sha256(path)?; // 任一文件哈希失败则整次扫描失败

            Ok((
                idx,
                ScannedFileInfo {
                    name: file_name,
                    size: file_size,
                    sha256: hash_str,
                },
            ))
        })
        .collect();

    let mut valid_files = Vec::with_capacity(results.len());
    for res in results {
        valid_files.push(res?); // 向上抛出任一并行任务的 IO 错误
    }
    valid_files.sort_by_key(|(idx, _)| *idx); // 还原并行前的文件名顺序

    Ok(valid_files.into_iter().map(|(_, file)| file).collect())
}
