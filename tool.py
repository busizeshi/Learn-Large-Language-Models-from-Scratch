from pathlib import Path

from pypdf import PdfReader, PdfWriter


def extract_pdf_pages(src_path, start_page, end_page, dst_path=None):
    """从 PDF 文件中截取 [start_page, end_page] 区间的页面，生成新 PDF。

    Args:
        src_path: 源 PDF 文件路径。
        start_page: 起始页（从 1 开始计数，包含）。
        end_page: 结束页（从 1 开始计数，包含）。
        dst_path: 输出 PDF 路径。不传时默认在源文件同目录生成
                  {源文件名}_p{start}-{end}.pdf。
    """
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(f"源文件不存在: {src}")

    if start_page < 1 or end_page < start_page:
        raise ValueError(f"非法的页码范围: start_page={start_page}, end_page={end_page}")

    reader = PdfReader(src)
    total = len(reader.pages)
    if end_page > total:
        raise ValueError(f"结束页 {end_page} 超出文档总页数 {total}")

    writer = PdfWriter()
    # 页码从 1 开始计数，转换为内部 0-based 索引
    for idx in range(start_page - 1, end_page):
        writer.add_page(reader.pages[idx])

    dst = Path(dst_path) if dst_path else src.with_name(f"{src.stem}_p{start_page}-{end_page}.pdf")
    with dst.open("wb") as f:
        writer.write(f)
    print(f"已截取第 {start_page}~{end_page} 页（共 {end_page - start_page + 1} 页）→ {dst}")
    return str(dst)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="从 PDF 中截取指定起止页")
    parser.add_argument("src", nargs="?", default=r"D:\dev\src\流畅的Python.pdf", help="源 PDF 文件路径")
    parser.add_argument("start", nargs="?", default=51, type=int, help="起始页（从 1 开始）")
    parser.add_argument("end", nargs="?", default=64, type=int, help="结束页（从 1 开始）")
    parser.add_argument("-o", "--out", default="pdf1.pdf", help="输出 PDF 路径（可选）")
    args = parser.parse_args()

    extract_pdf_pages(args.src, args.start, args.end, args.out)
