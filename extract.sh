#!/bin/bash
set -e  # 遇到错误立即退出
set -o pipefail

DATA_DIR="GenVideo-100K"
cd "$DATA_DIR" || { echo "Error: Directory $DATA_DIR not found."; exit 1; }

echo "📦 Starting to extract GenVideo datasets in $DATA_DIR ..."
mkdir -p extracted

# --- 解压单文件 tar.gz ---
for f in *.tar.gz; do
    if [ -f "$f" ]; then
        echo "🧩 Extracting $f ..."
        tar -xzf "$f" -C extracted/
    fi
done

# --- 合并并解压分片数据 (I2VGEN_XL / Real 等) ---
merge_and_extract() {
    PREFIX=$1
    OUTPUT="${PREFIX}.tar.gz"

    PARTS=$(ls ${PREFIX}_part_* 2>/dev/null | wc -l)
    if [ "$PARTS" -gt 0 ]; then
        echo "🧩 Merging ${PARTS} parts for $PREFIX ..."
        cat ${PREFIX}_part_* > "$OUTPUT"
        echo "📂 Extracting merged archive $OUTPUT ..."
        tar -xzf "$OUTPUT" -C extracted/
        rm "$OUTPUT"  # 可选：合并后删除中间文件
    fi
}

merge_and_extract "I2VGEN_XL"
merge_and_extract "Real"

echo "✅ All datasets extracted successfully to $DATA_DIR/extracted/"
