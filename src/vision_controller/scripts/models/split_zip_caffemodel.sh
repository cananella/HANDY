#!/bin/bash

# 분할 크기 (45MB = 45 * 1024 KB = 46080 KB)
SPLIT_SIZE=46080

# 현재 디렉토리 이하의 .caffemodel 파일 탐색
find . -type f -name "*.caffemodel" | while read -r file; do
    # 디렉토리와 파일명 분리
    dir_name=$(dirname "$file")
    base_name=$(basename "$file")
    zip_name="${base_name%.*}.zip"
    zip_path="$dir_name/$zip_name"

    echo "압축 중: $file → $zip_path (분할 크기: ${SPLIT_SIZE}KB)"

    # 해당 디렉토리로 이동해서 zip 실행 (압축 파일이 그 위치에 생기도록)
    (
        cd "$dir_name" || exit
        zip -s ${SPLIT_SIZE}k "$zip_name" "$base_name"
    )
done

