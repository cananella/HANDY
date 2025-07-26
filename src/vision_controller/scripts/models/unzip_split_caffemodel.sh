#!/bin/bash

# .zip 확장자를 가진 분할 압축 파일 탐색
find . -type f -name "*.zip" | while read -r zip_file; do
    # 디렉토리와 파일명 추출
    dir_name=$(dirname "$zip_file")
    zip_base=$(basename "$zip_file")
    model_name="${zip_base%.*}.caffemodel"
    model_path="$dir_name/$model_name"

    # 원본 파일이 이미 있는지 확인
    if [ -f "$model_path" ]; then
        echo "✅ 이미 존재함: $model_path → 압축 해제하지 않음"
        continue
    fi

    echo "📦 압축 해제 중: $zip_file → $model_path"

    # 압축 파일이 있는 디렉토리로 이동 후 압축 해제
    (
        cd "$dir_name" || exit 1
        unzip -o "$zip_base"
    )
done

