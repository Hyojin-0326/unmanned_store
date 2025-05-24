#!/bin/bash

# 경로 설정
SRC_IMG="/home/aistore02/Datasets/project_aistore/images/train"
SRC_LABEL="/home/aistore02/Datasets/project_aistore/labels/train"

DEST_IMG="/home/aistore02/Datasets/project_aistore/images/val"
DEST_LABEL="/home/aistore02/Datasets/project_aistore/labels/val"

LOG_FILE="./move_val_log.txt"
> "$LOG_FILE"  # 기존 로그 비우기

# 총 파일 수 계산
TOTAL=$(find "$SRC_IMG" -type f -name "*.png" | wc -l)
VAL_COUNT=$((TOTAL / 10))

echo "총 이미지 수: $TOTAL"
echo "옮길 val 이미지 수: $VAL_COUNT"
echo "[START] $(date)" >> "$LOG_FILE"
echo "총 이미지 수: $TOTAL" >> "$LOG_FILE"
echo "옮길 이미지 수: $VAL_COUNT" >> "$LOG_FILE"

# 랜덤 파일 선택
FILES=$(find "$SRC_IMG" -type f -name "*.png" | shuf -n "$VAL_COUNT")

SUCCESS_COUNT=0
FAIL_COUNT=0

for IMG_PATH in $FILES; do
    IMG_FILE=$(basename "$IMG_PATH")
    LABEL_FILE="${IMG_FILE%.png}.txt"

    echo "-> $IMG_FILE" >> "$LOG_FILE"

    # move image
    if mv "$SRC_IMG/$IMG_FILE" "$DEST_IMG/"; then
        echo "  [OK] 이미지 이동 완료"
    else
        echo "  [FAIL] 이미지 이동 실패"
        echo "  [FAIL] 이미지 이동 실패" >> "$LOG_FILE"
        ((FAIL_COUNT++))
        continue
    fi

    # move label
    if [ -f "$SRC_LABEL/$LABEL_FILE" ]; then
        mv "$SRC_LABEL/$LABEL_FILE" "$DEST_LABEL/"
        echo "  [OK] 레이블도 이동 완료" >> "$LOG_FILE"
        ((SUCCESS_COUNT++))
    else
        echo "  [WARN] 레이블 없음: $LABEL_FILE" >> "$LOG_FILE"
        ((FAIL_COUNT++))
    fi
done

echo "[END] $(date)" >> "$LOG_FILE"
echo "성공: $SUCCESS_COUNT개, 실패 또는 레이블 누락: $FAIL_COUNT개" | tee -a "$LOG_FILE"

echo "이동 로그: $LOG_FILE"
