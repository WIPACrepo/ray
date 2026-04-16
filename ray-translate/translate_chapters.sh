#!/bin/bash

CHAPTERS=$1
URL="https://raytest.icecube.aq/predict"
CHUNK_SIZE=5
TOKEN_SIZE=1000

TIMESTAMP=$(date +%s)

for i in $(seq 1 $CHAPTERS); do
  CHAPTER=$(printf "%03d\n" ${i})
  ./client.py --url "$URL" --chunk_size $CHUNK_SIZE --token_size $TOKEN_SIZE mobydick/chapters/chapter_$CHAPTER.txt >>translation_$TIMESTAMP &
done