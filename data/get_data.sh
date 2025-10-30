huggingface-cli download monology/pile-uncopyrighted --repo-type dataset --local-dir ./pile-uncopyrighted
cd pile_uncopyrighted/train
for i in $(seq -w 0 29); do
(
    unzstd ${i}.jsonl.zst
    python ../../data/process.py ${i}
) &
done
wait
