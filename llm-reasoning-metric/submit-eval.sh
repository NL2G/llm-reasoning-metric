TO_COMPUTES=(
    0
    1
    2
)
LANGUAGE_PAIRS=(
    "en-de"
    "en-es"
    "ja-zh"
)

for idx in ${TO_COMPUTES[@]}; do
    for lp in ${LANGUAGE_PAIRS[@]}; do
        echo "TO_COMPUTE=$idx LANGUAGE_PAIR=$lp"
        TO_COMPUTE=$idx LANGUAGE_PAIR=$lp sbatch llm-reasoning-metric/eval.sh
    done
done