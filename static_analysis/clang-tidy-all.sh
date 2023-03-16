export CLANG_TIDY_FLAGS="-extra-arg=-std=gnu++17 -p ./build/ --config-file=./.clang-tidy --header-filter=./include/*"
# export CLANG_TIDY_FLAGS="--fix --fix-errors --fix-notes -extra-arg=-std=gnu++17 -p ./build/ --config-file=./.clang-tidy --header-filter=./include/*"
export matchFiles=$(find ./include -iregex '.*\.\(cpp\|hpp\)$')
echo ${matchFiles} | xargs -r -n 1 -P $(nproc) clang-tidy ${CLANG_TIDY_FLAGS}