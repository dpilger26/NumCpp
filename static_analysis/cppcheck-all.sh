cd build
rm -rf *
cmake -DBUILD_CPPCHECK_TEST=ON ..
cppcheck --project=compile_commands.json --enable=all --std=c++17 --error-exitcode=2 --inline-suppr --suppressions-list=../suppressions.txt --suppress=missingIncludeSystem --suppress=missingInclude
cd ..