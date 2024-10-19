mkdir build
cmake -S. -B./build
cmake --build ./build --config Debug --target benchmarking_tool -j 10
copy .\build\Debug\benchmarking_tool.exe .\debug_benchmarking_tool.exe