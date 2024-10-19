mkdir build
cmake -S. -B./build
cmake --build ./build --config Release --target benchmarking_tool -j 10
copy .\build\Release\benchmarking_tool.exe .\benchmarking_tool.exe