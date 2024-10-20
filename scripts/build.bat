@echo off
rmdir /s /q build
mkdir build
cmake -S. -B./build -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE --no-warn-unused-cli
cmake --build ./build --config Debug --target ALL_BUILD -j 10
cmake --build ./build --config Release --target ALL_BUILD -j 10
copy .\build\Release\benchmarking_tool.exe .\benchmarking_tool.exe /y
copy .\build\Debug\benchmarking_tool.exe .\debug_benchmarking_tool.exe /y