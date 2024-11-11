# Usuwanie poprzednich katalogów budowania
rm -rf build_debug build_release

# Tworzenie katalogów budowania dla Debug i Release
mkdir build_debug build_release

# Konfiguracja i budowanie wersji Debug
cmake -S . -B ./build_debug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE
cmake --build ./build_debug -j 10

# Konfiguracja i budowanie wersji Release
cmake -S . -B ./build_release -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=TRUE
cmake --build ./build_release -j 10

# Kopiowanie wynikowych plików wykonywalnych
cp ./build_release/benchmarking_tool ./benchmarking_tool_release
cp ./build_debug/benchmarking_tool ./benchmarking_tool_debug
