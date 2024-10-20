@echo off
call scripts\build.bat > dupa
del dupa
cd build
ctest -C Debug -T test
rmdir /s /q Testing
cd ..