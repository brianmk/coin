cmake_minimum_required(VERSION 3.15)

project(CoinInstalledPackageSmoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Coin CONFIG REQUIRED)

add_executable(coin-installed-package-smoke main.cpp)
target_link_libraries(coin-installed-package-smoke PRIVATE Coin::Coin)
