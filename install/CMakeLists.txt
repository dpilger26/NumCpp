cmake_minimum_required(VERSION 3.10)

project(NumCpp CXX)

install(
    DIRECTORY
    "../include/"
    DESTINATION
    "/usr/include"
    FILES_MATCHING
    PATTERN
    "*.hpp"
)
