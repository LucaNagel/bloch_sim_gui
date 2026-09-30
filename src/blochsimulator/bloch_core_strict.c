/*
 * Separate translation unit for the strict ordinary Bloch reference build.
 * Keeping a distinct source name is deliberate: setuptools otherwise gives
 * the strict and legacy extensions the same object path and may silently
 * reuse whichever compiler variant was built first.
 */
#include "bloch_core_modified.c"
