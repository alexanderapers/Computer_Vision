#pragma once
#include "../../precomp.h"

/// <summary>
/// Logs a string input.
/// </summary>
inline void Log(String input)
{
    cout << input << endl;
}

/// <summary>
/// Returns the current date time in string format.
/// </summary>
inline String getCurrentDateTime()
{
    time_t t = time(0);
    struct tm now;
    localtime_s(&now, &t);
    char buffer[80];
    strftime(buffer, 80, "%Y-%m-%d-%H-%M-%S", &now);

    return buffer;
}