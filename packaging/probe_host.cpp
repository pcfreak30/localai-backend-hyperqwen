// Exercise relocated C++ headers, compilation and linking; no GPU execution.
#include <iostream>
#include <string>
int main() {
    const std::string message = "bundled C++ toolchain OK";
    std::cout << message << std::endl;
    return 0;
}
