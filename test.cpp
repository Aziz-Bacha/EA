#include <iostream>
#include <string>

void printSubstrings(const std::string& str) {
    for (size_t len = 1; len <= str.length(); ++len) {
        for (size_t start = 0; start <= str.length() - len; ++start) {
            std::string substring = str.substr(start, len);
            std::cout << substring << std::endl;
        }
    }
}

int main() {
    std::string str;
    std::cout << "Enter a string  : ";
    printSubstrings("hello world");
    return 0;
}
