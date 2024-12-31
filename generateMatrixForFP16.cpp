#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>  // 用于 std::hex 和 std::setfill
#include <cstring>

// g++ -o generate_matrices generateMatrixFile.cpp 
// ./generate_matrices 16 8 16

void generateMatrixFile(const std::string &filename, size_t rows, size_t cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file: " + filename);
    }

    std::random_device rd;
    std::default_random_engine engine{rd()};
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f); // 随机数范围 [-1, 1]

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            float value = uniform(engine);
            // uint32_t hexValue;
            // std::memcpy(&hexValue, &value, sizeof(float));  // 将 float 转换为 uint32_t
            uint32_t hexValue = 0xbc00;

            file << "0x" << std::setfill('0') << std::setw(4) << std::hex << hexValue;  // 转换为十六进制并写入
            // file << std::hex << std::setw(8) << std::setfill('0') << hexValue;
            if (j < cols - 1) {
                file << ","; // 不是最后一个元素就添加逗号
            }
        }
        file << "\n"; // 每一行结束后换行
    }

    file.close();
}

int main(int argc, char* argv[]) {
    int M  = std::atoi(argv[1]);
    int N  = std::atoi(argv[2]);
    int K  = std::atoi(argv[3]);
    try {
        generateMatrixFile("matrix_A.txt", M, K);
        generateMatrixFile("matrix_B.txt", K, N);
        std::cout << "Matrices A and B generated successfully!" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
    }

    return 0;
}
