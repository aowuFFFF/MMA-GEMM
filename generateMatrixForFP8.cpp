// g++ -o generate_matrices generateMatrixFile.cpp 
// ./generate_matrices 16 8 32
// E5M2 格式结构
// 符号位 (1 位)： 0 表示正数，1 表示负数。
// 指数位 (5 位)： 指数使用偏置编码，偏置值为 15。
// 尾数位 (2 位)： 用于表示小数部分，尾数总是隐含一个 1，即为规格化数。
// 非规格化数不包含隐含的 1。

#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>  // 用于 std::hex

// FP8类型的简单表示：8位
typedef uint8_t fp8_e4m3;  // 使用uint8_t存储FP8类型

// 生成矩阵并将其以FP8十六进制存储
void generateMatrixFile(const std::string &filename, size_t rows, size_t cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file: " + filename);
    }

    std::random_device rd;
    std::default_random_engine engine{rd()};
    //std::uniform_int_distribution<int> uniform(1, 240); 
    std::uniform_int_distribution<int> uniform(-10, -1); 

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            // fp8_e4m3 fp8_value = static_cast<fp8_e4m3>(uniform(engine));  
            // file << "0x" << std::setw(2) << std::setfill('0') << std::hex << (int)fp8_value;  // 以十六进制格式输出
            int value = 0x41; // 设置为十六进制数字 0x38
            file << "0x" << std::setw(2) << std::setfill('0') << std::hex << value ;

            if (j < cols - 1) {
                file << ",";  // 不是最后一个元素就添加逗号
            }
        }
        file << "\n";  // 每一行结束后换行
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
