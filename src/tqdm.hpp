#ifndef TQDM_HPP
#define TQDM_HPP

#include <iostream>
#include <iomanip>

class tqdm {
public:
    tqdm(int total, int width = 40) : total(total), width(width) {
        std::cout << "[";
        print_progress(0);
        std::cout << "]" << std::flush;
    }

    void progress(int current) {
        if (current >= 0 && current <= total) {
            int new_progress = static_cast<int>((current * 100.0) / total);
            if (new_progress != last_progress) {
                print_progress(new_progress);
            }
        }
    }

    void finish() {
        print_progress(100);
        std::cout << std::endl;
    }

private:
    void print_progress(int progress) {
        int pos = (width * progress) / 100;
        std::cout << "\r[";
        for (int i = 0; i < width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << progress << "%" << std::flush;
        last_progress = progress;
    }

    int total;
    int width;
    int last_progress;
};

#endif // TQDM_HPP