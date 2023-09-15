#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <sstream>
#include <cstring>

std::string ByteArrayToString(const uint8_t *arr, int size) {
    std::ostringstream convert;

    for (int a = 0; a < size; a++) {
        convert << arr[a];
    }

    return convert.str();
}

void StringToByteArray(std::string s, uint8_t *bytes, int batch_size, int offset, int length){
    std::memcpy(bytes+4+4+4, s.data(), s.length());
	bytes[0] = (batch_size & 0x000000ff);
	bytes[1] = ( batch_size & 0x0000ff00 ) >> 8;
	bytes[2] = ( batch_size & 0x00ff0000 ) >> 16;
	bytes[3] = ( batch_size & 0xff000000 ) >> 24;
	
	bytes[4] = (offset & 0x000000ff);
	bytes[5] = ( offset & 0x0000ff00 ) >> 8;
	bytes[6] = ( offset & 0x00ff0000 ) >> 16;
	bytes[7] = ( offset & 0xff000000 ) >> 24;

	bytes[8] = (length & 0x000000ff);
	bytes[9] = ( length & 0x0000ff00 ) >> 8;
	bytes[10] = ( length & 0x00ff0000 ) >> 16;
	bytes[11] = ( length & 0xff000000 ) >> 24;
}

#endif
