#include <iostream>
#include <vector>
#include <cuda_runtime_api.h>
// #include "manager_device.h"

// device pointer, host pointer, size

struct DataElement {
    void* devicePtr;
    void* hostPtr;
    size_t size;
};

std::vector<DataElement> ptrSizeStore;


void add_to_mem_list(void* devicePtr, size_t size){
    DataElement de;
    de.devicePtr = devicePtr;
    de.hostPtr = nullptr;
    de.size = size;
    ptrSizeStore.push_back(de);
}

void remove_from_mem_list(void* ptrToRemove){
    // Find and remove elements from ptrSizeStore where devicePtr equals ptrToRemove
    for (auto it = ptrSizeStore.begin(); it != ptrSizeStore.end(); ++it) {
        if (it->devicePtr == ptrToRemove) {
            ptrSizeStore.erase(it);
            break; // Exit loop after removing the first matching element
        }
    }
}

void swap_in() {
    // Reserve memory on device
    for (auto& buffer : ptrSizeStore) {
        // allocate size
        cudaMalloc(&buffer.devicePtr, buffer.size);
        cudaMemcpy(buffer.devicePtr, buffer.hostPtr, buffer.size, cudaMemcpyHostToDevice);
    }

    // Wait for transfers to complete
    cudaDeviceSynchronize();
}

void swap_out() {
    for (auto& buffer : ptrSizeStore) {
        void* tempVector;
        cudaHostAlloc(&tempVector, buffer.size, cudaHostAllocDefault);
        buffer.hostPtr = tempVector;
        // Schedule transfers from device to host
        cudaMemcpy(buffer.hostPtr, buffer.devicePtr, buffer.size, cudaMemcpyDeviceToHost);
    }

    // Wait for transfers to complete
    cudaDeviceSynchronize();

    // Delete all user buffers
    for (const auto& buffer : ptrSizeStore) {
        cudaFree(buffer.devicePtr);
    }
}

int main(){
    // Add some elements to ptrSizeStore
    void* devicePtr1 = reinterpret_cast<void*>(0x12345678);
    // void* devicePtr2 = reinterpret_cast<void*>(0xabcdef01);
    
    int hostData[2] = {11, 20};
    cudaMalloc(&devicePtr1, 2 * sizeof(int));
    std::cout << "hostData pointer " << hostData << std::endl;
    cudaMemcpy(devicePtr1, hostData, 2 * sizeof(int), cudaMemcpyHostToDevice);
    
    add_to_mem_list(devicePtr1, 2 * sizeof(int));

    std::cout << "\nptrSizeStore before swap_out:" << std::endl;
    for (const auto& element : ptrSizeStore) {
        std::cout << "devicePtr: " << element.devicePtr << ", size: " << element.size << std::endl;
    }

    swap_out();
    swap_in();

    for (const auto& element: ptrSizeStore) {
        int* hostBuffer = static_cast<int*>(element.hostPtr);
        std::cout << "\nData copied back from device to host: " << hostBuffer << std::endl;
        std::cout << "hostBuffer[0]: " << hostBuffer[0] << std::endl;
        std::cout << "hostBuffer[1]: " << hostBuffer[1] << std::endl;
    }

    remove_from_mem_list(devicePtr1);
    std::cout << "\nptrSizeStore after delete:" << std::endl;
    for (const auto& element : ptrSizeStore) {
        std::cout << "devicePtr: " << element.devicePtr << ", size: " << element.size << std::endl;
    }

    return 0;
}

