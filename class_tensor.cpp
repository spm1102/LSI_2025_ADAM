#ifndef TENSOR_MEM_H
#define TENSOR_MEM_H


#if defined(USE_HLS) || defined(__SYNTHESIS__)
    // --- MÔI TRƯỜNG HLS (Hardware) ---
    #include <hls_stream.h>
    #include <ap_axi_sdata.h>
    #include <ap_int.h>
    #include <ap_fixed.h>

    // Dùng ap_fixed cho phần cứng để tối ưu tài nguyên
    typedef ap_fixed<16, 8> data_t;

    // Alias stream của HLS
    template<typename T>
    using MyStream = hls::stream<T>;

#else
    // --- MÔI TRƯỜNG PURE C++ (Software Simulation) ---
    #include <iostream>
    #include <vector>
    #include <queue>
    #include <cmath>
    #include <cassert>

    // Dùng float cho mô phỏng phần mềm cho nhanh và chính xác
    typedef float data_t;

    // Giả lập hls::stream bằng std::queue
    template<typename T>
    class MyStream {
    private:
        std::queue<T> q;
    public:
        void write(const T& val) { q.push(val); }
        T read() { 
            if(q.empty()) return 0; // Xử lý lỗi cơ bản
            T val = q.front(); 
            q.pop(); 
            return val; 
        }
        bool empty() { return q.empty(); }
        bool full() { return false; } // Queue phần mềm coi như vô hạn
    };

#endif


struct Shape {
    int N; // Batch size
    int H; // Height
    int W; // Width
    int C; // Channels
};

template <typename T>
class TensorMem {
private:
    T* m_data;       
    Shape m_shape;  
    int m_size;      

public:
    // Constructor
    TensorMem(T* data, Shape shape) : m_data(data), m_shape(shape) {
        #if defined(USE_HLS) || defined(__SYNTHESIS__)
        #pragma HLS INLINE
        #endif
        m_size = shape.N * shape.H * shape.W * shape.C;
    }

    // Hàm tính index (Dùng chung cho cả 2 môi trường)
    int get_index(int n, int h, int w, int c) {
        #if defined(USE_HLS) || defined(__SYNTHESIS__)
        #pragma HLS INLINE
        #endif
        return n * (m_shape.H * m_shape.W * m_shape.C) + 
               h * (m_shape.W * m_shape.C) + 
               w * (m_shape.C) + 
               c;
    }

    T read_element(int n, int h, int w, int c) {
        #if defined(USE_HLS) || defined(__SYNTHESIS__)
        #pragma HLS INLINE
        #endif
        int idx = get_index(n, h, w, c);
        return m_data[idx];
    }

    void write_element(int n, int h, int w, int c, T val) {
        #if defined(USE_HLS) || defined(__SYNTHESIS__)
        #pragma HLS INLINE
        #endif
        int idx = get_index(n, h, w, c);
        m_data[idx] = val;
    }

    void load_tile_to_stream(int n, int h_start, int w_start, int h_size, int w_size, MyStream<T>& out_stream) {
        // Vòng lặp tính toán
        for (int h = 0; h < h_size; ++h) {
            for (int w = 0; w < w_size; ++w) {
                for (int c = 0; c < m_shape.C; ++c) {
                    
                    // Chỉ bật PIPELINE khi chạy HLS
                    #if defined(USE_HLS) || defined(__SYNTHESIS__)
                    #pragma HLS PIPELINE II=1
                    #endif

                    // Logic tính toán vị trí bộ nhớ
                    int idx = get_index(n, h_start + h, w_start + w, c);
                    T val = m_data[idx];
                    out_stream.write(val);
                }
            }
        }
    }

    void store_stream_to_mem(int n, int h_start, int w_start, int h_size, int w_size, MyStream<T>& in_stream) {
        for (int h = 0; h < h_size; ++h) {
            for (int w = 0; w < w_size; ++w) {
                for (int c = 0; c < m_shape.C; ++c) {
                    
                    #if defined(USE_HLS) || defined(__SYNTHESIS__)
                    #pragma HLS PIPELINE II=1
                    #endif

                    int idx = get_index(n, h_start + h, w_start + w, c);
                    
                    if (!in_stream.empty()) {
                        T val = in_stream.read();
                        m_data[idx] = val;
                    }
                }
            }
        }
    }
};

#endif 